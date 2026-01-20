# recommendations.py
# FleetMate — Recommendations + Reporting Helpers
# Goals:
# - No missing integration: functions are defensive and tolerate missing keys.
# - Cleaner look: consistent formatting + structured outputs.
# - Adds EV charging feasibility (peak/cap), tours/charging time best-effort, and new power chart.
# - Provides "Gemini final answer" structured inputs + persuasive recommendations + outlook.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Generic helpers
# -------------------------

def _get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Safe nested getter. path like 'a.b.c'."""
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

def _num(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float, np.number)):
            return float(x)
        if isinstance(x, str):
            s = x.strip().replace(" ", "").replace(",", ".")
            return float(s)
        return float(x)
    except Exception:
        return default

def _int(x: Any, default: int = 0) -> int:
    try:
        return int(round(_num(x, default)))
    except Exception:
        return default

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _hours_between(start_h: int, end_h: int) -> int:
    """Hours in half-open interval [start, end) with wrap, in integer hours."""
    start_h %= 24
    end_h %= 24
    diff = (end_h - start_h) % 24
    return 24 if diff == 0 else diff

def _in_window(h: int, start_h: int, end_h: int) -> bool:
    """h in [start,end) with wrap."""
    h %= 24
    start_h %= 24
    end_h %= 24
    if start_h == end_h:
        return True
    if start_h < end_h:
        return start_h <= h < end_h
    return (h >= start_h) or (h < end_h)

def _fmt_eur(x: float, digits: int = 0) -> str:
    if digits <= 0:
        return f"{x:,.0f} €".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{x:,.{digits}f} €".replace(",", "X").replace(".", ",").replace("X", ".")

def _fmt_kw(x: float, digits: int = 1) -> str:
    return f"{x:,.{digits}f} kW".replace(",", "X").replace(".", ",").replace("X", ".")

def _fmt_kwh(x: float, digits: int = 0) -> str:
    return f"{x:,.{digits}f} kWh".replace(",", "X").replace(".", ",").replace("X", ".")

def _fmt_pct(x: float, digits: int = 0) -> str:
    return f"{x*100:,.{digits}f}%".replace(",", "X").replace(".", ",").replace("X", ".")


# -------------------------
# Data extraction (robust)
# -------------------------

def _extract_inputs(res: Dict[str, Any]) -> Dict[str, Any]:
    return _get(res, "inputs", {}) or {}

def _extract_capacity(res: Dict[str, Any]) -> Dict[str, Any]:
    return res.get("capacity_analysis", {}) or {}

def _extract_energy_cost(res: Dict[str, Any]) -> Dict[str, Any]:
    return res.get("energy_cost", {}) or {}

def _extract_load(res: Dict[str, Any]) -> Dict[str, Any]:
    return res.get("load", {}) or {}

def _extract_profile(res: Dict[str, Any]) -> Dict[str, Any]:
    return res.get("profile", {}) or {}

def _extract_load_profile_df(res: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Attempts to find a load profile DataFrame in result dict.
    Supported shapes:
      - pd.DataFrame
      - list[dict]
      - dict of lists
    Expected columns: timestamp/datetime + power (kw).
    """
    candidates = [
        _get(res, "profile.load_profile_df"),
        _get(res, "profile.load_profile"),
        _get(res, "profile.load_profile_raw"),
        res.get("load_profile"),
        _get(res, "load.profile"),
        _get(res, "profile.load"),
    ]
    obj = None
    for c in candidates:
        if c is None:
            continue
        obj = c
        break
    if obj is None:
        return None

    try:
        if isinstance(obj, pd.DataFrame):
            df = obj.copy()
        elif isinstance(obj, list):
            df = pd.DataFrame(obj)
        elif isinstance(obj, dict):
            df = pd.DataFrame(obj)
        else:
            return None
    except Exception:
        return None

    # Find datetime column
    dt_col = None
    for col in df.columns:
        lc = str(col).lower()
        if any(k in lc for k in ("timestamp","datetime","date_time","date/time","date","time","datum","zeit","ts")):
            dt_col = col
            break
    if dt_col is None:
        # Try index
        if isinstance(df.index, pd.DatetimeIndex):
            dt_col = None
        else:
            return None

    if dt_col is not None:
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
        df = df.dropna(subset=[dt_col]).set_index(dt_col)

    if not isinstance(df.index, pd.DatetimeIndex):
        return None

    # Find power column
    # Find power column (robust)
    p_col = None
    for col in df.columns:
        lc = str(col).lower()
        norm = "".join(ch for ch in lc if ch.isalnum())  # removes spaces, (), [], etc.
        # accept lots of real-world variants like "Power (kW)", "Leistung[kW]", "kW_total"
        if ("kw" in norm and ("power" in norm or "leistung" in norm or "load" in norm)) or norm in ("kw", "powerkw", "loadkw", "pkw", "leistungkw"):
            p_col = col
            break

    if p_col is None:
        # Fallback: first numeric column
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            p_col = num_cols[0]
        else:
            return None

    out = df[[p_col]].rename(columns={p_col: "kw"}).sort_index()
    # Make sure frequency isn't required; we will resample to hourly mean later.
    out["kw"] = pd.to_numeric(out["kw"], errors="coerce")
    out = out.dropna(subset=["kw"])
    if out.empty:
        return None
    return out


# -------------------------
# Core EV charging feasibility
# -------------------------

@dataclass
class ChargingFeasibility:
    window_hours: int
    kw_limit: float
    baseline_avg_kw_in_window: float
    headroom_kw: float

    ev_trucks: int
    ev_kwh_day_total: float
    ev_kw_avg_needed: float

    feasible_under_cap: bool
    max_trucks_under_cap: int

    extra_peak_kw_if_full: float
    extra_capacity_cost_eur_year: float

    time_needed_at_headroom_h: Optional[float]
    energy_possible_in_window_kwh: float
    tour_completion_ratio: float  # best-effort: 0..1

def compute_charging_feasibility(res: Dict[str, Any]) -> ChargingFeasibility:
    """
    Best-effort computation:
    - Determine charging window (start/end), kVA limit and cos(phi) => kW limit.
    - Compute baseline average kW in window from load profile or capacity_analysis fields.
    - Compute EV daily energy from inputs if available; else from res['load'] if present.
    - Determine if EV charging fits under cap, max trucks under cap, and extra costs if exceeding.
    """
    inputs = _extract_inputs(res)
    cap = _extract_capacity(res)
    ec = _extract_energy_cost(res)
    load = _extract_load(res)

    start_h = _int(_get(inputs, "start_hour", _get(cap, "start_hour", 22)), 22) % 24
    end_h = _int(_get(inputs, "end_hour", _get(cap, "end_hour", 6)), 6) % 24
    window_hours = _hours_between(start_h, end_h)

    kva_limit = _num(_get(inputs, "kva_limit_kva", _get(cap, "kva_limit_kva", _get(load, "kva_limit_kva", 0.0))), 0.0)
    cos_phi = _num(_get(inputs, "cos_phi", _get(cap, "cos_phi", _get(load, "cos_phi", 0.95))), 0.95)
    cos_phi = _clamp(cos_phi, 0.5, 1.0)
    kw_limit = _num(_get(cap, "kw_limit", 0.0), 0.0)
    if kw_limit <= 0 and kva_limit > 0:
        kw_limit = kva_limit * cos_phi

    # Baseline avg power in window (kW)
    baseline_avg_kw_in_window = _num(_get(cap, "baseline_avg_kw_in_window", 0.0), 0.0)

    lp = _extract_load_profile_df(res)
    if baseline_avg_kw_in_window <= 0 and lp is not None:
        # Hourly mean by hour-of-day over available data
        hourly = lp["kw"].resample("1H").mean().dropna()
        if not hourly.empty:
            by_h = hourly.groupby(hourly.index.hour).mean()
            vals = [float(by_h.get(h, np.nan)) for h in range(24)]
            # average over window hours
            hs = [h for h in range(24) if _in_window(h, start_h, end_h)]
            w_vals = [vals[h] for h in hs if not np.isnan(vals[h])]
            if w_vals:
                baseline_avg_kw_in_window = float(np.mean(w_vals))

    # If still missing, try other fields
    if baseline_avg_kw_in_window <= 0:
        baseline_kwh_window_day = _num(_get(cap, "baseline_kwh_in_window_per_day", 0.0), 0.0)
        if baseline_kwh_window_day > 0 and window_hours > 0:
            baseline_avg_kw_in_window = baseline_kwh_window_day / window_hours

    # Headroom under cap
    headroom_kw = _num(_get(cap, "headroom_kw", 0.0), 0.0)
    if headroom_kw <= 0 and kw_limit > 0:
        headroom_kw = max(0.0, kw_limit - baseline_avg_kw_in_window)

    # EV energy/day
    # --- EV trucks (support aliases from UI) ---
    ev_trucks = _int(
        _get(inputs, "ev_truck_count",
            _get(inputs, "ev_trucks",
                _get(inputs, "num_ev_trucks",
                    _get(load, "ev_truck_count",
                            _get(load, "ev_trucks", 0))))),
        0
    )

    # --- km per truck per day (support aliases) ---
    km_per_truck_day = _num(
        _get(inputs, "km_per_truck_day",
            _get(inputs, "km_per_truck_per_day",
                _get(inputs, "km_per_truck_per_day_km",
                    _get(load, "km_per_truck_day",
                            _get(load, "km_per_truck_per_day", 0.0))))),
        0.0
    )


    # consumption: support multiple keys
    ev_kwh_per_km = _num(
        _get(inputs, "ev_kwh_per_km",
             _get(inputs, "ev_consumption_kwh_per_km",
                  _get(inputs, "ev_kwh_per_km_avg", _get(load, "ev_kwh_per_km", 0.0)))),
        0.0
    )
    if ev_kwh_per_km <= 0:
        # Fallback if user stores "kWh per 100km"
        ev_kwh_per_100km = _num(_get(inputs, "ev_kwh_per_100km", 0.0), 0.0)
        if ev_kwh_per_100km > 0:
            ev_kwh_per_km = ev_kwh_per_100km / 100.0

    ev_kwh_day_total = _num(_get(load, "ev_kwh_day_total", 0.0), 0.0)
    if ev_kwh_day_total <= 0 and ev_trucks > 0 and km_per_truck_day > 0 and ev_kwh_per_km > 0:
        ev_kwh_day_total = ev_trucks * km_per_truck_day * ev_kwh_per_km

    # required average charging power during window
    ev_kw_avg_needed = (ev_kwh_day_total / window_hours) if window_hours > 0 else 0.0

    feasible_under_cap = (ev_kw_avg_needed <= headroom_kw + 1e-9) if (headroom_kw > 0) else (ev_kwh_day_total <= 0)

    max_trucks_under_cap = 0
    if ev_trucks > 0 and km_per_truck_day > 0 and ev_kwh_per_km > 0 and window_hours > 0 and headroom_kw > 0:
        per_truck_kwh_day = km_per_truck_day * ev_kwh_per_km
        max_trucks_under_cap = int(math.floor((headroom_kw * window_hours) / max(1e-9, per_truck_kwh_day)))
        max_trucks_under_cap = max(0, min(ev_trucks, max_trucks_under_cap))

    extra_peak_kw_if_full = 0.0
    if not feasible_under_cap:
        extra_peak_kw_if_full = max(0.0, ev_kw_avg_needed - headroom_kw)

    # Capacity cost estimate
    # We interpret "grid power price" as EUR/kW/year (as in your UI pill).
    power_price_eur_kw_year = _num(
        _get(ec, "grid_power_eur_kw_year", _get(ec, "power_eur_kw_year", _get(ec, "net_power_eur_kw_year", 0.0))),
        0.0
    )
    extra_capacity_cost_eur_year = extra_peak_kw_if_full * power_price_eur_kw_year

    # Charging time needed if we only use headroom (kW)
    time_needed_at_headroom_h = None
    if headroom_kw > 0 and ev_kwh_day_total > 0:
        time_needed_at_headroom_h = ev_kwh_day_total / headroom_kw

    energy_possible_in_window_kwh = headroom_kw * window_hours
    tour_completion_ratio = 1.0
    if ev_kwh_day_total > 0:
        tour_completion_ratio = _clamp(energy_possible_in_window_kwh / ev_kwh_day_total, 0.0, 1.0)

    return ChargingFeasibility(
        window_hours=window_hours,
        kw_limit=kw_limit,
        baseline_avg_kw_in_window=baseline_avg_kw_in_window,
        headroom_kw=headroom_kw,
        ev_trucks=ev_trucks,
        ev_kwh_day_total=ev_kwh_day_total,
        ev_kw_avg_needed=ev_kw_avg_needed,
        feasible_under_cap=feasible_under_cap,
        max_trucks_under_cap=max_trucks_under_cap,
        extra_peak_kw_if_full=extra_peak_kw_if_full,
        extra_capacity_cost_eur_year=extra_capacity_cost_eur_year,
        time_needed_at_headroom_h=time_needed_at_headroom_h,
        energy_possible_in_window_kwh=energy_possible_in_window_kwh,
        tour_completion_ratio=tour_completion_ratio,
    )


# -------------------------
# Issues + solution set (used in report)
# -------------------------

def detect_issues(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Returns a list of issue dicts. Keep stable signature (used by main app).
    """
    issues: List[Dict[str, Any]] = []

    cf = compute_charging_feasibility(res)

    if cf.ev_trucks > 0 and cf.ev_kwh_day_total <= 0:
        issues.append({
            "id": "EV_ENERGY_MISSING",
            "severity": "medium",
            "title": "EV-Ladeenergie nicht ableitbar",
            "detail": "Es fehlen ausreichende Inputs (km/Tag, Verbrauch), um die tägliche Ladeenergie der EV-Flotte robust zu berechnen.",
        })

    if cf.kw_limit > 0 and cf.headroom_kw <= 0:
        issues.append({
            "id": "NO_HEADROOM",
            "severity": "high",
            "title": "Kein Headroom im Ladefenster",
            "detail": "Im Ladefenster bleibt unter dem kW-Limit kein Spielraum. EV-Laden würde den Peak direkt erhöhen oder muss zeitlich verlagert werden.",
            "kw_limit": cf.kw_limit,
            "baseline_avg_kw_in_window": cf.baseline_avg_kw_in_window,
        })

    if (cf.ev_trucks > 0 and cf.ev_kwh_day_total > 0) and (not cf.feasible_under_cap):
        issues.append({
            "id": "EV_CHARGING_EXCEEDS_CAP",
            "severity": "high",
            "title": "EV-Laden passt nicht unter den Peak/Cap",
            "detail": "Für alle EV-Trucks reicht das Ladefenster unter dem kW-Limit nicht aus. Entweder Peak erhöhen oder Lade-/Betriebsstrategie ändern.",
            "ev_kw_avg_needed": cf.ev_kw_avg_needed,
            "headroom_kw": cf.headroom_kw,
            "extra_peak_kw": cf.extra_peak_kw_if_full,
            "max_trucks_under_cap": cf.max_trucks_under_cap,
        })

    # Optional: sanity checks for energy price inputs
    ec = _extract_energy_cost(res)
    total_price = _num(_get(ec, "total_price_eur_kwh", _get(ec, "total_eur_per_kwh", 0.0)), 0.0)
    if total_price <= 0:
        issues.append({
            "id": "ENERGY_PRICE_MISSING",
            "severity": "low",
            "title": "Strompreis unvollständig",
            "detail": "Es ist kein belastbarer Gesamtstrompreis (€/kWh) im Resultat abgelegt. Für Ladekosten und Finance-Teil wird ein Default/Fallback genutzt.",
        })

    return issues


def generate_solution_set(res: Dict[str, Any], issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Returns actionable solutions; keep stable signature (used by main app).
    """
    cf = compute_charging_feasibility(res)
    ec = _extract_energy_cost(res)

    power_price_eur_kw_year = _num(
        _get(ec, "grid_power_eur_kw_year", _get(ec, "power_eur_kw_year", _get(ec, "net_power_eur_kw_year", 0.0))),
        0.0
    )

    solutions: List[Dict[str, Any]] = []

    # 1) Stay under cap
    if cf.ev_trucks > 0:
        solutions.append({
            "id": "STAY_UNDER_CAP",
            "title": "Unter Cap bleiben (Peak stabil halten)",
            "what": "EV-Laden so steuern, dass das kW-Limit nicht überschritten wird.",
            "how": [
                f"Max. {cf.max_trucks_under_cap} von {cf.ev_trucks} Trucks vollständig im Ladefenster laden (bei gleichmäßigem Laden).",
                "Restliche Energie zeitlich verlagern (z. B. Vorladen, Midday-Top-up, oder Ladefenster erweitern).",
                "Lastmanagement/Charging-Controller einsetzen (Prioritäten, Ramp-up, harte Peak-Kappung).",
            ],
            "tradeoff": "Peak bleibt stabil, aber nicht jeder Truck ist rein im Ladefenster vollständig ladbar.",
            "recommended": True if any(i["id"] == "EV_CHARGING_EXCEEDS_CAP" for i in issues) else False
        })

    # 2) Exceed cap (pay more)
    if (cf.ev_trucks > 0 and not cf.feasible_under_cap and cf.extra_peak_kw_if_full > 0):
        solutions.append({
            "id": "ALLOW_PEAK_INCREASE",
            "title": "Peak bewusst erhöhen (aber transparent bepreisen)",
            "what": "Alle Trucks im Ladefenster laden und dafür den Peak/Cap anheben.",
            "how": [
                f"Zusätzlicher Peakbedarf ca. { _fmt_kw(cf.extra_peak_kw_if_full, 1) } (Ø über das Fenster).",
                f"Grobe Mehrkosten über Leistungsentgelt: { _fmt_eur(cf.extra_capacity_cost_eur_year, 0) }/Jahr "
                f"(bei {power_price_eur_kw_year:.1f} €/kW/a).",
                "Prüfen: Netzvertrag, Messkonzept, mögliche Tarif-/Netzebenen-Effekte.",
            ],
            "tradeoff": "Operativ simpel, finanziell/vertraglich potenziell teurer.",
            "recommended": False
        })

    # 3) Extend window / shift operations
    solutions.append({
        "id": "EXTEND_OR_SHIFT_WINDOW",
        "title": "Ladefenster & Betrieb feinjustieren",
        "what": "Mehr Zeit oder andere Zeitpunkte nutzen, damit Laden ohne Peak-Anhebung aufgeht.",
        "how": [
            "Ladefenster um 1–3 Stunden erweitern (wenn operativ möglich).",
            "Tour-Start/Ende leicht verschieben, um mehr Off-Peak zu nutzen.",
            "Schnellladen vermeiden; lieber gleichmäßig (netzdienlich) laden.",
        ],
        "tradeoff": "Braucht Abstimmung mit Betrieb/Schichtplanung, kann aber Cap sparen.",
        "recommended": True
    })

    # 4) Infrastructure upgrade / on-site flexibility
    solutions.append({
        "id": "FLEXIBILITY_OPTIONS",
        "title": "Flexibilität ergänzen (Batterie/PV/Trafo/Anschluss)",
        "what": "Peak reduzieren oder Energie bereitstellen, ohne Netzpeak hochzuziehen.",
        "how": [
            "Stationäre Batterie als Peak-Shaver / Puffer.",
            "PV + intelligentes Laden (Eigenverbrauch maximieren).",
            "Netzanschluss/Trafoleistung prüfen und ggf. upgraden (Cap erhöhen, aber sauber geplant).",
        ],
        "tradeoff": "CAPEX, dafür langfristig stabilere Kosten und Skalierung.",
        "recommended": False
    })

    return solutions


# -------------------------
# New chart: baseline vs new (EV) power curve
# -------------------------

def _hourly_baseline_by_hour(res: Dict[str, Any]) -> np.ndarray:
    """
    Returns 24 values (kW) baseline by hour-of-day.
    Uses load profile if present; else zeros.
    """
    lp = _extract_load_profile_df(res)
    if lp is None:
        # Fallback: if we at least know a baseline average in the window, use it as flat baseline
        cap = _extract_capacity(res)
        base_avg = _num(_get(cap, "baseline_avg_kw_in_window", 0.0), 0.0)
        if base_avg > 0:
            return np.full(24, base_avg, dtype=float)
        return np.zeros(24, dtype=float)

    if lp is None:
        return np.zeros(24, dtype=float)

    hourly = lp["kw"].resample("1H").mean().dropna()
    if hourly.empty:
        return np.zeros(24, dtype=float)

    by_h = hourly.groupby(hourly.index.hour).mean()
    vals = np.array([float(by_h.get(h, 0.0)) for h in range(24)], dtype=float)
    return vals


def plot_power_consumption_comparison(
    res: Dict[str, Any],
    *,
    title: str = "Power consumption: baseline vs. EV charging",
) -> plt.Figure:
    """
    Chart requirements implemented:
    - Compare baseline curve (from load profile hourly mean) vs new curve (baseline + EV charging).
    - Show window slice: 2h before start to 2h after end.
    - Show: peak in window, peak overall (within plotted slice), kW limit (kVA*cos), baseline avg in window.
    - Describe via legend; annotations use each line's color for clarity.
    """
    inputs = _extract_inputs(res)
    cap = _extract_capacity(res)

    start_h = _int(_get(inputs, "start_hour", _get(cap, "start_hour", 22)), 22) % 24
    end_h = _int(_get(inputs, "end_hour", _get(cap, "end_hour", 6)), 6) % 24
    window_hours = _hours_between(start_h, end_h)

    cf = compute_charging_feasibility(res)

    base24 = _hourly_baseline_by_hour(res)

    # EV charging model in chart: uniform avg power during window (best-effort)
    ev_add_kw = 0.0
    if cf.ev_kwh_day_total > 0 and window_hours > 0:
        ev_add_kw = cf.ev_kwh_day_total / window_hours

    new24 = base24.copy()
    for h in range(24):
        if _in_window(h, start_h, end_h):
            new24[h] += ev_add_kw

    # Display hours: 2h before start to 2h after end
    # length = window_hours + 4
    display_len = window_hours + 4
    h0 = (start_h - 2) % 24
    hours = [(h0 + i) % 24 for i in range(display_len)]
    x = np.arange(display_len)

    y_base = np.array([base24[h] for h in hours], dtype=float)
    y_new = np.array([new24[h] for h in hours], dtype=float)

    fig, ax = plt.subplots()
    l1, = ax.plot(x, y_base, label="Baseline (Load profile)")
    l2, = ax.plot(x, y_new, label="New (Baseline + EV charging)")

    # Shade charging window inside plotted slice
    # window indices inside slice:
    win_idx = [i for i, h in enumerate(hours) if _in_window(h, start_h, end_h)]
    if win_idx:
        ax.axvspan(min(win_idx), max(win_idx) + 0.0001, alpha=0.15, label="Charging window")

    # kW limit
    if cf.kw_limit > 0:
        l3 = ax.axhline(cf.kw_limit, linestyle="--", label=f"kW limit (kVA·cosφ)")
        # annotation with same color
        ax.text(
            x[-1], cf.kw_limit,
            f"  Limit: {_fmt_kw(cf.kw_limit, 1)}",
            va="center",
            ha="left",
            color=l3.get_color(),
        )

    # baseline avg in window
    if cf.baseline_avg_kw_in_window > 0:
        l4 = ax.axhline(cf.baseline_avg_kw_in_window, linestyle=":", label="Baseline avg (in window)")
        ax.text(
            x[-1], cf.baseline_avg_kw_in_window,
            f"  Ø Window baseline: {_fmt_kw(cf.baseline_avg_kw_in_window, 1)}",
            va="center",
            ha="left",
            color=l4.get_color(),
        )

    # Peaks
    # overall peaks (within plotted slice)
    base_peak = float(np.max(y_base)) if len(y_base) else 0.0
    new_peak = float(np.max(y_new)) if len(y_new) else 0.0
    i_base_peak = int(np.argmax(y_base)) if len(y_base) else 0
    i_new_peak = int(np.argmax(y_new)) if len(y_new) else 0

    ax.scatter([i_base_peak], [base_peak])
    ax.scatter([i_new_peak], [new_peak])

    ax.text(i_base_peak, base_peak, f"  Peak baseline: {_fmt_kw(base_peak, 1)}", color=l1.get_color(), va="bottom")
    ax.text(i_new_peak, new_peak, f"  Peak new: {_fmt_kw(new_peak, 1)}", color=l2.get_color(), va="bottom")

    # peak in charging window (within slice)
    if win_idx:
        base_win_peak = float(np.max(y_base[win_idx]))
        new_win_peak = float(np.max(y_new[win_idx]))
        i_b = win_idx[int(np.argmax(y_base[win_idx]))]
        i_n = win_idx[int(np.argmax(y_new[win_idx]))]
        ax.scatter([i_b], [base_win_peak], marker="x")
        ax.scatter([i_n], [new_win_peak], marker="x")
        ax.text(i_b, base_win_peak, f"  Peak in window (base): {_fmt_kw(base_win_peak, 1)}",
                color=l1.get_color(), va="top")
        ax.text(i_n, new_win_peak, f"  Peak in window (new): {_fmt_kw(new_win_peak, 1)}",
                color=l2.get_color(), va="top")

    # X labels as time-of-day
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h:02d}:00" for h in hours], rotation=0)
    ax.set_xlabel("Time (2h before start → 2h after end)")
    ax.set_ylabel("kW")
    ax.set_title(title)
    ax.legend(loc="best")

    fig.tight_layout()
    return fig


# -------------------------
# Yearly energy consumption (EV + facility best-effort)
# -------------------------

def compute_ev_energy_year_kwh(res: Dict[str, Any], days_per_year: int = 365) -> float:
    cf = compute_charging_feasibility(res)
    return max(0.0, cf.ev_kwh_day_total * float(days_per_year))

def compute_facility_energy_year_kwh(res: Dict[str, Any], days_per_year: int = 365) -> float:
    base24 = _hourly_baseline_by_hour(res)
    kwh_day = float(np.sum(base24))  # kW * 1h
    return max(0.0, kwh_day * float(days_per_year))

def plot_energy_year(res: Dict[str, Any], *, title: str = "Annual energy consumption") -> plt.Figure:
    ev_y = compute_ev_energy_year_kwh(res, 365)
    fac_y = compute_facility_energy_year_kwh(res, 365)

    labels = ["Facility (baseline)", "EV charging (trucks)"]
    vals = [fac_y, ev_y]

    fig, ax = plt.subplots()
    ax.bar(labels, vals)
    ax.set_ylabel("kWh/year")
    ax.set_title(title)

    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:,.0f}".replace(",", "X").replace(".", ",").replace("X", "."), ha="center", va="bottom")

    fig.tight_layout()
    return fig


# -------------------------
# Textual recommendations (more marketing) + outlook + ZYNE modules
# -------------------------

def build_recommendation_markdown(res: Dict[str, Any], lang: Optional[str] = None) -> str:
    """
    Produces a single markdown block that reads "clean" in Streamlit.
    Language: DE/EN (controlled via `lang`).
    """
    # --- language resolution ---
    lang = (lang or (res.get("inputs") or {}).get("lang") or "DE").upper()
    if lang not in ("DE", "EN"):
        lang = "DE"

    def L(de: str, en: str) -> str:
        return en if lang == "EN" else de

    def _pick(v: Any) -> str:
        """Allow solution fields to be str or {DE/EN} dicts."""
        if isinstance(v, dict):
            return str(v.get(lang) or v.get("EN") or v.get("DE") or "").strip()
        return str(v or "").strip()

    cf = compute_charging_feasibility(res)
    issues = detect_issues(res)
    solutions = generate_solution_set(res, issues)

    ec = _extract_energy_cost(res)
    total_price = _num(_get(ec, "total_price_eur_kwh", _get(ec, "total_eur_per_kwh", 0.0)), 0.0)

    ev_charge_cost_day = cf.ev_kwh_day_total * (total_price if total_price > 0 else 0.0)
    ev_charge_cost_year = ev_charge_cost_day * 365.0

    lines: List[str] = []

    # Headline
    lines.append(L("### Empfehlungen (kompakt, entscheidungsorientiert)",
                   "### Recommendations (concise, decision-oriented)"))

    # Core facts
    lines.append(L("**Kernzahlen (Laden & Peak):**",
                   "**Key numbers (charging & peak):**"))
    lines.append(f"- {L('Ladefenster', 'Charging window')}: **{cf.window_hours} h**")

    if cf.kw_limit > 0:
        lines.append(f"- {L('kW-Limit (kVA·cosφ)', 'kW limit (kVA·cosφ)')}: **{_fmt_kw(cf.kw_limit, 1)}**")
    if cf.baseline_avg_kw_in_window > 0:
        lines.append(f"- {L('Baseline im Ladefenster (Ø)', 'Baseline in window (avg)')}: **{_fmt_kw(cf.baseline_avg_kw_in_window, 1)}**")

    lines.append(f"- {L('Headroom im Ladefenster', 'Headroom in window')}: **{_fmt_kw(cf.headroom_kw, 1)}**")

    if cf.ev_trucks > 0:
        lines.append(
            f"- {L('EV-Flotte', 'EV fleet')}: **{cf.ev_trucks} {L('Trucks', 'trucks')}**, "
            f"{L('tägliche Ladeenergie', 'daily charging energy')}: **{_fmt_kwh(cf.ev_kwh_day_total, 0)}**"
        )
        lines.append(
            f"- {L('Ø Ladeleistung nötig (wenn nur im Fenster)', 'Avg charging power needed (if only within window)')}: "
            f"**{_fmt_kw(cf.ev_kw_avg_needed, 1)}**"
        )

    if total_price > 0 and cf.ev_kwh_day_total > 0:
        lines.append("")
        lines.append(L("**Ladekosten (EV):**", "**Charging cost (EV):**"))
        lines.append(f"- {L('ca.', 'approx.')} **{_fmt_eur(ev_charge_cost_day, 0)} / {L('Tag', 'day')}**")
        lines.append(f"- {L('ca.', 'approx.')} **{_fmt_eur(ev_charge_cost_year, 0)} / {L('Jahr', 'year')}** "
                     f"({L('bei 365 Tagen', '365 days')})")

    # Decision: feasible?
    lines.append("")
    if cf.ev_trucks <= 0:
        lines.append(L("**Status:** Keine EV-Trucks konfiguriert – Lade-/Peak-Entscheidung entfällt.",
                       "**Status:** No EV trucks configured — charging/peak decision not applicable."))
    else:
        if cf.feasible_under_cap:
            lines.append(L("**Status:** **Laden passt unter den Peak/Cap** – gute Ausgangslage für skalierbaren Betrieb.",
                           "**Status:** **Charging fits under peak/cap** — good baseline for scalable operations."))
            if cf.time_needed_at_headroom_h is not None:
                lines.append(
                    L("- Benötigte Ladezeit (bei Headroom): ",
                      "- Charging time needed (at headroom): ")
                    + f"**{cf.time_needed_at_headroom_h:.1f} h** "
                      f"{L('vs. Fenster', 'vs window')} **{cf.window_hours} h**"
                )
        else:
            lines.append(L("**Status:** **Laden passt NICHT unter den Peak/Cap** – es braucht eine bewusste Strategie.",
                           "**Status:** **Charging does NOT fit under peak/cap** — requires an explicit strategy."))
            lines.append(
                L("- Unter Cap gehen sich ca. ",
                  "- Under cap, approx. ")
                + f"**{cf.max_trucks_under_cap} / {cf.ev_trucks} {L('Trucks', 'trucks')}** "
                  + L("aus (gleichmäßiges Laden angenommen).",
                      "(assuming uniform charging).")
            )
            lines.append(
                L("- Alternativ: Peak-Erhöhung um ca. ",
                  "- Alternative: increase peak by approx. ")
                + f"**{_fmt_kw(cf.extra_peak_kw_if_full, 1)}**."
            )
            if cf.extra_capacity_cost_eur_year > 0:
                lines.append(
                    L("- Grobe Mehrkosten Leistungsentgelt: ",
                      "- Rough additional demand-charge cost: ")
                    + f"**{_fmt_eur(cf.extra_capacity_cost_eur_year, 0)} / {L('Jahr', 'year')}**"
                )
            if cf.tour_completion_ratio < 1.0:
                lines.append(
                    L("- Best-effort Tour-Abdeckung (energie-basiert): ",
                      "- Best-effort tour coverage (energy-based): ")
                    + f"**{_fmt_pct(cf.tour_completion_ratio, 0)}** "
                    + L("(wenn strikt unter Cap geladen wird).",
                        "(if strictly charging under cap).")
                )

    # Top solutions (ordered)
    lines.append("")
    lines.append(L("**Empfohlene Optionen (pragmatisch):**",
                   "**Recommended options (pragmatic):**"))
    for s in solutions[:4]:
        flag = L(" (empfohlen)", " (recommended)") if s.get("recommended") else ""
        title = _pick(s.get("title", "Option"))
        what = _pick(s.get("what", ""))
        lines.append(f"- **{title}**{flag}: {what}")

        how = s.get("how", [])
        if isinstance(how, list) and how:
            for h in how[:3]:
                lines.append(f"  - {_pick(h)}")

    # Outlook
    lines.append("")
    lines.append(L("### Outlook (was noch möglich ist)",
                   "### Outlook (what else is possible)"))
    lines.append(L("- **Automatisches Peak-Shaving**: Ladeprofile dynamisch an Preise + Netzzustand koppeln (Kosten runter, Planbarkeit rauf).",
                   "- **Automated peak shaving**: link charging profiles dynamically to prices + grid state (lower cost, higher predictability)."))
    lines.append(L("- **Skalierungssimulation**: Was passiert bei +5 / +10 Trucks? Cap, Kosten, notwendige Infrastruktur auf Knopfdruck.",
                   "- **Scale simulation**: what happens at +5 / +10 trucks? Cap, cost, required infrastructure at the push of a button."))
    lines.append(L("- **CO₂ & ESG-Reporting**: auditierbare Einsparungen (Diesel→EV), Scope-Reporting, Förderlogik.",
                   "- **CO₂ & ESG reporting**: auditable savings (diesel→EV), scope reporting, subsidy logic."))
    lines.append(L("- **Dispatch & Tour-Optimierung**: Tourenplanung gekoppelt an Lade- und Energieverfügbarkeit (weniger Überraschungen im Betrieb).",
                   "- **Dispatch & tour optimisation**: route planning coupled with charging + energy availability (fewer operational surprises)."))

    # ZYNE modules (non-assertive)
    lines.append("")
    lines.append(L("### ZYNE – welche Bausteine du typischerweise jetzt brauchst",
                   "### ZYNE — typical building blocks you’ll need next"))
    lines.append(L("- **Energy & Tariff Intelligence**: belastbare Preis-/Netzkostenlogik inkl. Leistungsentgelt und Szenarien.",
                   "- **Energy & Tariff Intelligence**: robust price/grid-cost logic incl. demand charges and scenarios."))
    lines.append(L("- **Charging Control / EMS**: Priorisierung, Ladefenster-Regeln, harte Peak-Kappung, Alerts bei Cap-Risiko.",
                   "- **Charging Control / EMS**: prioritisation, charging-window rules, hard peak capping, alerts on cap risk."))
    lines.append(L("- **Monitoring & Reporting**: Dashboards (Peak, kWh, Kosten, Abweichungen), Export für Stakeholder/Controlling.",
                   "- **Monitoring & reporting**: dashboards (peak, kWh, cost, deviations), exports for stakeholders/controlling."))
    lines.append(L("- **Data Connectors**: Telematik/Depot/Charger/Load-Profile sauber anbinden (damit das Modell “lebt”).",
                   "- **Data connectors**: integrate telematics/depot/chargers/load profiles cleanly (so the model “lives”)."))

    return "\n".join(lines)



# -------------------------
# Gemini final answer: structured inputs + prompt
# -------------------------

def build_gemini_final_answer_inputs(res: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a structured, number-backed object you can feed into Gemini/LLM.
    Keeps wording neutral, data-driven.
    """
    cf = compute_charging_feasibility(res)
    issues = detect_issues(res)
    solutions = generate_solution_set(res, issues)

    ec = _extract_energy_cost(res)
    total_price = _num(_get(ec, "total_price_eur_kwh", _get(ec, "total_eur_per_kwh", 0.0)), 0.0)

    return {
        "charging": {
            "window_hours": cf.window_hours,
            "kw_limit": cf.kw_limit,
            "baseline_avg_kw_in_window": cf.baseline_avg_kw_in_window,
            "headroom_kw": cf.headroom_kw,
            "ev_trucks": cf.ev_trucks,
            "ev_kwh_day_total": cf.ev_kwh_day_total,
            "ev_kw_avg_needed_in_window": cf.ev_kw_avg_needed,
            "feasible_under_cap": cf.feasible_under_cap,
            "max_trucks_under_cap": cf.max_trucks_under_cap,
            "extra_peak_kw_if_full_in_window": cf.extra_peak_kw_if_full,
            "extra_capacity_cost_eur_year_est": cf.extra_capacity_cost_eur_year,
            "time_needed_at_headroom_h": cf.time_needed_at_headroom_h,
            "tour_completion_ratio_under_cap_est": cf.tour_completion_ratio,
        },
        "energy": {
            "price_eur_kwh_total": total_price,
            "ev_charge_cost_day_est": (cf.ev_kwh_day_total * total_price) if total_price > 0 else None,
            "ev_charge_cost_year_est": (cf.ev_kwh_day_total * total_price * 365.0) if total_price > 0 else None,
            "ev_energy_kwh_year": compute_ev_energy_year_kwh(res, 365),
            "facility_energy_kwh_year_baseline": compute_facility_energy_year_kwh(res, 365),
        },
        "issues": issues,
        "solutions": solutions,
    }


def build_gemini_prompt(res: Dict[str, Any], lang: str) -> str:
    """
    A ready-to-use prompt string (German) that asks Gemini to produce a structured,
    stakeholder-ready answer (supported description, marketing-ish but factual).
    """

    payload = build_gemini_final_answer_inputs(res)

    # Keep prompt stable and explicit about required structure.
    prompt = f"""
Sprache;
{lang}

Inhalt:
Du bist ein Berater für Flottenelektrifizierung und Energiekosten. Nutze ausschließlich die folgenden Zahlen/Objekte.
Antworte auf Deutsch, klar, mit leicht werblichem Ton, aber ohne Übertreibung. Keine Fantasiezahlen.

Pflicht-Struktur:
1) Executive Summary (3-5 Sätze)
2) Laden & Peak/Cap (klar: geht es sich aus? wenn nein: wie viele Trucks bis Cap, wie viel Mehr-Peak, grobe Mehrkosten)
3) Betrieb (Touren/Ladezeit: best-effort anhand der gelieferten Kennzahlen, Unsicherheit transparent markieren)
4) Finanzieller Impact (Ladekosten/ Jahr, zentrale Treiber)
5) Entscheidungsvorlage (Option A: unter Cap bleiben, Option B: Peak erhöhen, Option C: Fenster/Strategie ändern; je 2-3 bullets)
6) Short Outlook
7) ZYNE Produkte

Daten:
{payload}
""".strip()
    return prompt


# -------------------------
# Optional Streamlit wrappers (so main app changes are minimal)
# -------------------------

def render_power_consumption_chart_streamlit(res: Dict[str, Any], st, *, title: str = None) -> None:
    fig = plot_power_consumption_comparison(res, title=title or "New power consumption vs baseline (with EV charging)")
    st.pyplot(fig, clear_figure=True)

def render_energy_year_chart_streamlit(res: Dict[str, Any], st, *, title: str = None) -> None:
    fig = plot_energy_year(res, title=title or "Energy consumption per year")
    st.pyplot(fig, clear_figure=True)

def render_recommendations_streamlit(res: Dict[str, Any], st) -> None:
    st.markdown(build_recommendation_markdown(res, "EN"))



# =========================================================
# Report helpers (Analysis / Constraints / Next steps)
# =========================================================

def report_analysis_markdown(res: dict) -> str:
    """
    Short human-readable analysis summary.
    Must NEVER throw – UI safety first.
    """
    try:
        if not isinstance(res, dict):
            return "—"

        lines = []

        # High-level signals (only if present)
        if "diesel_vs_ev" in res:
            dv = res["diesel_vs_ev"]
            if isinstance(dv, dict) and "delta_total_cost" in dv:
                d = dv["delta_total_cost"]
                if d < 0:
                    lines.append("• **EV günstiger als Diesel** im betrachteten Szenario.")
                else:
                    lines.append("• **EV aktuell teurer als Diesel** – Optimierungspotenzial vorhanden.")

        if "capacity_analysis" in res:
            cap = res["capacity_analysis"]
            if isinstance(cap, dict) and cap.get("headroom_kw", 0) <= 0:
                lines.append("• **Kein Leistungs-Headroom** im Ladefenster – Peak-Risiko.")

        if not lines:
            lines.append("• Keine kritischen Auffälligkeiten im Überblick.")

        return "\n".join(lines)

    except Exception as e:
        return f"⚠️ Analyse nicht verfügbar (`{e}`)"


def report_constraints(res: dict):
    """
    Returns a list of tuples: (level, text)
    level ∈ {'ok', 'warn', 'bad'}
    """
    out = []

    try:
        if not isinstance(res, dict):
            return out

        cap = res.get("capacity_analysis", {})
        if isinstance(cap, dict):
            headroom = cap.get("headroom_kw")
            if headroom is not None:
                if headroom <= 0:
                    out.append(("bad", "Kein Leistungs-Headroom im Ladefenster verfügbar."))
                elif headroom < 100:
                    out.append(("warn", f"Geringer Leistungs-Headroom ({headroom:.0f} kW)."))
                else:
                    out.append(("ok", f"Ausreichender Leistungs-Headroom ({headroom:.0f} kW)."))

        load = res.get("load", {})
        if isinstance(load, dict):
            util = load.get("avg_utilization")
            if util is not None and util > 0.9:
                out.append(("warn", "Sehr hohe durchschnittliche Auslastung im Ladefenster."))

        return out

    except Exception:
        return out  # never crash UI


def report_next_steps(res: dict):
    """
    Returns list[str] – actionable next steps.
    """
    steps = []

    try:
        if not isinstance(res, dict):
            return steps

        cap = res.get("capacity_analysis", {})
        if isinstance(cap, dict):
            if cap.get("headroom_kw", 0) <= 0:
                steps.append("Ladeleistung pro LKW senken oder Ladefenster verlängern.")
                steps.append("Peak-Shaving (Batterie / EMS) prüfen.")

        dv = res.get("diesel_vs_ev", {})
        if isinstance(dv, dict):
            if dv.get("delta_total_cost", 0) > 0:
                steps.append("Stromtarif (Fix/Dynamik) und Netzentgelt optimieren.")

        if not steps:
            steps.append("Szenario feinjustieren (Smart Charging, Ladeleistung, Tarif).")

        return steps

    except Exception:
        return steps
