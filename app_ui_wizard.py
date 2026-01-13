# FleetMate — Streamlit app (2026-01-12)
# - Removes manual SoC inputs; derives SoC from km/day
# - EV consumption defaults adjusted to realistic range (0.8–1.2)
# - Replaces "dynamic share" with Fixed vs Dynamic price mode
# - Adds load profile CSV upload + charging-window capacity analysis
# - Updates toll defaults to ASFINAG-like values (simplified)
# - Adds clearer finance charts ordering


import os
import json
import re
import io
import datetime
import hmac
import hashlib

import requests
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # important in Docker/Headless
import matplotlib.pyplot as plt

from recommendations import *

# =========================================================
# Auth
# =========================================================
def require_login():
    """Simple password gate (shared-secret)."""
    try:
        expected = str(st.secrets.get("APP_PASSWORD", "")).strip()
    except Exception:
        expected = ""
    if not expected:
        expected = os.getenv("APP_PASSWORD", "").strip()  # optional env fallback

    st.session_state.setdefault("authed", False)
    if st.session_state["authed"]:
        return

    st.markdown("## 🔒 FleetMate Login")
    pw = st.text_input("Passwort", type="password")

    c1, c2 = st.columns([0.35, 0.65])
    with c1:
        if st.button("Anmelden", use_container_width=True):
            if expected and hmac.compare_digest(pw, expected):
                st.session_state["authed"] = True
                st.rerun()
            else:
                st.error("Falsches Passwort.")
    with c2:
        st.caption("Tipp: Das Passwort ist die Partner-Firma klein geschrieben. (7 Buchstaben - Österreich)")

    st.stop()


# =========================================================
# Core constants + model (same general logic)
# =========================================================
DIESEL_CO2_PER_L = 2.64  # kg CO2 per litre

GRID_CO2_G_PER_KWH = [
    80, 78, 75, 70, 65, 60, 60, 65, 70, 75, 60, 50,
    45, 45, 50, 60, 70, 80, 90, 95, 90, 83, 78, 76
]

# Fallback time-of-use curve (only used if no dynamic spot data is available)
TOU_PRICE_EUR_PER_KWH = [
    0.20, 0.195, 0.19, 0.185, 0.18, 0.18, 0.185, 0.19,
    0.21, 0.24, 0.23, 0.22, 0.20, 0.20, 0.205, 0.21,
    0.23, 0.26, 0.30, 0.33, 0.31, 0.28, 0.24, 0.22
]

# ---- Dynamic pricing: fixed country averages (€/MWh), NOT from upload ----
DYNAMIC_AVG_PRICE_EUR_PER_MWH = {
    "DE": 86.79,  # Germany average (current default) for 2025
    "AT": 86.79,  # TODO: set Austria average
    "EU": 86.79,  # TODO: set EU average
}


def compute_flags_and_shares(start_hour: int, end_hour: int):
    """Return 24 hourly flags + shares (1 inside the window)."""
    flags = [0] * 24
    for h in range(24):
        if start_hour < end_hour:
            flags[h] = 1 if start_hour <= h < end_hour else 0
        elif start_hour > end_hour:
            flags[h] = 1 if (h >= start_hour or h < end_hour) else 0
        else:
            flags[h] = 1
    total = sum(flags)
    shares = [(f / total) if total > 0 else 0.0 for f in flags]
    return flags, shares


def effective_grid_co2_details(shares):
    """Window-weighted grid CO2 (hourly curve)."""
    day_avg_g = float(sum(GRID_CO2_G_PER_KWH)) / 24.0
    window_avg_g = float(sum(c * s for c, s in zip(GRID_CO2_G_PER_KWH, shares)))
    return (window_avg_g / 1000.0), {
        "grid_day_avg_g_per_kwh": day_avg_g,
        "grid_window_avg_g_per_kwh": window_avg_g,
        "grid_window_avg_kg_per_kwh": window_avg_g / 1000.0,
    }


def effective_energy_price_details(
    avg_price_eur_per_mwh: float,
    dynamic_share: float,
    shares,
    spot_curve_avg_eur_per_kwh: float | None = None,
    spot_window_avg_eur_per_kwh: float | None = None,
):
    """
    Effective electricity price (€/MWh).

    - FIXED mode -> dynamic_share will be 0.0 -> returns avg_price (no TOU weighting)
    - DYNAMIC mode -> dynamic_share will be 1.0:
        - If spot_curve_avg/window_avg are available -> effective == spot_window_avg (correct)
        - Else fallback to TOU curve scaling (rough estimate)
    """
    avg_price_eur_per_kwh = max(0.0, float(avg_price_eur_per_mwh)) / 1000.0
    dynamic_share = min(max(float(dynamic_share), 0.0), 1.0)

    # Default: TOU curve (fallback only)
    curve_avg = float(sum(TOU_PRICE_EUR_PER_KWH)) / 24.0
    window_avg = float(sum(p * s for p, s in zip(TOU_PRICE_EUR_PER_KWH, shares)))
    rel = (window_avg / curve_avg) if curve_avg > 0 else 1.0

    # If we have real spot data, use its window/day ratio instead of TOU heuristic
    if spot_curve_avg_eur_per_kwh is not None and spot_window_avg_eur_per_kwh is not None:
        curve_avg = float(max(1e-9, spot_curve_avg_eur_per_kwh))
        window_avg = float(spot_window_avg_eur_per_kwh)
        rel = window_avg / curve_avg

    fixed_part = avg_price_eur_per_kwh * (1.0 - dynamic_share)
    dynamic_part = avg_price_eur_per_kwh * dynamic_share * rel
    eff_eur_per_kwh = fixed_part + dynamic_part

    eff_eur_per_mwh = eff_eur_per_kwh * 1000.0
    details = {
        "avg_price_eur_per_kwh": avg_price_eur_per_kwh,
        "curve_avg_eur_per_kwh": curve_avg,
        "window_avg_eur_per_kwh": window_avg,
        "rel_factor": rel,
        "dynamic_share": dynamic_share,
        "effective_price_eur_per_mwh": eff_eur_per_mwh,
        "used_spot_data": bool(spot_curve_avg_eur_per_kwh is not None and spot_window_avg_eur_per_kwh is not None),
    }
    return eff_eur_per_mwh, details


def run_model(
    num_trucks: int,
    operating_days: int,
    events_per_truck_per_day: float,
    battery_kwh: float,
    start_soc: float,
    target_soc: float,
    avg_elec_price_eur_per_mwh: float,
    dynamic_price_share: float,
    start_hour: int,
    end_hour: int,
    charging_window_hours: float,
    existing_site_peak_kw: float,
    charger_power_per_truck_kw: float,
    site_capacity_limit_kva: float,
    ev_consumption_kwh_per_km: float,
    diesel_price_eur_per_l: float,
    diesel_l_per_100km: float,
    toll_rate_eur_per_km: float,
    tolled_share_0_1: float,
    ev_toll_exempt: bool,
    desired_peak_limit_kw: float,
    peak_duration_h: float,
    # Optional spot price stats (from upload)
    spot_curve_avg_eur_per_kwh: float | None = None,
    spot_window_avg_eur_per_kwh: float | None = None,
):
    """
    Core business-case model.

    NOTE:
    - We keep start_soc/target_soc in the model for backward compatibility.
      In the UI, they are derived from km/day (not user inputs anymore).
    - We keep dynamic_price_share internally (0=fixed, 1=dynamic) for compatibility.
      In the UI, the user chooses a price mode instead of entering a share.
    """
    num_trucks = max(1, int(num_trucks))
    operating_days = max(1, int(operating_days))
    events_per_truck_per_day = max(0.0, float(events_per_truck_per_day))
    battery_kwh = max(0.0, float(battery_kwh))
    start_soc = min(max(float(start_soc), 0.0), 1.0)
    target_soc = min(max(float(target_soc), 0.0), 1.0)
    ev_consumption_kwh_per_km = max(0.01, float(ev_consumption_kwh_per_km))
    charging_window_hours = max(0.1, float(charging_window_hours))
    dynamic_price_share = min(max(float(dynamic_price_share), 0.0), 1.0)
    tolled_share_0_1 = min(max(float(tolled_share_0_1), 0.0), 1.0)
    peak_duration_h = max(0.0, float(peak_duration_h))

    start_h = int(start_hour) % 24
    end_h = int(end_hour) % 24
    flags, shares = compute_flags_and_shares(start_h, end_h)

    # Energy from SoC diff
    soc_diff = max(0.0, target_soc - start_soc)
    energy_per_event_kwh = battery_kwh * soc_diff
    energy_per_event_mwh = energy_per_event_kwh / 1000.0

    total_daily_energy_mwh = num_trucks * events_per_truck_per_day * energy_per_event_mwh
    annual_energy_mwh = total_daily_energy_mwh * operating_days

    eff_price_eur_per_mwh, price_details = effective_energy_price_details(
        avg_elec_price_eur_per_mwh,
        dynamic_price_share,
        shares,
        spot_curve_avg_eur_per_kwh=spot_curve_avg_eur_per_kwh,
        spot_window_avg_eur_per_kwh=spot_window_avg_eur_per_kwh,
    )

    annual_cost_eur = annual_energy_mwh * eff_price_eur_per_mwh

    # CO2
    eff_grid_co2_kg_per_kwh, co2_details = effective_grid_co2_details(shares)
    annual_energy_kwh = annual_energy_mwh * 1000.0
    annual_ev_co2_kg = annual_energy_kwh * eff_grid_co2_kg_per_kwh

    # Distance (derived from energy + consumption)
    daily_energy_kwh = total_daily_energy_mwh * 1000.0
    kwh_per_truck_per_day = daily_energy_kwh / num_trucks
    km_per_truck_per_day = kwh_per_truck_per_day / ev_consumption_kwh_per_km
    annual_km_per_truck = km_per_truck_per_day * operating_days
    annual_km_fleet = annual_km_per_truck * num_trucks

    # Diesel baseline
    diesel_litres_baseline = (annual_km_fleet * diesel_l_per_100km) / 100.0
    diesel_cost_baseline = diesel_litres_baseline * diesel_price_eur_per_l
    diesel_co2_baseline_kg = diesel_litres_baseline * DIESEL_CO2_PER_L

    # EV scenario
    ev_cost = annual_cost_eur
    ev_co2_kg = annual_ev_co2_kg

    cost_savings_eur = diesel_cost_baseline - ev_cost
    co2_savings_kg = diesel_co2_baseline_kg - ev_co2_kg

    # Toll
    baseline_toll_cost = annual_km_fleet * tolled_share_0_1 * toll_rate_eur_per_km
    ev_toll_cost = 0.0 if ev_toll_exempt else baseline_toll_cost
    toll_savings = baseline_toll_cost - ev_toll_cost
    total_savings_incl_toll = cost_savings_eur + toll_savings

    # Load / capacity
    total_charge_power_kw = num_trucks * charger_power_per_truck_kw
    new_theoretical_peak_kw = existing_site_peak_kw + total_charge_power_kw
    avg_charging_power_kw = daily_energy_kwh / charging_window_hours
    new_avg_load_kw = existing_site_peak_kw + avg_charging_power_kw
    capacity_ok = (new_theoretical_peak_kw <= site_capacity_limit_kva) if site_capacity_limit_kva > 0 else True

    # Optional peak shaving
    desired_peak_limit_kw = max(0.0, float(desired_peak_limit_kw))
    required_shaving_kw = max(0.0, new_theoretical_peak_kw - desired_peak_limit_kw) if desired_peak_limit_kw > 0 else 0.0
    required_battery_energy_kwh = required_shaving_kw * peak_duration_h if peak_duration_h > 0 else 0.0

    return {
        "inputs": {
            "num_trucks": num_trucks,
            "operating_days": operating_days,
            "events_per_truck_per_day": events_per_truck_per_day,
            "battery_kwh": battery_kwh,
            "start_soc": start_soc,
            "target_soc": target_soc,
            "avg_elec_price_eur_per_mwh": avg_elec_price_eur_per_mwh,
            "dynamic_price_share": dynamic_price_share,
            "start_hour": start_h,
            "end_hour": end_h,
            "charging_window_hours": charging_window_hours,
            "existing_site_peak_kw": existing_site_peak_kw,
            "charger_power_per_truck_kw": charger_power_per_truck_kw,
            "site_capacity_limit_kva": site_capacity_limit_kva,
            "ev_consumption_kwh_per_km": ev_consumption_kwh_per_km,
            "diesel_price_eur_per_l": diesel_price_eur_per_l,
            "diesel_l_per_100km": diesel_l_per_100km,
            "toll_rate_eur_per_km": toll_rate_eur_per_km,
            "tolled_share_0_1": tolled_share_0_1,
            "ev_toll_exempt": ev_toll_exempt,
            "desired_peak_limit_kw": desired_peak_limit_kw,
            "peak_duration_h": peak_duration_h,
        },
        "charging_profile": {
            "flags": flags,
            "shares": shares,
            "grid_co2_g_per_kwh": GRID_CO2_G_PER_KWH,
            "tou_price_eur_per_kwh": TOU_PRICE_EUR_PER_KWH,  # may be overridden for charts
        },
        "energy_cost": {
            "soc_diff": soc_diff,
            "energy_per_event_kwh": energy_per_event_kwh,
            "total_daily_energy_mwh": total_daily_energy_mwh,
            "annual_energy_mwh": annual_energy_mwh,
            "effective_price_eur_per_mwh": eff_price_eur_per_mwh,
            "annual_cost_eur": annual_cost_eur,
            "price_details": price_details,
        },
        "co2": {
            "effective_grid_co2_kg_per_kwh": eff_grid_co2_kg_per_kwh,
            "annual_ev_co2_kg": annual_ev_co2_kg,
            "co2_details": co2_details,
        },
        "distance": {
            "km_per_truck_per_day": km_per_truck_per_day,
            "annual_km_fleet": annual_km_fleet,
        },
        "diesel_vs_ev": {
            "diesel_cost_baseline_eur": diesel_cost_baseline,
            "diesel_co2_baseline_kg": diesel_co2_baseline_kg,
            "ev_cost_eur": ev_cost,
            "ev_co2_kg": ev_co2_kg,
            "cost_savings_eur": cost_savings_eur,
            "co2_savings_kg": co2_savings_kg,
            "baseline_toll_cost_eur": baseline_toll_cost,
            "toll_savings_eur": toll_savings,
            "total_savings_incl_toll_eur": total_savings_incl_toll,
        },
        "load": {
            "total_charge_power_kw": total_charge_power_kw,
            "new_theoretical_peak_kw": new_theoretical_peak_kw,
            "new_avg_load_kw": new_avg_load_kw,
            "capacity_ok": capacity_ok,
            "required_shaving_kw": required_shaving_kw,
            "required_battery_energy_kwh": required_battery_energy_kwh,
        },
    }


# =========================================================
# Defaults + persistent input storage
# =========================================================
DEFAULT_LANG = "DE"
# English comments: theme defaults
DEFAULT_THEME = "system"

DEFAULT_INPUTS = dict(
    # Fleet
    num_trucks=10,
    operating_days=360,
    events_per_truck=1.0,

    # Usage (NEW): km/day (replaces SoC inputs)
    km_per_truck_per_day=220.0,

    # Battery / consumption
    vehicle_profile="heavy_regional",
    battery_kwh=500.0,
    # SoC is now derived from km/day (kept for internal compatibility)
    start_soc=0.20,
    target_soc=1.00,

    # EV consumption default adjusted (was too high)
    ev_consumption=1.0,

    # Electricity
    price_mode="fixed",           # "fixed" | "dynamic"
    fixed_elec_price_mwh=86.79,   # used when price_mode=fixed

    # NEW: only used when price_mode="dynamic"
    dynamic_price_region="DE",

    # Internals (computed):
    avg_elec_price_mwh=86.79,     # gets set to fixed price or last-year spot average
    dynamic_share=0.0,            # 0=fixed, 1=dynamic

    # Charging window
    start_hour=22,
    end_hour=6,
    charging_window_hours=8.0,

    # Site / capacity
    # Manual fallback (if no load profile uploaded)
    existing_peak_kw=3000.0,
    charger_power_kw=150.0,
    site_capacity_kva=5530.0,

    # Optional peak limit
    desired_peak_limit_kw=0.0,
    peak_duration_h=0.25,

    # Diesel baseline + toll
    market_region="AT",
    diesel_price=1.75,
    diesel_l_per_100=22.0,

    # Toll (default updated; simplified ASFINAG-style rate for 4+ axles EURO VI-ish)
    # NOTE: Real GO-toll depends on axles + emission classes + special sections. Keep editable.
    toll_rate=0.456,     # €/km (excl. VAT), representative for 4+ axles EURO VI in 2025
    tolled_share=0.60,
    ev_toll_exempt=True,
)

# Gemini may only update these (avoid letting the LLM switch profiles/regions or hidden internals)
HIDDEN_OR_CONTROLLED = {
    "vehicle_profile", "market_region",
    "start_soc", "target_soc",
    "avg_elec_price_mwh", "dynamic_share",
    "charging_window_hours",
}
ALLOWED_INPUT_KEYS = set(DEFAULT_INPUTS.keys()) - HIDDEN_OR_CONTROLLED


def ensure_defaults():
    st.session_state.setdefault("lang", DEFAULT_LANG)
    st.session_state.setdefault("theme", DEFAULT_THEME)  # English comment: persist theme across reruns
    st.session_state.setdefault("inputs", DEFAULT_INPUTS.copy())

    # Flow control
    st.session_state.setdefault("flow", "wizard")  # wizard | report
    st.session_state.setdefault("wizard_step", 0)

    # Model and report caches
    st.session_state.setdefault("model_results", None)
    st.session_state.setdefault("report_md", None)
    st.session_state.setdefault("report_meta", None)
    st.session_state.setdefault("report_pdf_bytes", None)
    st.session_state.setdefault("assistant_messages", [])
    st.session_state.setdefault("last_gemini_payload", None)
    st.session_state.setdefault("charts_md", None)

    # Upload cache
    st.session_state.setdefault("profile_cache", {
        "digest": None,
        "df": None,
        "timestamp_col": None,
        "consumption_col": None,
        "meta": {},
        "metrics": {},
    })


def get_inp(k: str):
    return st.session_state["inputs"].get(k, DEFAULT_INPUTS.get(k))


def set_inp(k: str, v):
    st.session_state["inputs"][k] = v


def window_len_hours(start_h: int, end_h: int) -> int:
    start_h = int(start_h) % 24
    end_h = int(end_h) % 24
    if start_h < end_h:
        return end_h - start_h
    if start_h > end_h:
        return 24 - start_h + end_h
    return 24


# =========================================================
# Derived inputs (SoC + pricing)
# =========================================================
def derive_soc_from_km(inputs: dict) -> dict:
    """
    Derive start_soc/target_soc from km/day.

    Assumption (simple & customer-friendly):
    - The truck starts the day fully charged (target_soc = 100%).
    - km/day consumes energy = km * kWh/km.
    - If events/day > 1, km are split evenly across charging events.

    Returns dict with:
      start_soc, target_soc, soc_diff, km_per_event, energy_per_event_kwh, daily_energy_kwh, warning
    """
    km_day = float(max(0.0, inputs.get("km_per_truck_per_day", 0.0)))
    events = float(max(0.0, inputs.get("events_per_truck", 0.0)))
    events = events if events > 0 else 1.0

    battery_kwh = float(max(1e-9, inputs.get("battery_kwh", 0.0)))
    cons = float(max(0.01, inputs.get("ev_consumption", 0.01)))

    km_per_event = km_day / events
    energy_per_event_kwh = km_per_event * cons
    daily_energy_kwh = km_day * cons

    soc_diff = min(1.0, energy_per_event_kwh / battery_kwh) if battery_kwh > 0 else 1.0
    start_soc = max(0.0, 1.0 - soc_diff)
    target_soc = 1.0

    warn = None
    if daily_energy_kwh > battery_kwh * 0.98 and events <= 1.01:
        warn = "Die km/Tag übersteigen (nahezu) die Batteriekapazität. Entweder mehrmals laden oder Batterie/Verbrauch prüfen."
    if soc_diff >= 0.98:
        warn = warn or "Pro Lade-Event fast Vollzyklus — prüfe km/Tag, Verbrauch oder Batteriegröße."

    return {
        "start_soc": start_soc,
        "target_soc": target_soc,
        "soc_diff": soc_diff,
        "km_per_event": km_per_event,
        "energy_per_event_kwh": energy_per_event_kwh,
        "daily_energy_kwh": daily_energy_kwh,
        "warning": warn,
    }


def apply_pricing_mode(inputs: dict) -> dict:
    """
    Map price_mode -> internal avg_elec_price_mwh + dynamic_share.

    - fixed:
        - avg_elec_price_mwh = fixed_elec_price_mwh
        - dynamic_share = 0.0
    - dynamic:
        - avg_elec_price_mwh = fixed country average (AT/DE/EU), NOT from upload
        - dynamic_share = 1.0 (TOU weighting via TOU_PRICE_EUR_PER_KWH)
    """
    mode = str(inputs.get("price_mode", "fixed")).strip().lower()
    fixed_mwh = float(max(0.0, inputs.get("fixed_elec_price_mwh", 0.0)))

    if mode == "dynamic":
        inputs["dynamic_share"] = 1.0
        region = str(inputs.get("dynamic_price_region", "DE")).strip().upper()
        if region not in DYNAMIC_AVG_PRICE_EUR_PER_MWH:
            region = "DE"
        inputs["dynamic_price_region"] = region
        inputs["avg_elec_price_mwh"] = float(DYNAMIC_AVG_PRICE_EUR_PER_MWH[region])
    else:
        inputs["price_mode"] = "fixed"
        inputs["dynamic_share"] = 0.0
        inputs["avg_elec_price_mwh"] = fixed_mwh

    return inputs



# =========================================================
# CSV load profile upload + parsing
# =========================================================
def _guess_timestamp_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        cl = str(c).lower()
        if cl in ("timestamp", "datetime", "date_time", "time", "date"):
            return c
    # If first column looks like datetime, use it
    if len(df.columns) > 0:
        c0 = df.columns[0]
        s = df[c0].astype(str).head(5).tolist()
        if any((":" in x and "-" in x) or ("T" in x) for x in s):
            return c0
    return None


def _guess_consumption_col(df: pd.DataFrame) -> str | None:
    # Prefer typical naming
    prefs = ["consumption_kw", "consumption_kW", "load_kw", "site_load_kw", "consumption", "load"]
    for p in prefs:
        for c in df.columns:
            if str(c).lower() == p.lower():
                return c
    # Heuristic: any numeric column containing 'consumption' or 'load' or 'kw'
    cand = []
    for c in df.columns:
        cl = str(c).lower()
        if any(k in cl for k in ["consumption", "load"]) and ("pv" not in cl):
            cand.append(c)
    if not cand:
        for c in df.columns:
            cl = str(c).lower()
            if "kw" in cl and "pv" not in cl:
                cand.append(c)
    # Pick first numeric
    for c in cand:
        try:
            pd.to_numeric(df[c].dropna().head(50))
            return c
        except Exception:
            continue
    return None


def parse_uploaded_csv(upload) -> pd.DataFrame | None:
    """Parse uploaded CSV with best-effort delimiter and date parsing."""
    if upload is None:
        return None
    try:
        data = upload.getvalue()
        if not data:
            return None
        bio = io.BytesIO(data)
        # sep=None triggers python engine sniffing; handles commas/semicolons often
        df = pd.read_csv(bio, sep=None, engine="python")
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def _in_window_hours(hour: int, start_h: int, end_h: int) -> bool:
    if start_h < end_h:
        return start_h <= hour < end_h
    if start_h > end_h:
        return hour >= start_h or hour < end_h
    return True


def compute_profile_metrics(
    df_raw: pd.DataFrame,
    timestamp_col: str,
    consumption_col: str,
    start_h: int,
    end_h: int,
) -> dict:
    """
    Compute capacity-relevant metrics from load profile.

    We focus only on the charging window, because that's where charging happens.
    - peak_kw_in_window: max site consumption during charging window (conservative)
    - hourly_avg_kw / hourly_max_kw: for charts
    - spot_* averages: from price column if available (supports €/MWh or €/kWh)
    """
    df = df_raw.copy()

    # Timestamp parsing
    ts = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.loc[ts.notna()].copy()
    df["ts"] = ts.loc[ts.notna()].dt.tz_localize(None)

    # Consumption numeric
    df["site_kw"] = pd.to_numeric(df[consumption_col], errors="coerce")
    df = df.loc[df["site_kw"].notna()].copy()

    df["hour"] = df["ts"].dt.hour
    df["in_window"] = df["hour"].apply(lambda h: _in_window_hours(int(h), start_h, end_h))

    # Hourly series for charts
    hourly_avg = df.groupby("hour")["site_kw"].mean().reindex(range(24))
    hourly_max = df.groupby("hour")["site_kw"].max().reindex(range(24))
    hourly_avg = hourly_avg.fillna(method="ffill").fillna(method="bfill").fillna(0.0)
    hourly_max = hourly_max.fillna(method="ffill").fillna(method="bfill").fillna(0.0)

    inw = df.loc[df["in_window"]]
    peak_kw_in_window = float(inw["site_kw"].max()) if not inw.empty else float(df["site_kw"].max())

    out = {
        "peak_kw_in_window": peak_kw_in_window,
        "hourly_avg_kw": [float(x) for x in hourly_avg.values],
        "hourly_max_kw": [float(x) for x in hourly_max.values],
        "has_price": False,
        "spot_hourly_avg_eur_per_kwh": None,
        "spot_day_avg_eur_per_mwh": None,
        "spot_window_avg_eur_per_kwh": None,
        "spot_curve_avg_eur_per_kwh": None,
    }

    return out


# =========================================================
# Wizard persistence helpers
# =========================================================
def wkey(step_id: str, field: str) -> str:
    return f"w_{step_id}_{field}"


def bind_number(step_id: str, field: str, label: str, **kwargs):
    k = wkey(step_id, field)
    if k not in st.session_state:
        st.session_state[k] = get_inp(field)

    def _sync():
        set_inp(field, st.session_state[k])

    st.number_input(label, key=k, on_change=_sync, **kwargs)


def bind_slider(step_id: str, field: str, label: str, min_value, max_value, step, **kwargs):
    k = wkey(step_id, field)
    if k not in st.session_state:
        st.session_state[k] = get_inp(field)

    def _sync():
        set_inp(field, st.session_state[k])

    st.slider(label, min_value, max_value, step=step, key=k, on_change=_sync, **kwargs)


def bind_checkbox(step_id: str, field: str, label: str, **kwargs):
    k = wkey(step_id, field)
    if k not in st.session_state:
        st.session_state[k] = bool(get_inp(field))

    def _sync():
        set_inp(field, bool(st.session_state[k]))

    st.checkbox(label, key=k, on_change=_sync, **kwargs)


def bind_select(step_id: str, field: str, label: str, options, format_func=None, **kwargs):
    k = wkey(step_id, field)
    if k not in st.session_state:
        st.session_state[k] = get_inp(field)

    def _sync():
        set_inp(field, st.session_state[k])

    select_kwargs = dict(options=options, key=k, on_change=_sync, **kwargs)
    if callable(format_func):
        select_kwargs["format_func"] = format_func

    st.selectbox(label, **select_kwargs)


# =========================================================
# Recalc orchestrator
# =========================================================
def recalc_from_inputs():
    """
    Central recalculation:
    1) Apply derived SoC from km/day
    2) Parse load profile (if uploaded) and apply:
       - existing_peak_kw from charging-window peak
       - spot-price averages (if dynamic)
    3) Run core model
    4) Attach capacity analysis to results (customer-friendly reverse calculation)
    """
    inp = st.session_state["inputs"]

    # ---- Load profile (if any) ----
    prof = st.session_state.get("profile_cache", {})
    prof_metrics = prof.get("metrics") or {}

    # ---- Derive SoC from km/day (no manual SoC inputs anymore) ----
    derived = derive_soc_from_km(inp)
    inp["start_soc"] = derived["start_soc"]
    inp["target_soc"] = derived["target_soc"]

    # ---- Pricing mode mapping (fixed vs dynamic) ----
    inp = apply_pricing_mode(inp)

    # ---- Existing site peak: prefer upload peak in charging window (if present) ----
    existing_peak_kw = float(inp.get("existing_peak_kw", 0.0))
    site_peak_source = "manual"
    if prof_metrics and prof_metrics.get("peak_kw_in_window") is not None:
        existing_peak_kw = float(max(0.0, prof_metrics["peak_kw_in_window"]))
        site_peak_source = "load_profile"

    # Spot stats are intentionally disabled (dynamic pricing uses fixed averages + TOU curve)
    spot_curve_avg = None
    spot_window_avg = None


    inp["charging_window_hours"] = float(window_len_hours(int(inp["start_hour"]), int(inp["end_hour"])))

    # ---- Run model ----
    res = run_model(
        num_trucks=inp["num_trucks"],
        operating_days=inp["operating_days"],
        events_per_truck_per_day=inp["events_per_truck"],
        battery_kwh=inp["battery_kwh"],
        start_soc=inp["start_soc"],
        target_soc=inp["target_soc"],
        avg_elec_price_eur_per_mwh=inp["avg_elec_price_mwh"],
        dynamic_price_share=inp["dynamic_share"],
        start_hour=inp["start_hour"],
        end_hour=inp["end_hour"],
        charging_window_hours=inp["charging_window_hours"],
        existing_site_peak_kw=existing_peak_kw,
        charger_power_per_truck_kw=inp["charger_power_kw"],
        site_capacity_limit_kva=inp["site_capacity_kva"],
        ev_consumption_kwh_per_km=inp["ev_consumption"],
        diesel_price_eur_per_l=inp["diesel_price"],
        diesel_l_per_100km=inp["diesel_l_per_100"],
        toll_rate_eur_per_km=inp["toll_rate"],
        tolled_share_0_1=inp["tolled_share"],
        ev_toll_exempt=inp["ev_toll_exempt"],
        desired_peak_limit_kw=inp["desired_peak_limit_kw"],
        peak_duration_h=inp["peak_duration_h"],
        spot_curve_avg_eur_per_kwh=spot_curve_avg,
        spot_window_avg_eur_per_kwh=spot_window_avg,
    )

    # ---- Attach profile metrics (serialisable) ----
    res.setdefault("profile", {})
    res["profile"].update({
        "site_peak_source": site_peak_source,
        "derived_soc": {
            "km_per_truck_per_day": float(inp.get("km_per_truck_per_day", 0.0)),
            "km_per_event": float(derived["km_per_event"]),
            "soc_diff": float(derived["soc_diff"]),
            "energy_per_event_kwh": float(derived["energy_per_event_kwh"]),
            "daily_energy_kwh": float(derived["daily_energy_kwh"]),
            "warning": derived.get("warning"),
        }
    })
    if prof_metrics:
        res["profile"].update({
            "peak_kw_in_window": prof_metrics.get("peak_kw_in_window"),
            "hourly_avg_kw": prof_metrics.get("hourly_avg_kw"),
            "hourly_max_kw": prof_metrics.get("hourly_max_kw"),
            "has_price": prof_metrics.get("has_price"),
            "spot_hourly_avg_eur_per_kwh": prof_metrics.get("spot_hourly_avg_eur_per_kwh"),
            "spot_curve_avg_eur_per_kwh": prof_metrics.get("spot_curve_avg_eur_per_kwh"),
            "spot_window_avg_eur_per_kwh": prof_metrics.get("spot_window_avg_eur_per_kwh"),
            "spot_day_avg_eur_per_mwh": prof_metrics.get("spot_day_avg_eur_per_mwh"),
        })

    # ---- Customer-friendly reverse calculation (capacity during charging window) ----
    # Available kW = site capacity - site consumption peak in charging window
    cap = float(max(0.0, inp.get("site_capacity_kva", 0.0)))
    peak_in_window = float(existing_peak_kw)
    avail_kw = max(0.0, cap - peak_in_window)

    wh = window_len_hours(int(inp.get("start_hour", 0)), int(inp.get("end_hour", 0)))
    # Energy available if you use the whole window at the available headroom
    energy_avail_kwh = avail_kw * float(wh)

    # Energy needed per truck per day (from km/day & consumption)
    e_truck_day = float(derived["daily_energy_kwh"])
    max_trucks_energy_based = int(np.floor(energy_avail_kwh / e_truck_day)) if e_truck_day > 1e-9 else 0

    # "Naive" simultaneous charging at full power
    charger_kw = float(max(0.01, inp.get("charger_power_kw", 0.01)))
    max_trucks_simultaneous = int(np.floor(avail_kw / charger_kw)) if avail_kw > 0 else 0

    # Recommended average charging power per truck to fit into effective charging time
    eff_h = float(max(0.1, inp.get("charging_window_hours", 0.1)))
    recommended_kw_per_truck = (e_truck_day / eff_h) if eff_h > 0 else None

    res.setdefault("capacity_analysis", {})
    res["capacity_analysis"].update({
        "charging_window_hours_total": int(wh),
        "site_capacity_kw": cap,
        "site_peak_kw_in_window": peak_in_window,
        "available_kw_at_peak": avail_kw,
        "energy_available_kwh_in_window": energy_avail_kwh,
        "energy_needed_kwh_per_truck_day": e_truck_day,
        "max_trucks_energy_based": max_trucks_energy_based,
        "max_trucks_simultaneous_at_full_power": max_trucks_simultaneous,
        "recommended_avg_kw_per_truck": float(recommended_kw_per_truck) if recommended_kw_per_truck is not None else None,
    })

    st.session_state["model_results"] = res


# =========================================================
# i18n
# =========================================================
TEXT = {
    "EN": {
        "questionnaire": "Questionnaire",
        "report": "Report",
        "progress": "Progress",
        "back": "Back",
        "next": "Next",
        "finish": "Finish & Generate Report",
        "restart": "Restart questionnaire",
        "edit_jump": "Edit a section",
        "live_preview": "Live Preview",
        "assistant_title": "Eva — your FleetMate",
        "assistant_hint": "Ask during the questionnaire. I can explain terms and adjust inputs.",
        "ask_placeholder": "Ask a question or say what to change…",
        "welcome_title": "FleetMate",
        "welcome_sub": "A guided EV site loading questionnaire with an always-on assistant.",
        "kpi_savings": "Total savings incl. toll",
        "kpi_co2": "CO₂ savings",
        "kpi_peak": "New theoretical peak",
        "per_year": "per year",
        "capacity_ok_yes": "Capacity OK",
        "capacity_ok_no": "Capacity exceeded",
        "constraints": "Constraints & recommendations",
        "narrative": "Analysis & Recommendation",
        "gemini_missing": "_(Gemini not configured — add GEMINI_API_KEY to enable narrative.)_",
        "debug": "Debug: last Gemini payload",
        "apply_defaults": "Use recommended defaults",
        "apply_window": "Use full window",
    },
    "DE": {
        "questionnaire": "Fragebogen",
        "report": "Report",
        "progress": "Fortschritt",
        "back": "Zurück",
        "next": "Weiter",
        "finish": "Abschließen & Report erzeugen",
        "restart": "Fragebogen neu starten",
        "edit_jump": "Bereich bearbeiten",
        "live_preview": "Live-Vorschau",
        "assistant_title": "Eva — dein FleetMate",
        "assistant_hint": "Frag während des Fragebogens. Ich erkläre Begriffe und kann Inputs anpassen.",
        "ask_placeholder": "Frage stellen oder Werte ändern…",
        "welcome_title": "FleetMate",
        "welcome_sub": "Geführter EV-Lade-Standort Fragebogen mit permanent aktivem Assistant.",
        "kpi_savings": "Einsparung inkl. Maut",
        "kpi_co2": "CO₂-Ersparnis",
        "kpi_peak": "Neuer theoretischer Peak",
        "per_year": "pro Jahr",
        "capacity_ok_yes": "Kapazität OK",
        "capacity_ok_no": "Kapazität überschritten",
        "constraints": "Constraints & Empfehlungen",
        "narrative": "Analyse & Empfehlung",
        "gemini_missing": "_(Gemini nicht konfiguriert — füge GEMINI_API_KEY hinzu.)_",
        "debug": "Debug: letztes Gemini-Payload",
        "apply_defaults": "Empfohlene Defaults übernehmen",
        "apply_window": "Vollständiges Fenster nutzen",
    }
}


def t(key: str) -> str:
    lang = st.session_state.get("lang", DEFAULT_LANG)
    return TEXT.get(lang, TEXT["EN"]).get(key, TEXT["EN"].get(key, key))


# =========================================================
# Gemini helpers (unchanged; only small key list update)
# =========================================================
def _extract_json(text: str) -> str:
    t0 = (text or "").strip()
    if "```" in t0:
        t0 = t0.replace("```json", "```")
        parts = t0.split("```")
        if len(parts) >= 3:
            t0 = parts[1].strip()
    i, j = t0.find("{"), t0.rfind("}")
    return t0[i:j + 1] if i != -1 and j != -1 and j > i else t0


def _gemini_key() -> str:
    k = os.getenv("GEMINI_API_KEY", "").strip()
    if k:
        return k
    try:
        return str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        return ""


def _gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite").strip()


def _compact_for_llm(results: dict) -> dict:
    ec = results.get("energy_cost", {})
    co2 = results.get("co2", {})
    dv = results.get("diesel_vs_ev", {})
    dist = results.get("distance", {})
    load = results.get("load", {})
    inp = results.get("inputs", {})
    cap = results.get("capacity_analysis", {})
    prof = results.get("profile", {})
    return {
        "inputs": {
            "num_trucks": inp.get("num_trucks"),
            "operating_days": inp.get("operating_days"),
            "battery_kwh": inp.get("battery_kwh"),
            "ev_consumption_kwh_per_km": inp.get("ev_consumption_kwh_per_km"),
            "avg_elec_price_eur_per_mwh": inp.get("avg_elec_price_eur_per_mwh"),
            "dynamic_price_share": inp.get("dynamic_price_share"),
            "diesel_price_eur_per_l": inp.get("diesel_price_eur_per_l"),
            "diesel_l_per_100km": inp.get("diesel_l_per_100km"),
            "toll_rate_eur_per_km": inp.get("toll_rate_eur_per_km"),
            "ev_toll_exempt": inp.get("ev_toll_exempt"),
        },
        "derived": {
            "km_per_truck_per_day_input": prof.get("derived_soc", {}).get("km_per_truck_per_day"),
            "soc_diff_per_event": prof.get("derived_soc", {}).get("soc_diff"),
            "site_peak_source": prof.get("site_peak_source"),
        },
        "key_results": {
            "annual_energy_mwh": ec.get("annual_energy_mwh"),
            "effective_price_eur_per_mwh": ec.get("effective_price_eur_per_mwh"),
            "annual_ev_cost_eur": ec.get("annual_cost_eur"),
            "baseline_diesel_cost_eur": dv.get("diesel_cost_baseline_eur"),
            "cost_savings_eur": dv.get("cost_savings_eur"),
            "total_savings_incl_toll_eur": dv.get("total_savings_incl_toll_eur"),
            "annual_ev_co2_kg": co2.get("annual_ev_co2_kg"),
            "baseline_diesel_co2_kg": dv.get("diesel_co2_baseline_kg"),
            "co2_savings_kg": dv.get("co2_savings_kg"),
            "annual_km_fleet": dist.get("annual_km_fleet"),
            "new_theoretical_peak_kw": load.get("new_theoretical_peak_kw"),
            "capacity_ok": load.get("capacity_ok"),
            "available_kw_at_peak": cap.get("available_kw_at_peak"),
            "max_trucks_energy_based": cap.get("max_trucks_energy_based"),
        }
    }


def call_gemini_assistant(user_msg: str, results: dict, current_step_name: str) -> dict:
    api_key = _gemini_key()
    if not api_key:
        return {"reply": "GEMINI_API_KEY is missing.", "update_inputs": None, "show_payload": False}

    model = _gemini_model()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    params = {"key": api_key}

    compact = _compact_for_llm(results)
    current_inputs = {k: st.session_state["inputs"].get(k) for k in sorted(ALLOWED_INPUT_KEYS)}

    lang = st.session_state.get("lang", DEFAULT_LANG)
    lang_instruction = "Reply in German." if lang == "DE" else "Reply in English."

    prompt = (
        "You are Eva, the user's EV site loading assistant (brand name: FleetMate).\n"
        f"{lang_instruction}\n"
        "Be crystal-clear, customer-friendly, and explain jargon briefly.\n"
        "If the user asks to change parameters, return them in update_inputs.\n"
        "Only use keys from CURRENT INPUTS.\n\n"
        f"CURRENT STEP: {current_step_name}\n\n"
        "Return ONLY valid JSON with exactly these keys:\n"
        "reply: string\n"
        "update_inputs: object or null (only include changed keys)\n"
        "show_payload: boolean\n\n"
        "CURRENT INPUTS:\n"
        f"{json.dumps(current_inputs, ensure_ascii=False)}\n\n"
        "MODEL RESULTS (compact):\n"
        f"{json.dumps(compact, ensure_ascii=False)}\n\n"
        "USER MESSAGE:\n"
        f"{user_msg}\n"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.25, "maxOutputTokens": 420},
    }

    st.session_state["last_gemini_payload"] = {"url": url, "params": params, "payload": payload, "model": model}

    try:
        resp = requests.post(url, params=params, json=payload, timeout=25)
    except Exception as e:
        return {"reply": f"Gemini request failed: {e}", "update_inputs": None, "show_payload": False}

    if resp.status_code >= 300:
        return {"reply": f"Gemini error {resp.status_code}: {resp.text}", "update_inputs": None, "show_payload": False}

    data = resp.json()
    try:
        cands = data.get("candidates", [])
        parts = cands[0].get("content", {}).get("parts", []) if cands else []
        text = "\n".join([p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")]).strip()
    except Exception:
        text = ""

    if not text:
        return {"reply": "Gemini returned no text.", "update_inputs": None, "show_payload": False}

    try:
        obj = json.loads(_extract_json(text))
    except Exception:
        return {"reply": text, "update_inputs": None, "show_payload": False}

    upd = obj.get("update_inputs", None)
    if isinstance(upd, dict):
        upd = {k: v for k, v in upd.items() if k in ALLOWED_INPUT_KEYS}
    else:
        upd = None

    return {
        "reply": str(obj.get("reply", "")).strip() or "OK.",
        "update_inputs": upd,
        "show_payload": bool(obj.get("show_payload", False)),
    }


def call_gemini_report(results: dict, issues: list, solutions: list) -> str:
    api_key = _gemini_key()
    if not api_key:
        return ""

    model = _gemini_model()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    params = {"key": api_key}

    lang = st.session_state.get("lang", DEFAULT_LANG)
    lang_instruction = "Write in German." if lang == "DE" else "Write in English."

    compact = _compact_for_llm(results)
    prompt = (
        "You are Eva, writing a customer-friendly EV charging site report (brand: FleetMate).\n"
        f"{lang_instruction}\n"
        "Write in short sections with headings. Avoid excessive numbers; keep only the most important.\n"
        "Explain what matters and what to do next. Be confident but honest.\n\n"
        "Return ONLY Markdown (no JSON).\n\n"
        "Key data:\n"
        f"{json.dumps(compact, ensure_ascii=False)}\n\n"
        "Detected issues:\n"
        f"{json.dumps(issues, ensure_ascii=False)}\n\n"
        "Recommended actions:\n"
        f"{json.dumps(solutions, ensure_ascii=False)}\n\n"
        "Structure:\n"
        "1) Executive Summary (3–5 bullets)\n"
        "2) What your numbers mean (short paragraphs)\n"
        "3) Constraints & Risks (bullets)\n"
        "4) Recommended next steps (bullets)\n"
        "5) Questions to confirm with the customer (bullets)\n"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.25, "maxOutputTokens": 750},
    }

    try:
        resp = requests.post(url, params=params, json=payload, timeout=30)
    except Exception:
        return ""

    if resp.status_code >= 300:
        return ""

    data = resp.json()
    try:
        cands = data.get("candidates", [])
        parts = cands[0].get("content", {}).get("parts", []) if cands else []
        text = "\n".join([p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")]).strip()
    except Exception:
        text = ""
    return text or ""


def call_gemini_markdown(prompt: str) -> str:
    api_key = _gemini_key()
    if not api_key:
        return ""

    model = _gemini_model()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    params = {"key": api_key}

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.25, "maxOutputTokens": 750},
    }

    try:
        resp = requests.post(url, params=params, json=payload, timeout=30)
        if resp.status_code >= 300:
            return ""
        data = resp.json()
        cands = data.get("candidates", [])
        parts = cands[0].get("content", {}).get("parts", []) if cands else []
        text = "\n".join([p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")]).strip()
        return text or ""
    except Exception:
        return ""


# =========================================================
# UI helpers + formatting
# =========================================================
def fmt_eur(x):
    try:
        return f"{float(x):,.0f} €"
    except Exception:
        return "—"


def fmt_num(x):
    try:
        return f"{float(x):,.0f}"
    except Exception:
        return "—"


def fmt_kg(x):
    try:
        return f"{float(x):,.0f} kg"
    except Exception:
        return "—"


def gs_hr():
    st.markdown("<hr class='gs-hr'/>", unsafe_allow_html=True)


# =========================================================
# PDF export (unchanged)
# =========================================================
def _md_to_text(md: str) -> str:
    """Small markdown-to-text cleanup for PDF export."""
    if not md:
        return ""
    txt = md
    txt = re.sub(r"```.*?```", "", txt, flags=re.S)
    txt = re.sub(r"^#{1,6}\s*", "", txt, flags=re.M)
    txt = txt.replace("**", "").replace("*", "")
    txt = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


def build_report_pdf_bytes(title: str, kpis: dict, narrative_md: str, issues: list, solutions: list, lang: str, charts: list = None) -> bytes:
    """Build a clean PDF using reportlab. Returns b'' if reportlab is missing."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import Image as RLImage, PageBreak
        from reportlab.lib.utils import ImageReader
    except Exception:
        return b""

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2.0 * cm,
        rightMargin=2.0 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title=title,
    )

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=16, leading=20, spaceAfter=10)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12.5, leading=15, spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontName="Helvetica", fontSize=10.2, leading=13.5, spaceAfter=6)
    meta = ParagraphStyle("meta", parent=body, textColor=colors.grey, spaceAfter=8)

    story = []
    story.append(Paragraph(title, h1))
    story.append(Paragraph(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), meta))

    # KPI table
    kpi_rows = [[("KPI" if lang == "EN" else "Kennzahl"), ("Value" if lang == "EN" else "Wert")]]
    for k, v in (kpis or {}).items():
        kpi_rows.append([str(k), str(v)])

    tbl = Table(kpi_rows, colWidths=[9.0 * cm, 6.0 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 10),
        ("BACKGROUND", (0, 1), (-1, -1), colors.Color(0.98, 0.98, 1.0)),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 10))

    # Narrative
    story.append(Paragraph("Report" if lang == "EN" else "Report", h2))
    txt = _md_to_text(narrative_md)
    for para in [p.strip() for p in txt.split("\n\n") if p.strip()]:
        story.append(Paragraph(para.replace("\n", "<br/>"), body))

    # Issues
    story.append(Paragraph("Constraints & Risks" if lang == "EN" else "Constraints & Risiken", h2))
    if not issues:
        story.append(Paragraph("No major constraints detected." if lang == "EN" else "Keine großen Constraints erkannt.", body))
    else:
        for it in issues:
            story.append(Paragraph("• " + str(it.get("description", "")), body))

    # Solutions
    story.append(Paragraph("Recommended next steps" if lang == "EN" else "Empfohlene nächste Schritte", h2))
    if not solutions:
        story.append(Paragraph("—", body))
    else:
        for i, s in enumerate(solutions, 1):
            story.append(Paragraph(f"{i}. <b>{s.get('title','')}</b> — {s.get('definition','')}", body))

    # Charts (optional)
    if charts:
        story.append(PageBreak())
        story.append(Paragraph("Charts", h2))
        for title_c, png in charts:
            story.append(Paragraph(str(title_c), body))
            try:
                img_reader = ImageReader(io.BytesIO(png))
                iw, ih = img_reader.getSize()
                max_w = 16.5 * cm
                scale = max_w / float(iw)
                w = max_w
                h = float(ih) * scale
                story.append(RLImage(io.BytesIO(png), width=w, height=h))
                story.append(Spacer(1, 10))
            except Exception:
                story.append(Paragraph("Chart rendering failed.", body))

    doc.build(story)
    return buf.getvalue()


# =========================================================
# App state utilities
# =========================================================
def reset_all():
    lang = st.session_state.get("lang", DEFAULT_LANG)
    theme = st.session_state.get("theme", DEFAULT_THEME)  # English comment: keep theme on reset

    st.session_state["inputs"] = DEFAULT_INPUTS.copy()
    st.session_state["lang"] = lang
    st.session_state["wizard_step"] = 0
    st.session_state["flow"] = "wizard"
    st.session_state["report_md"] = None
    st.session_state["report_meta"] = None
    st.session_state["report_pdf_bytes"] = None
    st.session_state["assistant_messages"] = []
    st.session_state["last_gemini_payload"] = None
    st.session_state["profile_cache"] = {
        "digest": None,
        "df": None,
        "timestamp_col": None,
        "consumption_col": None,
        "meta": {},
        "metrics": {},
    }


def apply_pending_flow():
    """Apply deferred flow changes BEFORE the flow radio is instantiated."""
    v = st.session_state.pop("pending_flow_switch", None)
    if v in ("wizard", "report"):
        st.session_state["flow"] = v
        if "flow_switch" in st.session_state:
            del st.session_state["flow_switch"]


def ensure_assistant_greeting():
    if st.session_state["assistant_messages"]:
        return
    lang = st.session_state.get("lang", DEFAULT_LANG)
    greeting = "Hi — I’m Eva. Ask me anything while you fill this in." if lang == "EN" else "Hi — ich bin Eva. Frag mich jederzeit während du das ausfüllst."
    st.session_state["assistant_messages"].append({"role": "assistant", "content": greeting})


def handle_user_chat(msg: str, current_step_name: str):
    if not msg:
        return
    ensure_assistant_greeting()
    st.session_state["assistant_messages"].append({"role": "user", "content": msg})

    recalc_from_inputs()
    results = st.session_state.get("model_results")
    if not results:
        st.session_state["assistant_messages"].append({"role": "assistant", "content": "Run at least one step so I have results to work with."})
        return

    out = call_gemini_assistant(msg, results, current_step_name)
    st.session_state["assistant_messages"].append({"role": "assistant", "content": out["reply"]})

    if out.get("update_inputs"):
        for k, v in out["update_inputs"].items():
            st.session_state["inputs"][k] = v
        recalc_from_inputs()
        ack = "✅ Updated inputs and recalculated." if st.session_state.get("lang") == "EN" else "✅ Inputs aktualisiert und neu berechnet."
        st.session_state["assistant_messages"].append({"role": "assistant", "content": ack})


# =========================================================
# Tables & chart dataframes
# =========================================================
def build_calculation_df(res: dict) -> pd.DataFrame:
    inp = res["inputs"]
    ec = res["energy_cost"]
    dv = res["diesel_vs_ev"]
    co2 = res["co2"]
    dist = res["distance"]
    load = res["load"]
    cap = res.get("capacity_analysis", {})
    prof = res.get("profile", {}).get("derived_soc", {})

    annual_km = float(dist.get("annual_km_fleet", 0.0))
    annual_energy_mwh = float(ec.get("annual_energy_mwh", 0.0))
    annual_cost = float(ec.get("annual_cost_eur", 0.0))

    cost_per_km = (annual_cost / annual_km) if annual_km > 0 else np.nan
    ev_co2_per_km = (float(co2.get("annual_ev_co2_kg", 0.0)) / annual_km) if annual_km > 0 else np.nan
    diesel_co2_per_km = (float(dv.get("diesel_co2_baseline_kg", 0.0)) / annual_km) if annual_km > 0 else np.nan

    rows = [
        ("km/day input", "km_per_truck_per_day", prof.get("km_per_truck_per_day", np.nan), "km/day"),
        ("Energy per truck/day", "km/day * kWh/km", prof.get("daily_energy_kwh", np.nan), "kWh/day"),
        ("Energy per event", "(km/day/events) * kWh/km", prof.get("energy_per_event_kwh", np.nan), "kWh"),
        ("SoC diff per event", "energy_event / battery_kwh", prof.get("soc_diff", np.nan), "0–1"),
        ("Daily energy (fleet)", "num_trucks * events/day * energy/event", float(ec.get("total_daily_energy_mwh", 0.0)) * 1000.0, "kWh/day"),
        ("Annual energy", "daily_energy * operating_days", annual_energy_mwh, "MWh/year"),
        ("Effective electricity price", "fixed or dynamic (window)", float(ec.get("effective_price_eur_per_mwh", 0.0)), "€/MWh"),
        ("EV electricity cost", "annual_energy_mwh * effective_price", annual_cost, "€/year"),
        ("Diesel cost baseline", "annual_km * L/100km * diesel_price", float(dv.get("diesel_cost_baseline_eur", 0.0)), "€/year"),
        ("Baseline toll cost", "annual_km * tolled_share * toll_rate", float(dv.get("baseline_toll_cost_eur", 0.0)), "€/year"),
        ("Total savings incl. toll", "diesel_cost - ev_cost + toll_savings", float(dv.get("total_savings_incl_toll_eur", 0.0)), "€/year"),
        ("Theoretical peak (new)", "site_peak + (num_trucks * charger_kw)", float(load.get("new_theoretical_peak_kw", 0.0)), "kW"),
        ("Available kW at peak", "site_capacity - site_peak_in_window", float(cap.get("available_kw_at_peak", np.nan)), "kW"),
        ("Max trucks (energy-based)", "(avail_kw*window_h)/kWh_per_truck_day", float(cap.get("max_trucks_energy_based", np.nan)), "trucks"),
        ("Cost per km (EV)", "annual_ev_cost / annual_km", cost_per_km, "€/km"),
        ("CO₂ per km (EV)", "annual_ev_co2 / annual_km", ev_co2_per_km, "kg/km"),
        ("CO₂ per km (diesel)", "diesel_co2 / annual_km", diesel_co2_per_km, "kg/km"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Formula", "Value", "Unit"])


def build_hourly_df(res: dict) -> pd.DataFrame:
    """
    Hourly view for charts.

    If a load profile was uploaded, we use its hourly_max_kw as baseline (conservative).
    Otherwise we fall back to a flat baseline = existing_site_peak_kw.
    """
    inp = res["inputs"]
    prof = res.get("profile", {})
    ec = res["energy_cost"]
    charging = res["charging_profile"]

    flags = charging["flags"]
    shares = charging["shares"]

    total_daily_kwh = float(ec["total_daily_energy_mwh"]) * 1000.0
    clock_hours = int(sum(flags)) if sum(flags) > 0 else 24
    power_if_spread_kw = (total_daily_kwh / clock_hours) if clock_hours > 0 else 0.0

    # Effective average charging power (to deliver energy within effective charging time)
    eff_power_kw = (total_daily_kwh / float(inp["charging_window_hours"])) if inp["charging_window_hours"] > 0 else 0.0

    ev_kw = [power_if_spread_kw if flags[h] else 0.0 for h in range(24)]
    ev_kwh = [ev_kw[h] for h in range(24)]  # 1h bins

    # Baseline load
    if prof.get("hourly_max_kw"):
        base = [float(x) for x in prof["hourly_max_kw"]]
        base_label = "Baseline (from upload)"
    else:
        base = [float(inp["existing_site_peak_kw"])] * 24
        base_label = "Baseline (manual)"

    site_load_kw_spread = [base[h] + ev_kw[h] for h in range(24)]

    df = pd.DataFrame({
        "hour": list(range(24)),
        "in_window": [bool(f) for f in flags],
        "share": shares,
        "price_eur_per_kwh": charging["tou_price_eur_per_kwh"],  # may be spot avg if uploaded
        "grid_co2_g_per_kwh": charging["grid_co2_g_per_kwh"],
        "ev_power_kw_spread": ev_kw,
        "ev_energy_kwh": ev_kwh,
        "site_load_baseline_kw": base,
        "site_load_kw_spread": site_load_kw_spread,
        "eff_power_kw_required": [eff_power_kw] * 24,
        "baseline_label": [base_label] * 24,
    })
    return df


# =========================================================
# Charts
# =========================================================
def fig_price_co2(df: pd.DataFrame, label_price: str):
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()

    ax1.plot(df["hour"], df["price_eur_per_kwh"], marker="o")
    ax2.plot(df["hour"], df["grid_co2_g_per_kwh"], marker="s")

    ax1.set_xlabel("Hour")
    ax1.set_ylabel(label_price)
    ax2.set_ylabel("Grid CO₂ (g/kWh)")

    # Window shading
    inw = df["in_window"].to_numpy()
    if inw.any():
        for h in range(24):
            if inw[h]:
                ax1.axvspan(h - 0.5, h + 0.5, alpha=0.08)

    ax1.set_xticks(range(0, 24, 1))
    ax1.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def fig_load(df: pd.DataFrame, capacity_kva: float, theoretical_peak_kw: float):
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(df["hour"], df["site_load_baseline_kw"], marker="o", label="Site baseline load")
    ax.plot(df["hour"], df["site_load_kw_spread"], marker="o", label="Site load + EV (spread in window)")

    if capacity_kva and capacity_kva > 0:
        ax.axhline(capacity_kva, linestyle="--", label="Site capacity limit (kVA≈kW)")
    ax.axhline(theoretical_peak_kw, linestyle=":", label="Theoretical peak (all chargers @ max)")

    ax.set_xlabel("Hour")
    ax.set_ylabel("kW")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def fig_waterfall(res: dict):
    dv = res["diesel_vs_ev"]
    diesel = float(dv["diesel_cost_baseline_eur"])
    ev = float(dv["ev_cost_eur"])
    toll = float(dv["toll_savings_eur"])
    total = float(dv["total_savings_incl_toll_eur"])

    labels = ["Diesel cost", "EV electricity cost", "Toll savings", "Total savings"]
    vals = [diesel, -ev, toll, total]

    fig, ax = plt.subplots(figsize=(10, 4))
    cum = 0.0
    xs = np.arange(len(labels))
    for i, v in enumerate(vals[:-1]):
        ax.bar(xs[i], v, bottom=cum)
        cum += v
    ax.bar(xs[-1], vals[-1])

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("€/year")
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def fig_finance_bars(res: dict):
    """Two simple bar charts: (1) Diesel vs EV costs, (2) Toll effect. Order matches explanation text."""
    dv = res["diesel_vs_ev"]
    diesel_cost = float(dv.get("diesel_cost_baseline_eur", 0.0))
    ev_cost = float(dv.get("ev_cost_eur", 0.0))
    total_savings = float(dv.get("total_savings_incl_toll_eur", 0.0))
    toll_base = float(dv.get("baseline_toll_cost_eur", 0.0))
    toll_sav = float(dv.get("toll_savings_eur", 0.0))
    toll_ev = toll_base - toll_sav

    # Chart 1
    fig1, ax1 = plt.subplots(figsize=(7.5, 4))
    ax1.bar(["Diesel (baseline)", "EV (electricity)"], [diesel_cost, ev_cost])
    ax1.axhline(0, linestyle="--", alpha=0.3)
    ax1.set_ylabel("€/year")
    ax1.set_title("Annual energy cost comparison")
    ax1.grid(True, axis="y", alpha=0.2)
    fig1.tight_layout()

    # Chart 2 (swapped into second position to align with narrative)
    fig2, ax2 = plt.subplots(figsize=(7.5, 4))
    ax2.bar(["Toll (diesel)", "Toll (EV)"], [toll_base, toll_ev])
    ax2.set_ylabel("€/year")
    ax2.set_title("Toll effect")
    ax2.grid(True, axis="y", alpha=0.2)
    fig2.tight_layout()

    return fig1, fig2

def fig_pareto_shifted_windows(res: dict):
    """Trade-off: shift the same window length across the day, plot EV cost vs EV CO2."""
    try:
        # --- keep original window length ---
        start0 = int(res["inputs"].get("start_hour", 0)) % 24
        end0 = int(res["inputs"].get("end_hour", 0)) % 24
        flags0, _ = compute_flags_and_shares(start0, end0)
        win_len = int(sum(flags0)) if sum(flags0) > 0 else 24

        # --- base kwargs WITHOUT start/end (fix for TypeError) ---
        base_inp = res["inputs"].copy()
        base_inp.pop("start_hour", None)
        base_inp.pop("end_hour", None)

        # --- price curve: if you uploaded spot prices, this is already overridden in results ---
        price_curve = res.get("charging_profile", {}).get("tou_price_eur_per_kwh") or TOU_PRICE_EUR_PER_KWH
        if not isinstance(price_curve, (list, tuple)) or len(price_curve) != 24:
            price_curve = TOU_PRICE_EUR_PER_KWH

        curve_avg = float(sum(float(x) for x in price_curve)) / 24.0

        points = []
        for start in range(24):
            end = (start + win_len) % 24

            # compute window avg for THIS shifted window (important for meaningful Pareto!)
            _, shares = compute_flags_and_shares(start, end)
            window_avg = float(sum(float(price_curve[h]) * shares[h] for h in range(24)))

            r = run_model(
                **base_inp,
                start_hour=start,
                end_hour=end,
                # feed per-window spot stats (even if fixed mode: harmless)
                spot_curve_avg_eur_per_kwh=curve_avg,
                spot_window_avg_eur_per_kwh=window_avg,
            )

            points.append({
                "start": start,
                "end": end,
                "cost": float(r["energy_cost"]["annual_cost_eur"]),
                "co2": float(r["co2"]["annual_ev_co2_kg"]),
            })

        df = pd.DataFrame(points)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(df["cost"], df["co2"])
        for _, row in df.iterrows():
            ax.annotate(f'{int(row["start"]):02d}', (row["cost"], row["co2"]), fontsize=8)

        ax.set_xlabel("EV cost (€/year)")
        ax.set_ylabel("EV CO₂ (kg/year)")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        st.session_state["pareto_df"] = df
        st.session_state["pareto_win_len"] = win_len
        return fig

    except Exception as e:
        # never crash the whole report page because of this chart
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.axis("off")
        ax.text(0.01, 0.6, "Pareto chart unavailable.", fontsize=12)
        ax.text(0.01, 0.25, f"Reason: {e}", fontsize=9)
        fig.tight_layout()
        return fig


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

def inject_theme_css(theme: str):
    theme = (theme or "system").strip().lower()

    THEMES = {
        "dark": {
            "--bg1": "#070a12",
            "--bg2": "#070a12",
            "--card": "rgba(255,255,255,0.10)",
            "--card2": "rgba(255,255,255,0.14)",
            "--stroke": "rgba(255,255,255,0.16)",
            "--a1": "rgba(58,141,255,0.50)",
            "--a2": "rgba(255,152,0,0.32)",
            "--a3": "rgba(168,85,247,0.30)",
            "--a4": "rgba(34,211,238,0.22)",
            "--glass": "rgba(255,255,255,0.14)",
            "--glass2": "rgba(255,255,255,0.08)",
            "--txt": "rgba(255,255,255,0.92)",
            "--muted": "rgba(255,255,255,0.70)",
            "--inputBg": "rgba(255,255,255,0.92)",
            "--inputTxt": "#0b0d12",
            "--placeholder": "rgba(20,20,30,0.55)",
        },
        "light": {
            "--bg1": "#f6f7fb",
            "--bg2": "#eef1f7",
            "--card": "rgba(255,255,255,0.70)",
            "--card2": "rgba(255,255,255,0.86)",
            "--stroke": "rgba(10,12,18,0.10)",
            "--a1": "rgba(58,141,255,0.22)",
            "--a2": "rgba(255,152,0,0.14)",
            "--a3": "rgba(168,85,247,0.14)",
            "--a4": "rgba(34,211,238,0.12)",
            "--glass": "rgba(255,255,255,0.72)",
            "--glass2": "rgba(255,255,255,0.52)",
            "--txt": "rgba(10,12,18,0.92)",
            "--muted": "rgba(10,12,18,0.62)",
            "--inputBg": "rgba(255,255,255,0.96)",
            "--inputTxt": "#0b0d12",
            "--placeholder": "rgba(10,12,18,0.45)",
        },
    }

    def vars_block(d: dict) -> str:
        return "\n".join([f"  {k}: {v};" for k, v in d.items()])

    dark_vars = vars_block(THEMES["dark"])
    light_vars = vars_block(THEMES["light"])

    if theme == "system":
        css = f"""
        <style>
        :root {{
        {dark_vars}
        }}
        html, body {{ color-scheme: dark; }}

        @media (prefers-color-scheme: light) {{
          :root {{
          {light_vars}
          }}
          html, body {{ color-scheme: light; }}
        }}

        /* BaseWeb inputs readable in both modes */
        div[data-baseweb="input"] input,
        div[data-baseweb="textarea"] textarea {{
          background: var(--inputBg) !important;
          color: var(--inputTxt) !important;
        }}
        div[data-baseweb="input"] input::placeholder,
        div[data-baseweb="textarea"] textarea::placeholder {{
          color: var(--placeholder) !important;
        }}
        </style>
        """
    else:
        t = THEMES["light"] if theme == "light" else THEMES["dark"]
        scheme = "light" if theme == "light" else "dark"
        css = f"""
        <style>
        :root {{
        {vars_block(t)}
        }}
        html, body {{ color-scheme: {scheme}; }}

        div[data-baseweb="input"] input,
        div[data-baseweb="textarea"] textarea {{
          background: var(--inputBg) !important;
          color: var(--inputTxt) !important;
        }}
        div[data-baseweb="input"] input::placeholder,
        div[data-baseweb="textarea"] textarea::placeholder {{
          color: var(--placeholder) !important;
        }}
        </style>
        """

    st.markdown(css, unsafe_allow_html=True)

def tornado_takeaways(df, top_n=3):
    d = df.sort_values("Range", ascending=False).head(top_n)
    out = []
    for _, r in d.iterrows():
        direction = "steigt" if r["Delta_high"] > 0 else "fällt"
        out.append(
            f"- **{r['Parameter']}** ist ein Top-Hebel (Spannweite ~{r['Range']:,.0f} €/Jahr). "
            f"Wenn der Wert **hoch** geht, {direction} die Ersparnis."
        )
    out.append("- Das ist **ceteris paribus**: jeweils nur **ein** Parameter wird verändert.")
    return "\n".join(out)


def fmt_eur_per_kwh(x):
    try:
        return f"{float(x):,.3f} €/kWh"
    except Exception:
        return "—"

def fmt_eur_per_mwh(x):
    try:
        return f"{float(x):,.1f} €/MWh"
    except Exception:
        return "—"

def chart_note(lines: list[str], metrics: dict[str, str] | None = None, expanded: bool = False):
    """Uniformer Erklärungstext unter Charts + optionale KPI-Zeile."""
    with st.expander("Kurz erklärt (mit Werten)", expanded=expanded):
        if lines:
            st.markdown("\n".join([f"- {l}" for l in lines]))
        if metrics:
            cols = st.columns(min(4, len(metrics)))
            for i, (k, v) in enumerate(metrics.items()):
                cols[i % len(cols)].metric(k, v)

def _inputs_digest_for_tornado(inp: dict) -> str:
    # only include what affects results materially
    keys = [
        "num_trucks","operating_days","events_per_truck","battery_kwh","ev_consumption",
        "km_per_truck_per_day","avg_elec_price_mwh","dynamic_share",
        "start_hour","end_hour","existing_peak_kw","charger_power_kw","site_capacity_kva",
        "diesel_price","diesel_l_per_100","toll_rate","tolled_share","ev_toll_exempt",
    ]
    payload = {k: inp.get(k) for k in keys}
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


# =========================================================
# Page config + styling (kept)
# =========================================================
st.set_page_config(page_title="FleetMate — Guided Intake", layout="wide")
require_login()


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

:root{
  --bg1:#070a12; --bg2:#070a12;
  --card: rgba(255,255,255,0.10);
  --card2: rgba(255,255,255,0.14);
  --stroke: rgba(255,255,255,0.16);
  --shadow2: 0 18px 44px rgba(0,0,0,0.35);
  --shadow3: 0 12px 28px rgba(0,0,0,0.28);
  --radius: 18px;

  --a1: rgba(58,141,255,0.50);
  --a2: rgba(255,152,0,0.32);
  --a3: rgba(168,85,247,0.30);
  --a4: rgba(34,211,238,0.22);

  --glass: rgba(255,255,255,0.14);
  --glass2: rgba(255,255,255,0.08);

  --txt: rgba(255,255,255,0.92);
  --muted: rgba(255,255,255,0.70);
}


html, body, [class*="css"]  {
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
}

.stApp{
  color: var(--txt);
  background:
    radial-gradient(1200px 680px at 8% 10%, var(--a1), transparent 55%),
    radial-gradient(1150px 780px at 92% 0%, var(--a2), transparent 55%),
    radial-gradient(980px 720px at 82% 88%, var(--a3), transparent 55%),
    radial-gradient(920px 680px at 15% 85%, var(--a4), transparent 55%),
    linear-gradient(180deg, var(--bg1), var(--bg2));
}

.block-container{
  padding-top: 2.2rem;
  padding-bottom: 2.8rem;
  max-width: 100% !important;     /* fluid */
  padding-left: 2.2rem;
  padding-right: 2.2rem;
}

@media (max-width: 900px){
  .block-container{ padding-left: 1rem; padding-right: 1rem; }
}

div[data-testid="stVerticalBlock"]{gap: 0.75rem;}

/* Base: containers are neutral (prevents “everything is a card”) */
div[data-testid="stContainer"]{
  background: transparent;
  border: none !important;
  box-shadow: none;
  padding: 0;
}

/* Card look ONLY if the container contains our marker */
div[data-testid="stContainer"]:has(.gs-card-marker){
  background: linear-gradient(180deg, var(--card2), var(--card));
  border: 1px solid var(--stroke) !important;
  border-radius: var(--radius) !important;
  box-shadow: var(--shadow3);
  padding: 1.05rem 1.1rem;
  backdrop-filter: blur(14px);
}

.gs-card-marker{ display:none; }


div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea{
  background: rgba(255,255,255,0.92) !important;
  color: #0b0d12 !important;
  border-color: rgba(0,0,0,0.18) !important;
}
div[data-baseweb="input"] input::placeholder,
div[data-baseweb="textarea"] textarea::placeholder{
  color: rgba(20,20,30,0.55) !important;
}

/* Ensure general text stays readable */
label, .stMarkdown, .stCaption {color: var(--txt) !important;}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: var(--muted) !important; }
.small-note{font-size:0.90rem; color: var(--muted);}

.gs-hr{border:none; height:1px; background: rgba(255,255,255,0.14); margin: 0.55rem 0;}
.gs-title{margin:0; font-size: 2.08rem; letter-spacing:-0.02em;}
.gs-sub{color: var(--muted); margin:0.25rem 0 0 0;}

.gs-pill{
  display:inline-flex; align-items:center; gap:.4rem;
  padding: .30rem .70rem; border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.16);
  background: rgba(255,255,255,0.10);
  font-size: .86rem;
}

.stButton > button{
  border-radius: 999px;
  padding: 0.62rem 1.0rem;
  border: 1px solid rgba(255,255,255,0.18);
  color: var(--txt);
  background: linear-gradient(180deg, var(--glass), var(--glass2));
  box-shadow: var(--shadow2);
  backdrop-filter: blur(12px);
}
.stButton > button:hover{
  transform: translateY(-1px);
  box-shadow: 0 22px 58px rgba(0,0,0,0.45);
}
.stButton > button:active{transform: translateY(0px);}

div[role="radiogroup"]{gap: 0.55rem;}

/* Fix radio label text (e.g. "Fragebogen / Report", "DE / EN") */
div[data-testid="stRadio"] label,
div[data-testid="stRadio"] label *,
div[role="radiogroup"] label,
div[role="radiogroup"] label * {
  color: var(--txt) !important;
}

/* Fix tabs text (e.g. "Übersicht / Finanzen / Charts / Berechnungen ...") */
div[data-testid="stTabs"] button,
div[data-testid="stTabs"] button *,
div[data-baseweb="tab"] button,
div[data-baseweb="tab"] button * {
  color: var(--muted) !important;
}

/* Active tab brighter */
div[data-testid="stTabs"] button[aria-selected="true"],
div[data-testid="stTabs"] button[aria-selected="true"] *,
div[data-baseweb="tab"][aria-selected="true"] button,
div[data-baseweb="tab"][aria-selected="true"] button * {
  color: var(--txt) !important;
}

div[data-testid="stMetric"] *{ color: var(--txt) !important; }
div[data-testid="stMetricLabel"]{ color: var(--muted) !important; }
div[data-testid="stMetricDelta"]{ color: var(--muted) !important; }

details summary, details summary *{ color: var(--txt) !important; }

.stDownloadButton > button{
  border-radius: 999px !important;
  padding: 0.62rem 1.0rem !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
  color: var(--txt) !important;
  background: linear-gradient(180deg, var(--glass), var(--glass2)) !important;
  box-shadow: var(--shadow2) !important;
  backdrop-filter: blur(12px) !important;
}
</style>
""", unsafe_allow_html=True)



# =========================================================
# Wizard steps
# =========================================================
STEPS = [
    {"id": "fleet", "title_en": "Fleet & Usage", "title_de": "Flotte & Nutzung",
     "sub_en": "How much charging your operation needs.", "sub_de": "Wie viel Laden dein Betrieb braucht."},
    {"id": "battery", "title_en": "Battery & Consumption", "title_de": "Batterie & Verbrauch",
     "sub_en": "Derive energy need from km/day (no SoC inputs).", "sub_de": "Energiebedarf aus km/Tag ableiten (keine SoC-Inputs)."},
    {"id": "electricity", "title_en": "Electricity & Charging Window", "title_de": "Strom & Ladefenster",
     "sub_en": "Fixed vs dynamic price + typical charging hours.", "sub_de": "Fix vs dynamisch + typische Ladezeiten."},
    {"id": "site", "title_en": "Site Capacity & Load Profile", "title_de": "Standortkapazität & Lastprofil",
     "sub_en": "Upload load profile + check headroom during your charging window.", "sub_de": "Lastprofil hochladen + Headroom im Ladefenster prüfen."},
    {"id": "diesel", "title_en": "Diesel Baseline & Toll", "title_de": "Diesel-Baseline & Maut",
     "sub_en": "So the business case compares fairly.", "sub_de": "Damit der Vergleich fair ist."},
]
STEP_BY_ID = {s["id"]: i for i, s in enumerate(STEPS)}


def step_title(step):
    return step["title_de"] if st.session_state.get("lang") == "DE" else step["title_en"]


def step_sub(step):
    return step["sub_de"] if st.session_state.get("lang") == "DE" else step["sub_en"]


def current_step():
    i = int(st.session_state.get("wizard_step", 0))
    i = max(0, min(i, len(STEPS) - 1))
    return i, STEPS[i]


# =========================================================
# Render
# =========================================================
ensure_defaults()
apply_pending_flow()
ensure_assistant_greeting()

# Always recalc early (uses upload cache if available)
recalc_from_inputs()

top_l, top_r = st.columns([0.70, 0.30], vertical_alignment="center")
with top_l:
    st.markdown(f"<h1 class='gs-title'>{t('welcome_title')}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='gs-sub'>{t('welcome_sub')}</p>", unsafe_allow_html=True)

with top_r:
    c_lang, c_theme = st.columns([0.55, 0.45], vertical_alignment="center")

    with c_lang:
        current_lang = st.session_state.get("lang", DEFAULT_LANG)
        lang = st.radio(
            "",
            options=["DE", "EN"],
            index=0 if current_lang == "DE" else 1,
            horizontal=True,
            label_visibility="collapsed",
            key="lang_switch",
        )
        if lang != current_lang:
            st.session_state["lang"] = lang
            st.session_state["charts_md"] = None
            st.session_state["report_md"] = None
            st.session_state["report_meta"] = None
            st.session_state["report_pdf_bytes"] = None
            st.rerun()

    with c_theme:
        # English comment: simple theme selector (kept minimal)
        current_theme = st.session_state.get("theme", DEFAULT_THEME)
        theme = st.radio(
            "",
            options=["system", "dark", "light"],
            index=["system","dark","light"].index(st.session_state.get("theme", DEFAULT_THEME)),
            horizontal=True,
            format_func=lambda v: "🖥️" if v=="system" else ("🌙" if v=="dark" else "☀️"),
            label_visibility="collapsed",
            key="theme_switch",
        )
        if theme != st.session_state.get("theme"):
            st.session_state["theme"] = theme
            st.rerun()

    # IMPORTANT: inject theme CSS AFTER the switch so it applies immediately
    inject_theme_css(st.session_state.get("theme", DEFAULT_THEME))


left, right = st.columns([0.62, 0.38], gap="large")

# ---------- Right: Live preview + Assistant ----------
with right:
    i, step = current_step()
    step_name = step_title(step)

    with st.container(border=True):
        st.markdown("<span class='gs-card-marker'></span>", unsafe_allow_html=True)

        st.markdown(f"### {t('live_preview')}")
        res = st.session_state.get("model_results")
        if not res:
            st.caption("—")
        else:
            dv = res["diesel_vs_ev"]
            load = res["load"]

            c1, c2 = st.columns(2)
            with c1:
                st.metric(t("kpi_savings"), fmt_eur(dv.get("total_savings_incl_toll_eur")), t("per_year"))
            with c2:
                st.metric(t("kpi_co2"), fmt_kg(dv.get("co2_savings_kg")), t("per_year"))

            cap_ok = bool(load.get("capacity_ok", True))
            pill = f"🟢 {t('capacity_ok_yes')}" if cap_ok else f"⛔ {t('capacity_ok_no')}"
            st.markdown(
                f"<span class='gs-pill'>⚡ {t('kpi_peak')}: {fmt_num(load.get('new_theoretical_peak_kw'))} kW</span> "
                f"&nbsp; <span class='gs-pill'>{pill}</span>",
                unsafe_allow_html=True
            )

            # New: capacity headroom from load profile window
            cap = res.get("capacity_analysis", {})
            if cap:
                st.markdown(
                    f"<span class='gs-pill'>🏭 Peak im Ladefenster: {fmt_num(cap.get('site_peak_kw_in_window'))} kW</span> "
                    f"&nbsp; <span class='gs-pill'>🧩 Headroom: {fmt_num(cap.get('available_kw_at_peak'))} kW</span>",
                    unsafe_allow_html=True
                )

    with st.container(border=True):
        st.markdown("<span class='gs-card-marker'></span>", unsafe_allow_html=True)

        st.markdown(f"### 💬 {t('assistant_title')}")
        st.markdown(f"<div class='small-note'>{t('assistant_hint')}</div>", unsafe_allow_html=True)

        for m in st.session_state["assistant_messages"][-18:]:
            with st.chat_message(m["role"]):
                st.write(m["content"])

        user_msg = st.chat_input(t("ask_placeholder"))
        if user_msg:
            handle_user_chat(user_msg, step_name)
            st.rerun()

# ---------- Left: Wizard or Report ----------
with left:
    i, step = current_step()
    pct = int(((i + 1) / len(STEPS)) * 100)

    mode_l, mode_r = st.columns([0.55, 0.45], vertical_alignment="center")
    with mode_l:
        mode = st.radio(
            "",
            ["wizard", "report"],
            horizontal=True,
            index=0 if st.session_state.get("flow") == "wizard" else 1,
            format_func=lambda v: t("questionnaire") if v == "wizard" else t("report"),
            label_visibility="collapsed",
            key="flow_switch",
        )
        if mode != st.session_state.get("flow"):
            st.session_state["flow"] = mode
            st.rerun()

    with mode_r:
        if st.button("↺ " + t("restart"), use_container_width=True, key="restart_btn_top"):
            reset_all()
            st.session_state["pending_flow_switch"] = "wizard"
            st.rerun()

    gs_hr()

    # ============ WIZARD ============
    if st.session_state.get("flow") == "wizard":
        st.markdown(f"## {step_title(step)}")
        st.markdown(f"<div class='small-note'>{step_sub(step)}</div>", unsafe_allow_html=True)
        st.progress(pct, text=f"{t('progress')}: {pct}%")

        with st.container(border=True):
            st.markdown("<span class='gs-card-marker'></span>", unsafe_allow_html=True)

            sid = step["id"]
            lang = st.session_state.get("lang", DEFAULT_LANG)

            if sid == "fleet":
                bind_number(
                    sid, "num_trucks",
                    "How many trucks charge per day?" if lang == "EN" else "Wie viele LKW laden pro Tag?",
                    min_value=1, max_value=10000, step=1,
                )
                bind_number(
                    sid, "operating_days",
                    "Operating days per year" if lang == "EN" else "Betriebstage pro Jahr",
                    min_value=1, max_value=366, step=1,
                )
                bind_number(
                    sid, "events_per_truck",
                    "Charging events per truck per day" if lang == "EN" else "Lade-Events pro LKW pro Tag",
                    min_value=0.25, max_value=10.0, step=0.25,
                    help=("Example: 1.0 = once per day; 2.0 = two separate charging events." if lang == "EN"
                          else "Beispiel: 1,0 = 1× pro Tag; 2,0 = 2 getrennte Lade-Events.")
                )
                st.markdown(
                    "<div class='small-note'>If trucks sometimes charge twice per day, set this above 1.0.</div>" if lang == "EN"
                    else "<div class='small-note'>Wenn manchmal 2× pro Tag geladen wird, setze den Wert über 1,0.</div>",
                    unsafe_allow_html=True
                )

            elif sid == "battery":
                PROFILES = {
                    "city": {
                        "label_en": "City / Last‑Mile", "label_de": "City / Last‑Mile",
                        "battery_kwh": 220.0, "ev_consumption": 0.8,
                        "desc_en": "Short routes, lots of stops. Typical consumption: 0.7–1.1 kWh/km.",
                        "desc_de": "Kurze Strecken, viele Stopps. Typischer Verbrauch: 0,7–1,1 kWh/km.",
                    },
                    "regional": {
                        "label_en": "Regional Distribution", "label_de": "Regionale Distribution",
                        "battery_kwh": 350.0, "ev_consumption": 0.9,
                        "desc_en": "Mixed urban + regional. Typical consumption: 0.8–1.2 kWh/km.",
                        "desc_de": "Mix aus Stadt + Region. Typischer Verbrauch: 0,8–1,2 kWh/km.",
                    },
                    "heavy_regional": {
                        "label_en": "Heavy Regional (mixed)", "label_de": "Schwer regional (gemischt)",
                        "battery_kwh": 500.0, "ev_consumption": 1.0,
                        "desc_en": "Heavier loads. Typical consumption: ~1.0–1.3 kWh/km.",
                        "desc_de": "Schwerere Lasten. Typischer Verbrauch: ~1,0–1,3 kWh/km.",
                    },
                    "long_haul": {
                        "label_en": "Long‑haul / 40t", "label_de": "Langstrecke / 40t",
                        "battery_kwh": 650.0, "ev_consumption": 1.1,
                        "desc_en": "Long-haul baseline (kept conservative). If you have measured data, fine-tune.",
                        "desc_de": "Langstrecke Basis (konservativ). Mit Messdaten bitte feinjustieren.",
                    },
                }

                bind_select(
                    sid, "vehicle_profile",
                    "Vehicle profile (recommended defaults)" if lang == "EN" else "Fahrprofil (empfohlene Defaults)",
                    options=list(PROFILES.keys()),
                    format_func=lambda k: PROFILES[k]["label_en"] if lang == "EN" else PROFILES[k]["label_de"],
                )
                prof = PROFILES.get(get_inp("vehicle_profile"), PROFILES["heavy_regional"])
                st.info(prof["desc_en"] if lang == "EN" else prof["desc_de"])

                c_apply, c_note = st.columns([0.42, 0.58])
                with c_apply:
                    if st.button("✨ " + t("apply_defaults"), use_container_width=True, key="apply_profile_defaults"):
                        set_inp("battery_kwh", prof["battery_kwh"])
                        set_inp("ev_consumption", prof["ev_consumption"])
                        st.session_state.pop(wkey("battery", "battery_kwh"), None)
                        st.session_state.pop(wkey("battery", "ev_consumption"), None)
                        st.rerun()
                with c_note:
                    st.markdown(
                        "<div class='small-note'>"
                        + ("Defaults are designed to be realistic. If you have fleet telematics, use measured values."
                           if lang == "EN"
                           else "Defaults sind realistisch angesetzt. Wenn du Flottendaten hast: gemessene Werte verwenden.")
                        + "</div>",
                        unsafe_allow_html=True
                    )

                gs_hr()

                bind_number(
                    sid, "battery_kwh",
                    "Average battery capacity (kWh)" if lang == "EN" else "Durchschnittliche Batteriekapazität (kWh)",
                    min_value=1.0, max_value=2000.0, step=10.0,
                )
                bind_number(
                    sid, "ev_consumption",
                    "EV consumption (kWh per km)" if lang == "EN" else "EV-Verbrauch (kWh pro km)",
                    min_value=0.5, max_value=3.0, step=0.05,
                    help=("Typical range: 0.8–1.2 kWh/km for many use cases." if lang == "EN"
                          else "Typischer Bereich: 0,8–1,2 kWh/km für viele Anwendungen.")
                )
                bind_number(
                    sid, "km_per_truck_per_day",
                    "Average km driven per truck per day" if lang == "EN" else "Ø km pro LKW und Tag",
                    min_value=0.0, max_value=2000.0, step=10.0,
                    help=("This replaces SoC inputs. We derive the SoC drop from km/day, consumption, and battery size." if lang == "EN"
                          else "Ersetzt SoC-Inputs. Wir leiten den SoC-Abfall aus km/Tag, Verbrauch und Batterie ab.")
                )

                # Show derived SoC / energy clearly to the customer
                d = derive_soc_from_km(st.session_state["inputs"])
                st.markdown(
                    ("**Derived from your km/day:**" if lang == "EN" else "**Aus km/Tag abgeleitet:**")
                )
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Energy / truck / day" if lang == "EN" else "Energie / LKW / Tag",
                              f"{d['daily_energy_kwh']:,.0f} kWh")
                with c2:
                    st.metric("Energy / event" if lang == "EN" else "Energie / Event",
                              f"{d['energy_per_event_kwh']:,.0f} kWh")
                with c3:
                    st.metric("SoC drop / event" if lang == "EN" else "SoC-Abfall / Event",
                              f"{d['soc_diff']*100:,.0f}%")

                if d.get("warning"):
                    st.warning(d["warning"])

            elif sid == "electricity":
                # Price mode (fixed vs dynamic) — replaces dynamic_share input
                bind_select(
                    sid, "price_mode",
                    "Electricity price type" if lang == "EN" else "Strompreis-Typ",
                    options=["fixed", "dynamic"],
                    format_func=lambda v: ("Fixed contract" if lang == "EN" else "Fixpreis-Vertrag") if v == "fixed"
                    else ("Dynamic (spot price)" if lang == "EN" else "Dynamisch (Spotpreis)"),
                    help=("Dynamic uses historic spot prices from the uploaded profile (column with prices)."
                          if lang == "EN" else
                          "Dynamisch nutzt historische Spotpreise aus dem hochgeladenen Profil (Preisspalte).")
                )

                if get_inp("price_mode") == "fixed":
                    bind_number(
                        sid, "fixed_elec_price_mwh",
                        "Fixed electricity price (€/MWh)" if lang == "EN" else "Fixer Strompreis (€/MWh)",
                        min_value=0.0, max_value=2000.0, step=10.0,
                        help=("Standard: Germany average 86.79 €/MWh" if lang == "EN" else "Standard: Deutscher Durchschnitt 86,79 €/MWh")
                    )
                else:
                    bind_select(
                        sid, "dynamic_price_region",
                        "Dynamic price region" if lang == "EN" else "Dynamic-Preis Region",
                        options=["AT", "DE", "EU"],
                        help=("Uses a fixed national average (€/MWh) + TOU weighting in the charging window." if lang == "EN"
                            else "Nutzt einen fixen Länder-Durchschnitt (€/MWh) + TOU-Gewichtung im Ladefenster.")
                    )
                    r = str(get_inp("dynamic_price_region")).upper()
                    v = DYNAMIC_AVG_PRICE_EUR_PER_MWH.get(r, DYNAMIC_AVG_PRICE_EUR_PER_MWH["DE"])
                    st.metric("Applied average (€/MWh)" if lang == "EN" else "Verwendeter Durchschnitt (€/MWh)", f"{v:,.2f}")


                gs_hr()

                c1, c2 = st.columns(2)
                with c1:
                    bind_slider(
                        sid, "start_hour",
                        "Charging start hour" if lang == "EN" else "Ladebeginn (Stunde)",
                        0, 23, 1,
                    )
                with c2:
                    bind_slider(
                        sid, "end_hour",
                        "Charging end hour" if lang == "EN" else "Ladeende (Stunde)",
                        0, 23, 1,
                    )



            elif sid == "site":
                # ---- Upload section ----
                st.markdown("**Lastprofil Upload (CSV)**" if lang == "DE" else "**Load profile upload (CSV)**")
                st.caption(
                    ("Only the hours inside your charging window matter. We'll look at the peak site load in that window "
                     "to compute how much kW are still available for truck charging.")
                    if lang == "EN" else
                    ("Wichtig: Für die Auswertung zählt nur dein Ladefenster. Wir nehmen den Peak der Standortlast im Ladefenster "
                     "und berechnen daraus, wie viel kW noch für das Laden übrig sind.")
                )

                upload = st.file_uploader(
                    "Upload CSV (timestamp + site consumption kW, optional price column)" if lang == "EN"
                    else "CSV hochladen (Timestamp + Standortverbrauch kW, optional Preisspalte)",
                    type=["csv"],
                    key="load_profile_uploader",
                    help=("Example columns: timestamp, consumption_kW, da_price" if lang == "EN"
                          else "Beispiel-Spalten: timestamp, consumption_kW, da_price")
                )

                if upload is not None:
                    raw = upload.getvalue()
                    digest = hashlib.md5(raw).hexdigest()
                    cache = st.session_state["profile_cache"]

                    if cache.get("digest") != digest:
                        df_raw = parse_uploaded_csv(upload)
                        cache.update({"digest": digest, "df": df_raw, "metrics": {}, "meta": {}})
                        # Try auto-detect columns
                        if df_raw is not None and not df_raw.empty:
                            cache["timestamp_col"] = _guess_timestamp_col(df_raw)
                            cache["consumption_col"] = _guess_consumption_col(df_raw)

                    df_raw = cache.get("df")
                    if df_raw is None or df_raw.empty:
                        st.error("CSV konnte nicht gelesen werden." if lang == "DE" else "Could not read CSV.")
                    else:
                        # Column selectors (customer-friendly, minimal)
                        cols = list(df_raw.columns)

                        cA, cB, cC = st.columns(3)
                        with cA:
                            ts_col = st.selectbox(
                                "Timestamp column" if lang == "EN" else "Timestamp-Spalte",
                                options=cols,
                                index=cols.index(cache["timestamp_col"]) if cache.get("timestamp_col") in cols else 0,
                                key="ts_col_select",
                            )
                        with cB:
                            cons_col = st.selectbox(
                                "Consumption column (kW)" if lang == "EN" else "Verbrauchsspalte (kW)",
                                options=cols,
                                index=cols.index(cache["consumption_col"]) if cache.get("consumption_col") in cols else 0,
                                key="cons_col_select",
                            )

                        cache["timestamp_col"] = ts_col
                        cache["consumption_col"] = cons_col

                        # Compute metrics using current charging window
                        start_h = int(get_inp("start_hour")) % 24
                        end_h = int(get_inp("end_hour")) % 24
                        try:
                            metrics = compute_profile_metrics(df_raw, ts_col, cons_col, start_h, end_h)
                            cache["metrics"] = metrics
                        except Exception:
                            cache["metrics"] = {}
                            st.warning("Profil konnte nicht vollständig ausgewertet werden." if lang == "DE"
                                       else "Could not fully analyse profile.")

                        # Preview
                        with st.expander("Preview (first rows)", expanded=False):
                            st.dataframe(df_raw.head(60), use_container_width=True)

                        # Show window-peak and headroom instantly
                        m = cache.get("metrics") or {}
                        if m.get("peak_kw_in_window") is not None:
                            cap_kva = float(get_inp("site_capacity_kva"))
                            avail = max(0.0, cap_kva - float(m["peak_kw_in_window"]))
                            st.markdown("**Kapazität im Ladefenster**" if lang == "DE" else "**Capacity in charging window**")
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                st.metric("Peak Last im Fenster" if lang == "DE" else "Peak load in window",
                                          f"{float(m['peak_kw_in_window']):,.0f} kW")
                            with c2:
                                st.metric("Standort-Limit" if lang == "DE" else "Site limit",
                                          f"{cap_kva:,.0f} kW")
                            with c3:
                                st.metric("Headroom (Peak)" if lang == "DE" else "Headroom (peak)",
                                          f"{avail:,.0f} kW")
                        else:
                            st.info("Keine Peak-Auswertung möglich (Spalten prüfen)." if lang == "DE"
                                    else "No peak evaluation possible (check columns).")

                        # Trigger recalc with new profile
                        st.session_state["profile_cache"] = cache
                        recalc_from_inputs()
                else:
                    st.session_state["profile_cache"]["digest"] = None
                    st.session_state["profile_cache"]["df"] = None
                    st.session_state["profile_cache"]["metrics"] = {}

                gs_hr()

                # ---- Manual inputs still exist as fallback ----
                bind_number(
                    sid, "site_capacity_kva",
                    "Site capacity limit (kW/kVA)" if lang == "EN" else "Standort-Limit (kW/kVA)",
                    min_value=0.0, max_value=100000.0, step=10.0,
                    help=("Total site connection/limit. We'll compare this to the peak site load inside the charging window."
                          if lang == "EN" else
                          "Gesamtes Standort-Limit. Wir vergleichen das mit dem Peak im Ladefenster.")
                )
                bind_number(
                    sid, "charger_power_kw",
                    "Max charger power per truck (kW)" if lang == "EN" else "Max. Ladeleistung pro LKW (kW)",
                    min_value=0.0, max_value=2000.0, step=10.0,
                    help=("This is the maximum per-truck power. Smart charging can use a lower per-truck limit."
                          if lang == "EN" else
                          "Das ist die Maximalleistung pro LKW. Smart Charging kann eine geringere Leistung nutzen.")
                )
                bind_number(
                    sid, "existing_peak_kw",
                    "Manual site peak (kW) — used only if no profile is uploaded" if lang == "EN" else "Manueller Standort-Peak (kW) — nur ohne Profil",
                    min_value=0.0, max_value=100000.0, step=10.0,
                )

                # Reverse calculation summary (from results)
                recalc_from_inputs()
                res_now = st.session_state.get("model_results", {})
                cap = res_now.get("capacity_analysis", {}) if res_now else {}
                if cap:
                    gs_hr()
                    st.markdown("### Reverse calculation (how many trucks can you charge?)" if lang == "EN"
                                else "### Reverse Calculation (wie viele LKW kannst du laden?)")
                    st.caption(
                        ("We first take your **site headroom** (site limit minus peak load in your charging window). "
                         "Then we translate that into trucks using two perspectives: energy-based vs simultaneous-at-full-power.")
                        if lang == "EN" else
                        ("Wir nehmen zuerst deinen **Headroom** (Standort-Limit minus Peak-Last im Ladefenster). "
                         "Dann übersetzen wir das in LKW — einmal energie-basiert, einmal „alle gleichzeitig Vollgas“.")
                    )
                    a, b, c, d = st.columns(4)
                    with a:
                        st.metric("Headroom (kW)", f"{float(cap.get('available_kw_at_peak',0)):,.0f}")
                    with b:
                        st.metric("Max trucks (energy-based)", f"{int(cap.get('max_trucks_energy_based',0))}")
                    with c:
                        st.metric("Max trucks (simultaneous @ max kW)", f"{int(cap.get('max_trucks_simultaneous_at_full_power',0))}")
                    with d:
                        v = cap.get("recommended_avg_kw_per_truck")
                        st.metric("Recommended avg kW / truck", f"{float(v):,.0f} kW" if v else "—")

                    st.markdown(
                        ("**Why lowering per-truck power can help:** If your chargers are very powerful, only a few trucks fit *simultaneously* under the site limit. "
                         "With smart charging you cap per-truck kW and spread charging across the whole time window.")
                        if lang == "EN" else
                        ("**Warum geringere Ladeleistung helfen kann:** Wenn die Lader extrem stark sind, passen nur wenige LKW *gleichzeitig* unter das Standort-Limit. "
                         "Mit Smart Charging begrenzt du kW pro LKW und verteilst das Laden über das ganze Zeitfenster.")
                    )

                with st.expander(
                    "Optional: Peak limit (energy management / battery)" if lang == "EN" else "Optional: Peak-Limit (Energiemanagement / Batterie)",
                    expanded=False
                ):
                    bind_number(
                        sid, "desired_peak_limit_kw",
                        "Desired peak limit (kW) — 0 = off" if lang == "EN" else "Ziel-Peak-Limit (kW) — 0 = aus",
                        min_value=0.0, max_value=200000.0, step=50.0
                    )
                    bind_number(
                        sid, "peak_duration_h",
                        "Peak duration (hours)" if lang == "EN" else "Peak-Dauer (Stunden)",
                        min_value=0.0, max_value=24.0, step=0.05
                    )

            elif sid == "diesel":
                # Defaults: keep simple but update toll for AT based on representative ASFINAG rate
                PRESETS = {
                    "AT": {"diesel_price": 1.75, "toll_rate": 0.456, "tolled_share": 0.60},
                    "DE": {"diesel_price": 1.70, "toll_rate": 0.21, "tolled_share": 0.55},
                    "EU": {"diesel_price": 1.75, "toll_rate": 0.23, "tolled_share": 0.55},
                }

                bind_select(sid, "market_region", "Region" if lang == "EN" else "Region", options=["AT", "DE", "EU"])

                if st.button("Apply estimate" if lang == "EN" else "Schätzung übernehmen", use_container_width=True, key="apply_market_estimate"):
                    r = get_inp("market_region")
                    p = PRESETS.get(r, PRESETS["EU"])
                    set_inp("diesel_price", float(p["diesel_price"]))
                    set_inp("toll_rate", float(p["toll_rate"]))
                    set_inp("tolled_share", float(p["tolled_share"]))
                    st.session_state[wkey("diesel", "diesel_price")] = get_inp("diesel_price")
                    st.session_state[wkey("diesel", "toll_rate")] = get_inp("toll_rate")
                    st.session_state[wkey("diesel", "tolled_share")] = get_inp("tolled_share")
                    st.rerun()

                gs_hr()

                bind_number(
                    sid, "diesel_price",
                    "Diesel price (€/L)" if lang == "EN" else "Dieselpreis (€/L)",
                    min_value=0.0, max_value=5.0, step=0.05,
                )
                bind_number(
                    sid, "diesel_l_per_100",
                    "Diesel consumption (L/100km)" if lang == "EN" else "Dieselverbrauch (L/100km)",
                    min_value=0.0, max_value=200.0, step=1.0,
                )

                st.markdown("**Toll assumptions**" if lang == "EN" else "**Maut-Annahmen**")
                bind_number(
                    sid, "toll_rate",
                    "Toll rate (€/km)" if lang == "EN" else "Mautsatz (€/km)",
                    min_value=0.0, max_value=5.0, step=0.001,
                    help=("For AT, this default is a simplified representative ASFINAG GO-toll for 4+ axles EURO VI-ish."
                          if lang == "EN" else
                          "Für AT ist der Default eine vereinfachte, repräsentative ASFINAG GO-Maut (4+ Achsen, EURO VI-ähnlich).")
                )
                bind_slider(
                    sid, "tolled_share",
                    "Share of distance that is tolled" if lang == "EN" else "Anteil mautpflichtiger Strecke",
                    0.0, 1.0, 0.05,
                )
                bind_checkbox(
                    sid, "ev_toll_exempt",
                    "Assume EV toll exempt" if lang == "EN" else "EV mautbefreit annehmen"
                )

        recalc_from_inputs()

        nav1, nav2, nav3 = st.columns([0.22, 0.22, 0.56])
        with nav1:
            if st.button("← " + t("back"), use_container_width=True, disabled=(i == 0), key=f"nav_back_{i}"):
                st.session_state["wizard_step"] = max(0, i - 1)
                st.rerun()
        with nav2:
            if st.button(t("next") + " →", use_container_width=True, disabled=(i == len(STEPS) - 1), key=f"nav_next_{i}"):
                st.session_state["wizard_step"] = min(len(STEPS) - 1, i + 1)
                st.rerun()
        with nav3:
            if st.button("✨ " + t("finish"), use_container_width=True, disabled=(i != len(STEPS) - 1), key="nav_finish"):
                st.session_state["flow"] = "report"
                st.session_state["pending_flow_switch"] = "report"
                st.session_state["report_md"] = None
                st.session_state["report_pdf_bytes"] = None
                st.rerun()

    # ============ REPORT ============
    else:
        st.markdown("## " + t("report"))
        res = st.session_state.get("model_results")
        if not res:
            st.info("—")
        else:
            tabs = st.tabs([
                "Overview" if st.session_state["lang"] == "EN" else "Übersicht",
                "Finance" if st.session_state["lang"] == "EN" else "Finanzen",
                "Charts",
                "Calculations" if st.session_state["lang"] == "EN" else "Berechnungen",
                "What-if" if st.session_state["lang"] == "EN" else "Sensitivität",
                "Export"
            ])

            with tabs[0]:
                issues = detect_issues(res)
                solutions = generate_solution_set(res, issues)

                ec = res["energy_cost"]
                dv = res["diesel_vs_ev"]
                load = res["load"]
                cap = res.get("capacity_analysis", {})
                prof = res.get("profile", {}).get("derived_soc", {})

                a, b, c = st.columns(3)
                with a: st.metric(t("kpi_savings"), fmt_eur(dv.get("total_savings_incl_toll_eur")))
                with b: st.metric("EV cost / year" if st.session_state["lang"] == "EN" else "EV-Kosten / Jahr", fmt_eur(ec.get("annual_cost_eur")))
                with c: st.metric(t("kpi_peak"), f"{fmt_num(load.get('new_theoretical_peak_kw'))} kW")

                # Capacity window KPIs (new)
                if cap:
                    k1, k2, k3, k4 = st.columns(4)
                    with k1:
                        st.metric("Peak im Ladefenster" if st.session_state["lang"] == "DE" else "Peak in window",
                                  f"{float(cap.get('site_peak_kw_in_window',0.0)):,.0f} kW")
                    with k2:
                        st.metric("Headroom (Peak)" if st.session_state["lang"] == "DE" else "Headroom (peak)",
                                  f"{float(cap.get('available_kw_at_peak',0.0)):,.0f} kW")
                    with k3:
                        st.metric("Max Trucks (energy)" if st.session_state["lang"] == "DE" else "Max trucks (energy)",
                                  f"{int(cap.get('max_trucks_energy_based',0))}")
                    with k4:
                        v = cap.get("recommended_avg_kw_per_truck")
                        st.metric("Empf. Ø kW/LKW" if st.session_state["lang"] == "DE" else "Rec. avg kW/truck",
                                  f"{float(v):,.0f} kW" if v else "—")

                gs_hr()

                with st.container(border=True):
                    st.markdown("<span class='gs-card-marker'></span>", unsafe_allow_html=True)

                    st.markdown("## Analyse & Empfehlung")
                    st.markdown(report_analysis_markdown(res))

                    st.markdown("### Constraints & Empfehlungen")
                    constraints = report_constraints(res)
                    if not constraints:
                        st.success("Keine großen Constraints erkannt.")
                    else:
                        for lvl, txt in constraints:
                            if lvl == "ok":
                                st.success(txt)
                            elif lvl == "bad":
                                st.error(txt)
                            else:
                                st.warning(txt)

                    st.markdown("### Nächste sinnvolle Schritte")
                    for s in report_next_steps(res):
                        st.markdown(f"- {s}")


                gs_hr()

                with st.container(border=True):
                    st.markdown("<span class='gs-card-marker'></span>", unsafe_allow_html=True)

                    st.markdown("### " + t("constraints"))
                    if not issues:
                        st.success("No major constraints detected." if st.session_state["lang"] == "EN" else "Keine großen Constraints erkannt.")
                    else:
                        for it in issues:
                            sev = it.get("severity", "low")
                            msg = it.get("description", "")
                            if sev == "high":
                                st.error(msg)
                            elif sev == "medium":
                                st.warning(msg)
                            else:
                                st.info(msg)

                    st.markdown("#### " + ("Best next actions" if st.session_state["lang"] == "EN" else "Nächste sinnvolle Schritte"))
                    for idx, s in enumerate(solutions, 1):
                        st.markdown(f"**{idx}. {s.get('title','')}** — {s.get('definition','')}")

                gs_hr()

                lang = st.session_state.get("lang", "DE")

                # --- Reverse Calculation (robust, ohne Duplikate, KPI-Style bleibt) ---
                if cap:
                    with st.container(border=True):
                        st.markdown("<span class='gs-card-marker'></span>", unsafe_allow_html=True)

                        st.markdown(
                            "### " + (
                                "Reverse Calculation (how many trucks fit in the window?)"
                                if lang == "EN" else
                                "Reverse Calculation (wie viele LKW gehen im Ladefenster?)"
                            )
                        )

                        c1, c2, c3, c4 = st.columns(4)

                        with c1:
                            st.metric(
                                "Max trucks (energy)" if lang == "EN" else "Max LKW (Energie)",
                                f"{int(cap.get('max_trucks_energy_based', 0))}"
                            )

                        with c2:
                            st.metric(
                                "Max trucks (simultaneous)" if lang == "EN" else "Max LKW (simultan)",
                                f"{int(cap.get('max_trucks_simultaneous_at_full_power', 0))}"
                            )

                        with c3:
                            st.metric(
                                "Energy available (window)" if lang == "EN" else "Energie verfügbar (Fenster)",
                                f"{float(cap.get('energy_available_kwh_in_window', 0.0)):,.0f} kWh"
                            )

                        with c4:
                            st.metric(
                                "Energy / truck / day" if lang == "EN" else "Energie / LKW / Tag",
                                f"{float(cap.get('energy_needed_kwh_per_truck_day', 0.0)):,.0f} kWh"
                            )

                        with st.expander("Interpretation" if lang == "DE" else "Interpretation", expanded=False):
                            st.markdown(
                                (
                                    "- **Energy-based**: nutzt die verfügbare Energie im gesamten Ladefenster (realistisch mit Smart Charging).\n"
                                    "- **Simultan**: Worst-Case, alle ziehen gleichzeitig volle Leistung.\n"
                                    f"- Ladefenster: **{int(cap.get('charging_window_hours_total', 0))} h**\n"
                                    f"- Peak im Ladefenster: **{float(cap.get('site_peak_kw_in_window', 0.0)):,.0f} kW**\n"
                                    f"- Headroom (Peak): **{float(cap.get('available_kw_at_peak', 0.0)):,.0f} kW**\n"
                                ) if lang == "DE" else (
                                    "- **Energy-based**: uses total energy available over the charging window (realistic with smart charging).\n"
                                    "- **Simultaneous**: worst case, all trucks draw full power at the same time.\n"
                                    f"- Charging window: **{int(cap.get('charging_window_hours_total', 0))} h**\n"
                                    f"- Peak in window: **{float(cap.get('site_peak_kw_in_window', 0.0)):,.0f} kW**\n"
                                    f"- Headroom (peak): **{float(cap.get('available_kw_at_peak', 0.0)):,.0f} kW**\n"
                                )
                            )
                else:
                    st.info("Reverse-Calc nicht verfügbar (capacity_analysis fehlt)." if lang == "DE"
                            else "Reverse calc not available (capacity_analysis missing).")
                    



            with tabs[1]:
                # Finance charts (bar charts) — ordered to match the narrative (diesel vs EV, then toll)
                f1, f2 = fig_finance_bars(res)
                st.pyplot(f1, use_container_width=True)
                dv = res["diesel_vs_ev"]
                ec = res["energy_cost"]

                chart_note(
                    lines=[
                        "Vergleicht jährliche **Diesel-Baseline** vs. **EV-Stromkosten** (nur Energie, ohne CAPEX).",
                        "Die Differenz ist der Haupttreiber für die Einsparung."
                    ],
                    metrics={
                        "Diesel (€/Jahr)": fmt_eur(dv.get("diesel_cost_baseline_eur")),
                        "EV Strom (€/Jahr)": fmt_eur(ec.get("annual_cost_eur")),
                        "Δ (ohne Maut)": fmt_eur(dv.get("cost_savings_eur")),
                    },
                )

                st.pyplot(f2, use_container_width=True)
                chart_note(
                    lines=[
                        "Zeigt den **Maut-Effekt** (Diesel vs. EV).",
                        "Wenn EV mautbefreit angenommen wird, ist EV-Maut 0 €."
                    ],
                    metrics={
                        "Maut Diesel (€/Jahr)": fmt_eur(dv.get("baseline_toll_cost_eur")),
                        "Maut-Ersparnis (€/Jahr)": fmt_eur(dv.get("toll_savings_eur")),
                        "Netto inkl. Maut": fmt_eur(dv.get("total_savings_incl_toll_eur")),
                    },
                )


                dv = res["diesel_vs_ev"]
                st.markdown(
                    ("- First chart: compares **diesel baseline** vs **EV electricity**.\n"
                     "- Second chart: shows the **toll effect** (diesel vs EV).\n"
                     f"- Net savings incl. toll: **{float(dv.get('total_savings_incl_toll_eur',0.0)):,.0f} €/year**.\n")
                    if st.session_state["lang"] == "EN" else
                    ("- 1. Chart: **Diesel-Baseline** vs **EV-Stromkosten**.\n"
                     "- 2. Chart: **Maut-Effekt** (Diesel vs EV).\n"
                     f"- Netto-Einsparung inkl. Maut: **{float(dv.get('total_savings_incl_toll_eur',0.0)):,.0f} €/Jahr**.\n")
                )

            with tabs[2]:
                dfh = build_hourly_df(res)

                gs_hr()

                # Chart 1 (price + CO2)
                label_price = "Spot price (€/kWh)" if (res.get("profile", {}).get("has_price") and get_inp("price_mode") == "dynamic") else "Price (€/kWh)"
                f1 = fig_price_co2(dfh, label_price)
                st.pyplot(f1, use_container_width=True)
                pdets = res["energy_cost"].get("price_details", {}) or {}
                cdets = res["co2"].get("co2_details", {}) or {}

                chart_note(
                    lines=[
                        "Preis ist **stundenbasiert** und im Ladefenster gewichtet.",
                        "CO₂ ist ebenfalls **stundenbasiert** (Grid-Intensität über den Tag).",
                        f"Datenquelle Preis: {'Spot (Upload)' if pdets.get('used_spot_data') else 'TOU-Fallback/konstant'}."
                    ],
                    metrics={
                        "Effektiv (€/MWh)": fmt_eur_per_mwh(res["energy_cost"].get("effective_price_eur_per_mwh")),
                        "Ø Fenster (€/kWh)": fmt_eur_per_kwh(pdets.get("window_avg_eur_per_kwh")),
                        "Rel-Faktor": f"{float(pdets.get('rel_factor', 1.0)):.2f}×" if pdets.get("rel_factor") is not None else "—",
                        "CO₂ Fenster (g/kWh)": f"{float(cdets.get('grid_window_avg_g_per_kwh', 0.0)):,.0f}",
                    },
                )


                gs_hr()

                # Chart 2 (load profile)
                f2 = fig_load(dfh, res["inputs"]["site_capacity_limit_kva"], res["load"]["new_theoretical_peak_kw"])
                st.pyplot(f2, use_container_width=True)
                cap = res.get("capacity_analysis", {}) or {}
                load = res.get("load", {}) or {}

                chart_note(
                    lines=[
                        "Baseline ist entweder **Upload (stündliche Maxima)** oder **manueller Peak**.",
                        "„Spread in window“ verteilt die Ladeenergie gleichmäßig über das Ladefenster (Smart-Charging-ähnlich).",
                        "Die gestrichelte Linie ist dein **Standortlimit**; der Punkt-Strich ist der **Worst-Case Peak** (alle Lader voll)."
                    ],
                    metrics={
                        "Standortlimit": f"{float(cap.get('site_capacity_kw', 0.0)):,.0f} kW",
                        "Peak im Fenster": f"{float(cap.get('site_peak_kw_in_window', 0.0)):,.0f} kW",
                        "Headroom": f"{float(cap.get('available_kw_at_peak', 0.0)):,.0f} kW",
                        "Worst-Case Peak": f"{float(load.get('new_theoretical_peak_kw', 0.0)):,.0f} kW",
                    },
                )


                gs_hr()

                # Chart 3 (waterfall)
                f3 = fig_waterfall(res)
                st.pyplot(f3, use_container_width=True)
                dv = res["diesel_vs_ev"]
                chart_note(
                    lines=[
                        "Zerlegt die jährliche Ersparnis: Diesel-Kosten minus EV-Stromkosten plus Maut-Effekt.",
                        "Die letzten Balken sind die **Netto-Ersparnis**."
                    ],
                    metrics={
                        "Diesel (€/Jahr)": fmt_eur(dv.get("diesel_cost_baseline_eur")),
                        "EV Strom (€/Jahr)": fmt_eur(dv.get("ev_cost_eur")),
                        "Maut Δ (€/Jahr)": fmt_eur(dv.get("toll_savings_eur")),
                        "Netto (€/Jahr)": fmt_eur(dv.get("total_savings_incl_toll_eur")),
                    },
                )


                gs_hr()

                # Chart 4 (pareto)
                f4 = fig_pareto_shifted_windows(res)
                st.pyplot(f4, use_container_width=True)
                pdf = st.session_state.get("pareto_df")
                win_len = st.session_state.get("pareto_win_len")

                if isinstance(pdf, pd.DataFrame) and not pdf.empty:
                    best_cost = pdf.loc[pdf["cost"].idxmin()]
                    best_co2 = pdf.loc[pdf["co2"].idxmin()]
                    cur_s = int(res["inputs"].get("start_hour", 0)) % 24

                    chart_note(
                        lines=[
                            f"Es wird **dieselbe Fensterlänge** ({int(win_len)} h) über den Tag verschoben.",
                            "Jeder Punkt ist ein Startzeitpunkt (Label im Plot = Startstunde).",
                            "Links unten = billig & CO₂-arm (Trade-off möglich)."
                        ],
                        metrics={
                            "Günstigster Start": f"{int(best_cost['start']):02d}h (≈ {fmt_eur(best_cost['cost'])}/Jahr)",
                            "CO₂-min Start": f"{int(best_co2['start']):02d}h (≈ {fmt_kg(best_co2['co2'])}/Jahr)",
                            "Aktuell Start": f"{cur_s:02d}h",
                        },
                    )
                else:
                    chart_note(["Pareto-Tabelle nicht verfügbar (siehe Fehlerhinweis im Plot)."])


            with tabs[3]:
                df_calc = build_calculation_df(res)
                st.dataframe(df_calc, use_container_width=True, hide_index=True)

                with st.expander("Full model output (JSON)", expanded=False):
                    st.json(res)

            with tabs[4]:
                st.subheader("Tornado / Sensitivitätsanalyse (ceteris paribus)")

                # ---------------------------------------------------------
                # 1) Results holen (gekoppelt an session_state["model_results"])
                # ---------------------------------------------------------
                res2 = st.session_state.get("model_results")
                if not res2:
                    st.info("Noch keine Model-Results vorhanden. Bitte zuerst den Fragebogen ausfüllen.")
                    st.stop()

                base = (res2.get("inputs") or {}).copy()
                if not base:
                    st.error("`model_results['inputs']` fehlt oder ist leer. Tornado braucht die run_model()-Parameter.")
                    st.stop()

                # Basis-Savings
                try:
                    base_savings = float(res2["diesel_vs_ev"]["total_savings_incl_toll_eur"])
                except Exception:
                    st.error("Basis-Ersparnis fehlt: `diesel_vs_ev.total_savings_incl_toll_eur`.")
                    st.stop()

                # Spot-Infos (falls jemals aktiv) konsistent mitgeben
                pdets = (res2.get("energy_cost", {}) or {}).get("price_details", {}) or {}
                used_spot = bool(pdets.get("used_spot_data", False))
                spot_curve_avg = pdets.get("curve_avg_eur_per_kwh", None) if used_spot else None
                spot_window_avg = pdets.get("window_avg_eur_per_kwh", None) if used_spot else None

                # ---------------------------------------------------------
                # 2) Parameter-Setup
                # ---------------------------------------------------------
                # Default-Fraktionen: bewusst pragmatisch
                params = [
                    ("Electricity price (€/MWh)", "avg_elec_price_eur_per_mwh", 0.20),
                    ("Diesel price (€/L)", "diesel_price_eur_per_l", 0.15),
                    ("EV consumption (kWh/km)", "ev_consumption_kwh_per_km", 0.15),
                    ("Diesel L/100km", "diesel_l_per_100km", 0.15),
                    ("Events / truck / day", "events_per_truck_per_day", 0.20),
                    ("Tolled share", "tolled_share_0_1", 0.20),
                    ("Toll rate (€/km)", "toll_rate_eur_per_km", 0.20),
                ]

                missing = [k for _, k, _ in params if k not in base]
                if missing:
                    st.error(
                        "Tornado kann nicht laufen, weil diese Keys in `model_results['inputs']` fehlen:\n"
                        + ", ".join(missing)
                        + "\n\nAchtung: Tornado erwartet run_model()-Keys, nicht UI-Keys."
                    )
                    st.stop()

                # ---------------------------------------------------------
                # 3) Helpers (Clamp + sichere Defaults)
                # ---------------------------------------------------------
                ABS_FLOOR = {
                    "diesel_price_eur_per_l": 0.05,
                    "avg_elec_price_eur_per_mwh": 5.0,
                    "ev_consumption_kwh_per_km": 0.02,
                    "diesel_l_per_100km": 0.5,
                    "events_per_truck_per_day": 0.25,
                    "tolled_share_0_1": 0.01,
                    "toll_rate_eur_per_km": 0.01,
                }

                def _clamp_value(key: str, v):
                    # numerisch erzwingen
                    try:
                        v = float(v)
                    except Exception:
                        v = float(ABS_FLOOR.get(key, 0.0))

                    # Clamp je nach Typ
                    if key == "tolled_share_0_1":
                        v = min(max(v, 0.0), 1.0)
                    elif key == "events_per_truck_per_day":
                        v = max(v, 0.0)
                    else:
                        v = max(v, 0.0)

                    # Floors
                    if v <= 0 and key in ABS_FLOOR:
                        v = float(ABS_FLOOR[key])
                    return v

                def _run_savings(overrides: dict) -> float:
                    kw = base.copy()
                    kw.update(overrides)

                    r = run_model(
                        **kw,
                        spot_curve_avg_eur_per_kwh=spot_curve_avg,
                        spot_window_avg_eur_per_kwh=spot_window_avg,
                    )
                    return float(r["diesel_vs_ev"]["total_savings_incl_toll_eur"])

                # ---------------------------------------------------------
                # 4) Cache (kein Button) – nur neu rechnen wenn Inputs sich ändern
                # ---------------------------------------------------------
                digest = _inputs_digest_for_tornado(st.session_state["inputs"])
                cache_key = "tornado_cache"

                if (cache_key in st.session_state) and (st.session_state[cache_key].get("digest") == digest):
                    df_tornado = st.session_state[cache_key]["df"]
                    base_savings_cached = st.session_state[cache_key]["base_savings"]
                else:
                    rows = []
                    for label, key, frac in params:
                        base_val = _clamp_value(key, base.get(key, 0.0))

                        low = _clamp_value(key, base_val * (1.0 - frac))
                        high = _clamp_value(key, base_val * (1.0 + frac))

                        # Wenn base_val ≈ 0: nutze Floor als "base" und skaliere davon
                        if float(base.get(key, 0.0)) == 0.0 and key in ABS_FLOOR:
                            base_val = float(ABS_FLOOR[key])
                            low = _clamp_value(key, base_val * (1.0 - frac))
                            high = _clamp_value(key, base_val * (1.0 + frac))

                        try:
                            s_low = _run_savings({key: low})
                            s_high = _run_savings({key: high})
                        except Exception as e:
                            s_low = np.nan
                            s_high = np.nan

                        d_low = s_low - base_savings if np.isfinite(s_low) else np.nan
                        d_high = s_high - base_savings if np.isfinite(s_high) else np.nan

                        # Range = Spannweite, robust gegen Reihenfolge
                        if np.isfinite(d_low) and np.isfinite(d_high):
                            rng = float(abs(d_high - d_low))
                            lo = float(min(d_low, d_high))
                            hi = float(max(d_low, d_high))
                        else:
                            rng, lo, hi = np.nan, np.nan, np.nan

                        rows.append({
                            "Parameter": label,
                            "Key": key,
                            "Base": base_val,
                            "Low": low,
                            "High": high,
                            "Savings_base": base_savings,
                            "Savings_low": s_low,
                            "Savings_high": s_high,
                            "Delta_low": d_low,
                            "Delta_high": d_high,
                            "Lo": lo,
                            "Hi": hi,
                            "Range": rng,
                        })

                    df_tornado = pd.DataFrame(rows)
                    st.session_state[cache_key] = {"digest": digest, "df": df_tornado, "base_savings": base_savings}
                    base_savings_cached = base_savings

                # ---------------------------------------------------------
                # 5) Output: KPIs + Chart + Tabelle + Takeaways
                # ---------------------------------------------------------
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Basis-Ersparnis (inkl. Maut)" if st.session_state["lang"] == "DE" else "Base savings (incl. toll)", fmt_eur(base_savings_cached))
                with c2:
                    st.metric("Preis-Modus" if st.session_state["lang"] == "DE" else "Price mode",
                            "Spot (Upload)" if used_spot else ("Dynamic (TOU)" if float(base.get("dynamic_price_share", 0.0)) >= 0.5 else "Fixed"))
                with c3:
                    st.metric("Parameter getestet" if st.session_state["lang"] == "DE" else "Params tested", f"{len(df_tornado)}")

                # Sortiert nach Einfluss
                dfp = df_tornado.copy()
                dfp = dfp.replace([np.inf, -np.inf], np.nan)
                dfp = dfp.loc[dfp["Range"].notna()].sort_values("Range", ascending=False)

                if dfp.empty:
                    st.warning("Tornado nicht verfügbar (keine gültigen Läufe).")
                    st.dataframe(df_tornado, use_container_width=True, hide_index=True)
                    st.stop()

                # Tornado Plot (ohne Farben festzunageln)
                fig, ax = plt.subplots(figsize=(9, 5))
                y = np.arange(len(dfp))
                widths = (dfp["Hi"] - dfp["Lo"]).to_numpy()
                lefts = dfp["Lo"].to_numpy()

                ax.barh(y, widths, left=lefts)
                ax.axvline(0.0, linestyle="--", alpha=0.4)
                ax.set_yticks(y)
                ax.set_yticklabels(dfp["Parameter"].tolist())
                ax.set_xlabel("Δ Savings vs Base (€/year)" if st.session_state["lang"] == "EN" else "Δ Ersparnis vs Basis (€/Jahr)")
                ax.grid(True, axis="x", alpha=0.2)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)

                # Kurzerklärung + Top-Takeaways
                st.markdown("#### Interpretation" if st.session_state["lang"] == "DE" else "#### Interpretation")
                st.markdown(
                    tornado_takeaways(dfp, top_n=min(3, len(dfp)))
                )

                # Tabelle (kompakt)
                show_cols = ["Parameter", "Base", "Low", "High", "Savings_low", "Savings_high", "Delta_low", "Delta_high", "Range"]
                st.dataframe(dfp[show_cols], use_container_width=True, hide_index=True)

                with st.expander("Debug: run_model base inputs", expanded=False):
                    st.json(base)



            with tabs[5]:
                df_calc = build_calculation_df(res)
                dfh = build_hourly_df(res)

                st.download_button(
                    "Download results.json",
                    data=json.dumps(res, ensure_ascii=False, indent=2),
                    file_name="fleetmate_results.json",
                    mime="application/json",
                    use_container_width=True
                )
                st.download_button(
                    "Download calculations.csv",
                    data=df_calc.to_csv(index=False).encode("utf-8"),
                    file_name="fleetmate_calculations.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                charts = [
                    ("Price & grid CO₂", fig_to_png_bytes(fig_price_co2(dfh, "Price (€/kWh)"))),
                    ("Site load profile", fig_to_png_bytes(fig_load(dfh, res["inputs"]["site_capacity_limit_kva"], res["load"]["new_theoretical_peak_kw"]))),
                    ("Cost waterfall", fig_to_png_bytes(fig_waterfall(res))),
                    ("Pareto (shift window)", fig_to_png_bytes(fig_pareto_shifted_windows(res))),
                ]

                kpis = {
                    f"{t('kpi_savings')} ({t('per_year')})": fmt_eur(res["diesel_vs_ev"]["total_savings_incl_toll_eur"]),
                    f"{t('kpi_co2')} ({t('per_year')})": fmt_kg(res["diesel_vs_ev"]["co2_savings_kg"]),
                    t("kpi_peak"): f"{fmt_num(res['load']['new_theoretical_peak_kw'])} kW",
                    ("EV cost / year" if st.session_state["lang"] == "EN" else "EV-Kosten / Jahr"): fmt_eur(res["energy_cost"]["annual_cost_eur"]),
                }

                title = "FleetMate — Report"
                pdf_bytes = build_report_pdf_bytes(
                    title=title,
                    kpis=kpis,
                    narrative_md=st.session_state.get("report_md") or "",
                    issues=detect_issues(res),
                    solutions=generate_solution_set(res, detect_issues(res)),
                    lang=st.session_state.get("lang", "EN"),
                    charts=charts,
                )

                st.download_button(
                    "⬇️ Download PDF report (with charts)" if st.session_state["lang"] == "EN" else "⬇️ PDF-Report (mit Charts) downloaden",
                    data=pdf_bytes,
                    file_name="fleetmate_report_with_charts.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
