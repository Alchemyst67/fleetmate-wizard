import os
import json
import re
import requests
import streamlit as st
import io
import datetime
import hmac

from recommendations import detect_issues, generate_solution_set

def require_login():
    # Passwort aus Secrets (Cloud) oder Env
    try:
        expected = str(st.secrets.get("APP_PASSWORD", "")).strip()
    except Exception:
        expected = ""
    if not expected:
        expected = ""  # optional: fallback via os.getenv("APP_PASSWORD")

    if "authed" not in st.session_state:
        st.session_state["authed"] = False

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
        st.caption("Tipp: Passwort ist in Streamlit Secrets hinterlegt.")

    st.stop()

# =========================================================
# Core constants + model (same logic)
# =========================================================
DIESEL_CO2_PER_L = 2.64  # kg CO2 per litre

GRID_CO2_G_PER_KWH = [
    80, 78, 75, 70, 65, 60, 60, 65, 70, 75, 60, 50,
    45, 45, 50, 60, 70, 80, 90, 95, 90, 83, 78, 76
]

TOU_PRICE_EUR_PER_KWH = [
    0.20, 0.195, 0.19, 0.185, 0.18, 0.18, 0.185, 0.19,
    0.21, 0.24, 0.23, 0.22, 0.20, 0.20, 0.205, 0.21,
    0.23, 0.26, 0.30, 0.33, 0.31, 0.28, 0.24, 0.22
]


def compute_flags_and_shares(start_hour: int, end_hour: int):
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


def effective_grid_co2_kg_per_kwh(shares):
    g = sum(c * s for c, s in zip(GRID_CO2_G_PER_KWH, shares))
    return g / 1000.0


def effective_energy_price_eur_per_mwh(avg_price_eur_per_mwh: float, dynamic_share: float, shares):
    avg_price_eur_per_kwh = max(0.0, float(avg_price_eur_per_mwh)) / 1000.0
    dynamic_share = min(max(float(dynamic_share), 0.0), 1.0)

    curve_avg = sum(TOU_PRICE_EUR_PER_KWH) / 24.0
    window_avg = sum(p * s for p, s in zip(TOU_PRICE_EUR_PER_KWH, shares))
    rel = (window_avg / curve_avg) if curve_avg > 0 else 1.0

    fixed_part = avg_price_eur_per_kwh * (1.0 - dynamic_share)
    dynamic_part = avg_price_eur_per_kwh * dynamic_share * rel

    eff_eur_per_kwh = fixed_part + dynamic_part
    return eff_eur_per_kwh * 1000.0  # €/MWh


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
):
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

    # Energy
    soc_diff = max(0.0, target_soc - start_soc)
    energy_per_event_kwh = battery_kwh * soc_diff
    energy_per_event_mwh = energy_per_event_kwh / 1000.0

    total_daily_energy_mwh = num_trucks * events_per_truck_per_day * energy_per_event_mwh
    annual_energy_mwh = total_daily_energy_mwh * operating_days

    eff_price_eur_per_mwh = effective_energy_price_eur_per_mwh(
        avg_elec_price_eur_per_mwh, dynamic_price_share, shares
    )

    annual_cost_eur = annual_energy_mwh * eff_price_eur_per_mwh

    # CO2
    eff_grid_co2_kg_per_kwh = effective_grid_co2_kg_per_kwh(shares)
    annual_energy_kwh = annual_energy_mwh * 1000.0
    annual_ev_co2_kg = annual_energy_kwh * eff_grid_co2_kg_per_kwh

    # Distance (derived)
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
            "flags": compute_flags_and_shares(start_h, end_h)[0],
            "shares": shares,
            "grid_co2_g_per_kwh": GRID_CO2_G_PER_KWH,
            "tou_price_eur_per_kwh": TOU_PRICE_EUR_PER_KWH,
        },
        "energy_cost": {
            "soc_diff": soc_diff,
            "total_daily_energy_mwh": total_daily_energy_mwh,
            "annual_energy_mwh": annual_energy_mwh,
            "effective_price_eur_per_mwh": eff_price_eur_per_mwh,
            "annual_cost_eur": annual_cost_eur,
        },
        "co2": {
            "effective_grid_co2_kg_per_kwh": eff_grid_co2_kg_per_kwh,
            "annual_ev_co2_kg": annual_ev_co2_kg,
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
# Defaults + persistent input storage (FIX for "values reset")
# =========================================================
DEFAULT_LANG = "DE"

DEFAULT_INPUTS = dict(
    # Fleet
    num_trucks=10,
    operating_days=260,
    events_per_truck=1.0,

    # Battery / usage
    vehicle_profile="heavy_regional",
    battery_kwh=500.0,
    start_soc=0.20,   # empty default = 20%
    target_soc=1.00,  # target default = 100%
    ev_consumption=1.8,

    # Electricity / time
    avg_elec_price_mwh=200.0,  # €/MWh (=0.20 €/kWh)
    dynamic_share=0.30,        # 30%
    start_hour=22,             # overnight
    end_hour=8,                # overnight
    charging_window_hours=10.0,  # default = window length for 22–08

    # Site / capacity (realistic depot-ish starting values)
    existing_peak_kw=300.0,
    charger_power_kw=150.0,
    site_capacity_kva=630.0,

    # Optional peak limit
    desired_peak_limit_kw=0.0,
    peak_duration_h=0.25,

    # Diesel baseline + toll
    market_region="AT",
    diesel_price=1.75,
    diesel_l_per_100=28.0,
    toll_rate=0.25,
    tolled_share=0.60,
    ev_toll_exempt=True,
)

# Gemini may only update these (avoid letting the LLM switch profiles/regions)
ALLOWED_INPUT_KEYS = set(DEFAULT_INPUTS.keys()) - {"vehicle_profile", "market_region"}


def ensure_defaults():
    st.session_state.setdefault("lang", DEFAULT_LANG)
    st.session_state.setdefault("inputs", DEFAULT_INPUTS.copy())

    st.session_state.setdefault("flow", "wizard")  # wizard | report
    st.session_state.setdefault("wizard_step", 0)
    st.session_state.setdefault("model_results", None)
    st.session_state.setdefault("report_md", None)
    st.session_state.setdefault("report_meta", None)
    st.session_state.setdefault("report_pdf_bytes", None)
    st.session_state.setdefault("assistant_messages", [])
    st.session_state.setdefault("last_gemini_payload", None)


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


# --- Wizard persistence fix ---
# Widgets use step-scoped keys; values are synced into st.session_state["inputs"] (persistent).
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

    # IMPORTANT: Streamlit calls format_func(option). Passing None causes:
    # TypeError: 'NoneType' object is not callable
    select_kwargs = dict(options=options, key=k, on_change=_sync, **kwargs)
    if callable(format_func):
        select_kwargs["format_func"] = format_func

    st.selectbox(label, **select_kwargs)


def recalc_from_inputs():
    inp = st.session_state["inputs"]
    st.session_state["model_results"] = run_model(
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
        existing_site_peak_kw=inp["existing_peak_kw"],
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
    )


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
        "narrative": "Gemini narrative (customer-ready)",
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
        "narrative": "Gemini-Text (kundenfertig)",
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
# Gemini helpers
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
    return {
        "inputs": {
            "num_trucks": inp.get("num_trucks"),
            "operating_days": inp.get("operating_days"),
            "battery_kwh": inp.get("battery_kwh"),
            "start_soc": inp.get("start_soc"),
            "target_soc": inp.get("target_soc"),
            "avg_elec_price_eur_per_mwh": inp.get("avg_elec_price_eur_per_mwh"),
            "diesel_price_eur_per_l": inp.get("diesel_price_eur_per_l"),
            "diesel_l_per_100km": inp.get("diesel_l_per_100km"),
            "toll_rate_eur_per_km": inp.get("toll_rate_eur_per_km"),
            "ev_toll_exempt": inp.get("ev_toll_exempt"),
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


# =========================================================
# UI helpers
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


def build_report_pdf_bytes(title: str, kpis: dict, narrative_md: str, issues: list, solutions: list, lang: str) -> bytes:
    """
    Build a clean PDF using reportlab (no HTML renderer needed).
    Returns b"" if reportlab is missing.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
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

    doc.build(story)
    return buf.getvalue()



def reset_all():
    lang = st.session_state.get("lang", DEFAULT_LANG)
    st.session_state["inputs"] = DEFAULT_INPUTS.copy()
    st.session_state["lang"] = lang
    st.session_state["wizard_step"] = 0
    st.session_state["flow"] = "wizard"
    st.session_state["report_md"] = None
    st.session_state["report_meta"] = None
    st.session_state["report_pdf_bytes"] = None
    st.session_state["assistant_messages"] = []
    st.session_state["last_gemini_payload"] = None


def apply_pending_flow():
    """Apply deferred flow changes BEFORE the flow radio is instantiated."""
    v = st.session_state.pop("pending_flow_switch", None)
    if v in ("wizard", "report"):
        st.session_state["flow"] = v
        # Important: do NOT set the widget key directly here.
        # Instead, remove it so the radio can initialise cleanly (no Streamlit warning).
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
# Page config + styling (darker background)
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

.block-container {padding-top: 2.2rem; padding-bottom: 2.8rem; max-width: 1320px;}
div[data-testid="stVerticalBlock"]{gap: 0.75rem;}

div[data-testid="stContainer"]{
  background: linear-gradient(180deg, var(--card2), var(--card));
  border: 1px solid var(--stroke) !important;
  border-radius: var(--radius) !important;
  box-shadow: var(--shadow3);
  padding: 1.05rem 1.1rem;
  backdrop-filter: blur(14px);
}

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
label, .stMarkdown, .stCaption {color: var(--txt) !important;}
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

/* ===== Dark-mode fixes for dropdowns / menus / sliders / metrics ===== */
div[data-baseweb="select"] > div{
  background: rgba(255,255,255,0.06) !important;
  border-color: rgba(255,255,255,0.16) !important;
}
div[data-baseweb="select"] *{ color: var(--txt) !important; }
div[data-baseweb="select"] svg{ fill: var(--txt) !important; }

div[data-baseweb="popover"] > div{
  background: rgba(8,10,18,0.98) !important;
  border: 1px solid rgba(255,255,255,0.16) !important;
  box-shadow: 0 24px 72px rgba(0,0,0,0.55) !important;
}
ul[role="listbox"]{ background: transparent !important; }
li[role="option"]{ color: var(--txt) !important; }
li[role="option"]:hover{ background: rgba(255,255,255,0.08) !important; }

div[data-baseweb="slider"] *{ color: var(--txt) !important; }
div[data-testid="stMetric"] *{ color: var(--txt) !important; }
div[data-testid="stMetricLabel"]{ color: var(--muted) !important; }
div[data-testid="stMetricValue"]{ color: var(--txt) !important; }
div[data-testid="stMetricDelta"]{ color: var(--muted) !important; }

/* Expanders */
details summary, details summary *{ color: var(--txt) !important; }


/* Inputs: light surface + dark text (readable on dark cards) */
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea,
div[data-baseweb="select"] > div {
  background: rgba(255,255,255,0.92) !important;
  color: rgba(8,10,16,0.96) !important;
}
div[data-baseweb="input"] input::placeholder,
div[data-baseweb="textarea"] textarea::placeholder {
  color: rgba(8,10,16,0.45) !important;
}

/* Selectbox option list (dropdown) */
div[data-baseweb="select"] span,
div[data-baseweb="select"] input{
  color: rgba(8,10,16,0.96) !important;
}
div[data-baseweb="popover"] ul[role="listbox"]{
  background: rgba(255,255,255,0.98) !important;
  border: 1px solid rgba(10,10,20,0.16) !important;
}
div[data-baseweb="popover"] ul[role="listbox"] *{
  color: rgba(8,10,16,0.96) !important;
}

/* Make widget labels on dark cards readable */
div[data-testid="stContainer"] label,
div[data-testid="stContainer"] p,
div[data-testid="stContainer"] span {
  color: rgba(245,247,255,0.92);
}

/* Flow switch (questionnaire/report) readable */
div[data-testid="stRadio"][data-baseweb="radio-group"] label,
div[data-testid="stRadio"] label {
  color: rgba(245,247,255,0.92) !important;
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
     "sub_en": "How much energy each charging event represents.", "sub_de": "Wie viel Energie ein Lade-Event bedeutet."},
    {"id": "electricity", "title_en": "Electricity & Overnight Charging", "title_de": "Strom & Übernacht-Laden",
     "sub_en": "Price + typical charging hours.", "sub_de": "Preis + typische Ladezeiten."},
    {"id": "site", "title_en": "Site Capacity & Chargers", "title_de": "Standortkapazität & Lader",
     "sub_en": "Connection limit + charger power.", "sub_de": "Anschlusslimit + Ladeleistung."},
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
recalc_from_inputs()

top_l, top_r = st.columns([0.70, 0.30], vertical_alignment="center")
with top_l:
    st.markdown(f"<h1 class='gs-title'>{t('welcome_title')}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='gs-sub'>{t('welcome_sub')}</p>", unsafe_allow_html=True)

with top_r:
    current = st.session_state.get("lang", DEFAULT_LANG)
    lang = st.radio(
        "",
        options=["DE", "EN"],
        index=0 if current == "DE" else 1,
        horizontal=True,
        label_visibility="collapsed",
        key="lang_switch",
    )
    if lang != current:
        st.session_state["lang"] = lang
        st.session_state["report_md"] = None
        st.session_state["report_meta"] = None
        st.session_state["report_pdf_bytes"] = None
        st.rerun()

left, right = st.columns([0.62, 0.38], gap="large")

# ---------- Right: Live preview + Assistant ----------
with right:
    i, step = current_step()
    step_name = step_title(step)

    with st.container(border=True):
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

    with st.container(border=True):
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
        new_flow = mode
        if new_flow != st.session_state.get("flow"):
            st.session_state["flow"] = new_flow
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
            sid = step["id"]
            lang = st.session_state.get("lang", DEFAULT_LANG)

            if sid == "fleet":
                bind_number(
                    sid, "num_trucks",
                    "How many trucks charge per day?" if lang == "EN" else "Wie viele LKW laden pro Tag?",
                    min_value=1, max_value=10000, step=1,
                    help=("Total trucks that typically need charging on a normal day." if lang == "EN"
                          else "Wie viele Fahrzeuge typischerweise an einem normalen Tag laden müssen.")
                )
                bind_number(
                    sid, "operating_days",
                    "Operating days per year" if lang == "EN" else "Betriebstage pro Jahr",
                    min_value=1, max_value=366, step=1,
                    help=("How many days per year the fleet operates (used to annualise costs and CO₂)." if lang == "EN"
                          else "Wie viele Tage pro Jahr die Flotte fährt (für Jahreswerte von Kosten/CO₂).")
                )
                bind_number(
                    sid, "events_per_truck",
                    "Charging events per truck per day" if lang == "EN" else "Lade-Events pro LKW pro Tag",
                    min_value=0.0, max_value=10.0, step=0.25,
                    help=("Example: 1.0 = once per day; 1.5 = some days twice." if lang == "EN"
                          else "Beispiel: 1,0 = 1× pro Tag; 1,5 = an manchen Tagen 2×.")
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
                        "battery_kwh": 220.0, "ev_consumption": 1.0,
                        "desc_en": "Short routes, lots of stops. Typical battery: 150–300 kWh. Typical consumption: 0.7–1.3 kWh/km.",
                        "desc_de": "Kurze Strecken, viele Stopps. Typische Batterie: 150–300 kWh. Typischer Verbrauch: 0,7–1,3 kWh/km.",
                    },
                    "regional": {
                        "label_en": "Regional Distribution", "label_de": "Regionale Distribution",
                        "battery_kwh": 350.0, "ev_consumption": 1.4,
                        "desc_en": "Mixed urban + regional. Typical battery: 250–450 kWh. Typical consumption: 1.0–1.8 kWh/km.",
                        "desc_de": "Mix aus Stadt + Region. Typische Batterie: 250–450 kWh. Typischer Verbrauch: 1,0–1,8 kWh/km.",
                    },
                    "heavy_regional": {
                        "label_en": "Heavy Regional (mixed)", "label_de": "Schwer regional (gemischt)",
                        "battery_kwh": 500.0, "ev_consumption": 1.8,
                        "desc_en": "Heavier loads, more motorway. Typical battery: 350–600 kWh. Typical consumption: 1.3–2.2 kWh/km.",
                        "desc_de": "Schwerere Lasten, mehr Autobahn. Typische Batterie: 350–600 kWh. Typischer Verbrauch: 1,3–2,2 kWh/km.",
                    },
                    "long_haul": {
                        "label_en": "Long‑haul / 40t", "label_de": "Langstrecke / 40t",
                        "battery_kwh": 650.0, "ev_consumption": 2.2,
                        "desc_en": "High speed + high mass. Typical battery: 500–900 kWh. Typical consumption: 1.6–2.8 kWh/km.",
                        "desc_de": "Hohe Geschwindigkeit + hohe Masse. Typische Batterie: 500–900 kWh. Typischer Verbrauch: 1,6–2,8 kWh/km.",
                    },
                }

                bind_select(
                    sid, "vehicle_profile",
                    "Vehicle profile (recommended defaults)" if lang == "EN" else "Fahrprofil (empfohlene Defaults)",
                    options=list(PROFILES.keys()),
                    format_func=lambda k: PROFILES[k]["label_en"] if lang == "EN" else PROFILES[k]["label_de"],
                    help=("Pick what fits best — you can still adjust the numbers below." if lang == "EN"
                          else "Wähle das passendste Profil — du kannst die Werte unten trotzdem anpassen.")
                )
                prof = PROFILES.get(get_inp("vehicle_profile"), PROFILES["heavy_regional"])
                st.info(prof["desc_en"] if lang == "EN" else prof["desc_de"])

                c_apply, c_note = st.columns([0.42, 0.58])
                with c_apply:
                    if st.button("✨ " + t("apply_defaults"), use_container_width=True, key="apply_profile_defaults"):
                        set_inp("battery_kwh", prof["battery_kwh"])
                        set_inp("ev_consumption", prof["ev_consumption"])
                        st.session_state[wkey("battery", "battery_kwh")] = get_inp("battery_kwh")
                        st.session_state[wkey("battery", "ev_consumption")] = get_inp("ev_consumption")
                        st.rerun()
                with c_note:
                    st.markdown(
                        "<div class='small-note'>"
                        + ("Pragmatic real‑world defaults. Fine‑tune if you have fleet data." if lang == "EN"
                           else "Pragmatische Real‑World Defaults. Wenn du Flottendaten hast: feinjustieren.")
                        + "</div>",
                        unsafe_allow_html=True
                    )

                gs_hr()

                bind_number(
                    sid, "battery_kwh",
                    "Average battery capacity (kWh)" if lang == "EN" else "Durchschnittliche Batteriekapazität (kWh)",
                    min_value=1.0, max_value=2000.0, step=10.0,
                )
                bind_slider(
                    sid, "start_soc",
                    "Typical return SoC (how empty?)" if lang == "EN" else "Typischer Rückkehr-SoC (wie leer?)",
                    0.0, 1.0, 0.05,
                    help=("Default 20%." if lang == "EN" else "Default 20%.")
                )
                bind_slider(
                    sid, "target_soc",
                    "Target SoC (how full?)" if lang == "EN" else "Ziel-SoC (wie voll?)",
                    0.0, 1.0, 0.05,
                    help=("Default 100% (simple baseline)." if lang == "EN" else "Default 100% (einfache Basis).")
                )
                bind_number(
                    sid, "ev_consumption",
                    "EV consumption (kWh per km)" if lang == "EN" else "EV-Verbrauch (kWh pro km)",
                    min_value=0.1, max_value=10.0, step=0.1,
                )
                if float(get_inp("start_soc")) >= float(get_inp("target_soc")):
                    st.warning(
                        "Start SoC is not below Target SoC — that implies no charging energy." if lang == "EN"
                        else "Start-SoC ist nicht unter dem Ziel-SoC — das bedeutet praktisch keine Ladeenergie."
                    )

            elif sid == "electricity":
                bind_number(
                    sid, "avg_elec_price_mwh",
                    "Average electricity price (€/MWh)" if lang == "EN" else "Ø Strompreis (€/MWh)",
                    min_value=0.0, max_value=2000.0, step=10.0,
                    help=("Default 200 €/MWh (= 0.20 €/kWh)." if lang == "EN"
                          else "Default 200 €/MWh (= 0,20 €/kWh).")
                )
                bind_slider(
                    sid, "dynamic_share",
                    "Dynamic price share (0–1)" if lang == "EN" else "Dynamischer Preisanteil (0–1)",
                    0.0, 1.0, 0.05,
                    help=("Default 30%." if lang == "EN" else "Default 30%.")
                )

                c1, c2 = st.columns(2)
                with c1:
                    bind_slider(
                        sid, "start_hour",
                        "Charging start hour" if lang == "EN" else "Ladebeginn (Stunde)",
                        0, 23, 1,
                        help=("Default 22:00 (overnight)." if lang == "EN" else "Default 22:00 (Übernacht).")
                    )
                with c2:
                    bind_slider(
                        sid, "end_hour",
                        "Charging end hour" if lang == "EN" else "Ladeende (Stunde)",
                        0, 23, 1,
                        help=("Default 08:00 (overnight)." if lang == "EN" else "Default 08:00 (Übernacht).")
                    )

                wh = window_len_hours(get_inp("start_hour"), get_inp("end_hour"))
                st.markdown(
                    "<div class='small-note'>"
                    + (f"Your time window spans <b>{wh} hours</b>. Now set how many hours you are actually charging (effective charging time)."
                       if lang == "EN"
                       else f"Dein Zeitfenster umfasst <b>{wh} Stunden</b>. Jetzt: Wie viele Stunden davon wird wirklich effektiv geladen (effektive Ladezeit).")
                    + "</div>",
                    unsafe_allow_html=True
                )

                cW1, cW2 = st.columns([0.70, 0.30])
                with cW1:
                    bind_number(
                        sid, "charging_window_hours",
                        "Effective charging time (hours)" if lang == "EN" else "Effektive Ladezeit (Stunden)",
                        min_value=0.5, max_value=24.0, step=0.5,
                        help=("Usually ≤ window length. Example: plugged in 22–08 but only charges 6h → enter 6."
                              if lang == "EN"
                              else "Meist ≤ Fensterlänge. Beispiel: 22–08 eingesteckt, aber nur 6h aktiv → 6 eintragen.")
                    )
                with cW2:
                    if st.button(t("apply_window"), use_container_width=True, key="apply_full_window"):
                        set_inp("charging_window_hours", float(wh))
                        st.session_state[wkey("electricity", "charging_window_hours")] = float(wh)
                        st.rerun()

            elif sid == "site":
                bind_number(
                    sid, "existing_peak_kw",
                    "Existing site peak load (kW)" if lang == "EN" else "Bestehende Peak-Last (kW)",
                    min_value=0.0, max_value=100000.0, step=10.0,
                )
                bind_number(
                    sid, "charger_power_kw",
                    "Charger power per truck (kW)" if lang == "EN" else "Ladeleistung pro LKW (kW)",
                    min_value=0.0, max_value=2000.0, step=10.0,
                )
                bind_number(
                    sid, "site_capacity_kva",
                    "Site capacity limit (kVA)" if lang == "EN" else "Standort-Limit (kVA)",
                    min_value=0.0, max_value=100000.0, step=10.0,
                )

                gs_hr()
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
                PRESETS = {
                    "AT": {"diesel_price": 1.75, "toll_rate": 0.25, "tolled_share": 0.60},
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
                    min_value=0.0, max_value=5.0, step=0.01,
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
            issues = detect_issues(res)
            solutions = generate_solution_set(res, issues)

            ec = res["energy_cost"]
            dv = res["diesel_vs_ev"]
            load = res["load"]

            a, b, c = st.columns(3)
            with a: st.metric(t("kpi_savings"), fmt_eur(dv.get("total_savings_incl_toll_eur")))
            with b: st.metric("EV cost / year" if st.session_state["lang"] == "EN" else "EV-Kosten / Jahr", fmt_eur(ec.get("annual_cost_eur")))
            with c: st.metric(t("kpi_peak"), f"{fmt_num(load.get('new_theoretical_peak_kw'))} kW")

            gs_hr()

            with st.container(border=True):
                st.markdown("### " + t("narrative"))
                if st.session_state.get("report_md") is None:
                    md = call_gemini_report(res, issues, solutions)
                    st.session_state["report_md"] = md if md else t("gemini_missing")
                    st.session_state["report_pdf_bytes"] = None
                    st.session_state["report_meta"] = {"generated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z"}
                st.markdown(st.session_state["report_md"])

            gs_hr()

            with st.container(border=True):
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

            
            with st.container(border=True):
                st.markdown("### " + ("Download" if st.session_state["lang"] == "EN" else "Download"))
                cA, cB = st.columns(2)
                with cA:
                    if st.button("↺ " + t("restart"), use_container_width=True, key="restart_btn_report"):
                        reset_all()
                        st.rerun()
                with cB:
                    # (Re)build PDF if needed
                    if st.session_state.get("report_pdf_bytes") is None:
                        kpis = {
                            f"{t('kpi_savings')} ({t('per_year')})": fmt_eur(dv.get("total_savings_incl_toll_eur")),
                            f"{t('kpi_co2')} ({t('per_year')})": fmt_kg(dv.get("co2_savings_kg")),
                            t("kpi_peak"): f"{fmt_num(load.get('new_theoretical_peak_kw'))} kW",
                            ("EV cost / year" if st.session_state["lang"] == "EN" else "EV-Kosten / Jahr"): fmt_eur(ec.get("annual_cost_eur")),
                        }
                        title = "FleetMate — Report"
                        st.session_state["report_pdf_bytes"] = build_report_pdf_bytes(
                            title=title,
                            kpis=kpis,
                            narrative_md=st.session_state.get("report_md") or "",
                            issues=issues,
                            solutions=solutions,
                            lang=st.session_state.get("lang", "EN"),
                        )

                    pdf_bytes = st.session_state.get("report_pdf_bytes") or b""
                    if not pdf_bytes:
                        st.info("PDF export needs the Python package 'reportlab'. Add it to requirements.txt and rebuild Docker." if st.session_state["lang"] == "EN"
                                else "PDF-Export braucht das Python-Paket 'reportlab'. Füge es zur requirements.txt hinzu und baue Docker neu.")
                    else:
                        st.download_button(
                            "⬇️ Download PDF report" if st.session_state["lang"] == "EN" else "⬇️ PDF-Report downloaden",
                            data=pdf_bytes,
                            file_name="fleetmate_report.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            key="dl_pdf_report",
                        )