import os
import json
import requests
import streamlit as st
import pandas as pd
from recommendations import detect_issues, generate_solution_set


# 0) CONSTANTS
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

# 1) MODEL
def compute_flags_and_shares(start_hour: int, end_hour: int):
    flags = [0] * 24
    for h in range(24):
        if start_hour < end_hour:
            flags[h] = 1 if start_hour <= h < end_hour else 0
        elif start_hour > end_hour:
            flags[h] = 1 if (h >= start_hour or h < end_hour) else 0
        else:
            flags[h] = 1  # full day

    total = sum(flags)
    shares = [(f / total) if total > 0 else 0.0 for f in flags]
    return flags, shares


def effective_grid_co2_kg_per_kwh(shares):
    g = sum(c * s for c, s in zip(GRID_CO2_G_PER_KWH, shares))
    return g / 1000.0


def effective_energy_price_eur_per_mwh(avg_price_eur_per_mwh: float, dynamic_share: float, shares):
    avg_price_eur_per_kwh = max(0.0, avg_price_eur_per_mwh) / 1000.0
    dynamic_share = min(max(dynamic_share, 0.0), 1.0)

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
    charging_hours_by_clock = sum(flags)

    # Energy
    soc_diff = max(0.0, target_soc - start_soc)
    energy_per_event_kwh = battery_kwh * soc_diff
    energy_per_event_mwh = energy_per_event_kwh / 1000.0

    total_daily_energy_mwh = num_trucks * events_per_truck_per_day * energy_per_event_mwh
    annual_energy_mwh = total_daily_energy_mwh * operating_days

    eff_price_eur_per_mwh = effective_energy_price_eur_per_mwh(
        avg_elec_price_eur_per_mwh, dynamic_price_share, shares
    )

    daily_cost_eur = total_daily_energy_mwh * eff_price_eur_per_mwh
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
            "charging_hours_by_clock": charging_hours_by_clock,
            "flags": flags,
            "shares": shares,
            "grid_co2_g_per_kwh": GRID_CO2_G_PER_KWH,
            "tou_price_eur_per_kwh": TOU_PRICE_EUR_PER_KWH,
        },
        "energy_cost": {
            "soc_diff": soc_diff,
            "energy_per_event_mwh": energy_per_event_mwh,
            "total_daily_energy_mwh": total_daily_energy_mwh,
            "annual_energy_mwh": annual_energy_mwh,
            "effective_price_eur_per_mwh": eff_price_eur_per_mwh,
            "daily_cost_eur": daily_cost_eur,
            "annual_cost_eur": annual_cost_eur,
        },
        "co2": {
            "effective_grid_co2_kg_per_kwh": eff_grid_co2_kg_per_kwh,
            "annual_ev_co2_kg": annual_ev_co2_kg,
        },
        "distance": {
            "kwh_per_truck_per_day": kwh_per_truck_per_day,
            "km_per_truck_per_day": km_per_truck_per_day,
            "annual_km_per_truck": annual_km_per_truck,
            "annual_km_fleet": annual_km_fleet,
        },
        "diesel_vs_ev": {
            "diesel_litres_baseline": diesel_litres_baseline,
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
            "avg_charging_power_kw": avg_charging_power_kw,
            "new_avg_load_kw": new_avg_load_kw,
            "capacity_ok": capacity_ok,
            "required_shaving_kw": required_shaving_kw,
            "required_battery_energy_kwh": required_battery_energy_kwh,
        },
    }

def recalc_from_state():
    st.session_state["model_results"] = run_model(
        num_trucks=st.session_state.get("num_trucks", 10),
        operating_days=st.session_state.get("operating_days", 260),
        events_per_truck_per_day=st.session_state.get("events_per_truck", 1.0),
        battery_kwh=st.session_state.get("battery_kwh", 500.0),
        start_soc=st.session_state.get("start_soc", 0.2),
        target_soc=st.session_state.get("target_soc", 1.0),
        avg_elec_price_eur_per_mwh=st.session_state.get("avg_elec_price_mwh", 200.0),
        dynamic_price_share=st.session_state.get("dynamic_share", 1.0),
        start_hour=st.session_state.get("start_hour", 6),
        end_hour=st.session_state.get("end_hour", 20),
        charging_window_hours=st.session_state.get("charging_window_hours", 14.0),
        existing_site_peak_kw=st.session_state.get("existing_peak_kw", 300.0),
        charger_power_per_truck_kw=st.session_state.get("charger_power_kw", 150.0),
        site_capacity_limit_kva=st.session_state.get("site_capacity_kva", 630.0),
        ev_consumption_kwh_per_km=st.session_state.get("ev_consumption", 1.6),
        diesel_price_eur_per_l=st.session_state.get("diesel_price", 1.8),
        diesel_l_per_100km=st.session_state.get("diesel_l_per_100", 28.0),
        toll_rate_eur_per_km=st.session_state.get("toll_rate", 0.25),
        tolled_share_0_1=st.session_state.get("tolled_share", 0.6),
        ev_toll_exempt=st.session_state.get("ev_toll_exempt", True),
        desired_peak_limit_kw=st.session_state.get("desired_peak_limit_kw", 0.0),
        peak_duration_h=st.session_state.get("peak_duration_h", 0.25),
    )

# FleetMate
if "assistant_is_open" not in st.session_state:
    st.session_state["assistant_is_open"] = False

# Open when URL has ?fleetmate=1
if st.query_params.get("fleetmate") == "1":
    st.session_state["assistant_is_open"] = True
    # optional: clean URL so it won't auto-open on refresh
    try:
        del st.query_params["fleetmate"]
    except Exception:
        pass


# 2) GEMINI CALL
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
            "battery_kwh": inp.get("battery_kwh"),
            "start_soc": inp.get("start_soc"),
            "target_soc": inp.get("target_soc"),
            "avg_elec_price_eur_per_mwh": inp.get("avg_elec_price_eur_per_mwh"),
            "diesel_price_eur_per_l": inp.get("diesel_price_eur_per_l"),
            "diesel_l_per_100km": inp.get("diesel_l_per_100km"),
        },
        "key_results": {
            "daily_energy_mwh": ec.get("total_daily_energy_mwh"),
            "annual_energy_mwh": ec.get("annual_energy_mwh"),
            "effective_price_eur_per_mwh": ec.get("effective_price_eur_per_mwh"),
            "annual_ev_cost_eur": ec.get("annual_cost_eur"),
            "baseline_diesel_cost_eur": dv.get("diesel_cost_baseline_eur"),
            "cost_savings_eur": dv.get("cost_savings_eur"),
            "annual_ev_co2_kg": co2.get("annual_ev_co2_kg"),
            "baseline_diesel_co2_kg": dv.get("diesel_co2_baseline_kg"),
            "co2_savings_kg": dv.get("co2_savings_kg"),
            "annual_km_fleet": dist.get("annual_km_fleet"),
            "new_theoretical_peak_kw": load.get("new_theoretical_peak_kw"),
            "capacity_ok": load.get("capacity_ok"),
        }
    }

ALLOWED_INPUT_KEYS = {
    "num_trucks", "operating_days", "events_per_truck",
    "battery_kwh", "start_soc", "target_soc", "ev_consumption",
    "avg_elec_price_mwh", "dynamic_share",
    "start_hour", "end_hour", "charging_window_hours",
    "existing_peak_kw", "charger_power_kw", "site_capacity_kva",
    "desired_peak_limit_kw", "peak_duration_h",
    "diesel_price", "diesel_l_per_100",
    "toll_rate", "tolled_share", "ev_toll_exempt",
}

def _extract_json(text: str) -> str:
    t = text.strip()
    if "```" in t:
        t = t.replace("```json", "```")
        parts = t.split("```")
        if len(parts) >= 3:
            t = parts[1].strip()
    i, j = t.find("{"), t.rfind("}")
    return t[i:j+1] if i != -1 and j != -1 and j > i else t


def call_gemini_assistant(user_msg: str, results: dict) -> dict:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        try:
            api_key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
        except Exception:
            api_key = ""

    if not api_key:
        return {"reply": "GEMINI_API_KEY is missing.", "update_inputs": None, "show_payload": False}

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite").strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    params = {"key": api_key}

    compact = _compact_for_llm(results)
    current_inputs = {k: st.session_state.get(k) for k in sorted(ALLOWED_INPUT_KEYS)}

    prompt = (
        "You are 'FleetMate', a cool, practical assistant for an EV-vs-Diesel fleet calculator.\n"
        "You ALWAYS reply in English.\n\n"
        "You may propose input changes. If the user asks to change parameters, return them in update_inputs.\n"
        "Only use keys from the provided inputs.\n\n"
        "Return ONLY valid JSON with exactly these keys:\n"
        "reply: string (English)\n"
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
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 300,
        }
    }

    # store the actual API payload for "show me the JSON you send"
    st.session_state["last_gemini_payload"] = {"url": url, "params": params, "payload": payload, "model": model}

    try:
        resp = requests.post(url, params=params, json=payload, timeout=20)
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

    # parse strict JSON
    try:
        obj = json.loads(_extract_json(text))
    except Exception:
        # fallback: show raw text
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



# 3) STREAMLIT UI
st.set_page_config("Electric Vehicle Calculator", layout="wide")
st.title("Electric Vehicle Calculator")

if "model_results" not in st.session_state:
    st.session_state["model_results"] = None

with st.sidebar:
    st.header("Inputs (Excel Logic)")

    st.subheader("Fleet & Operations")
    num_trucks = st.number_input("Number of trucks charging per day", 1, 10000, 10, 1, key="num_trucks")
    operating_days = st.number_input("Operating days per year", 1, 366, 260, 1, key="operating_days")
    events_per_truck = st.number_input("Charging events per truck per day", 0.0, 10.0, 1.0, 0.25, key="events_per_truck")

    st.subheader("Battery & Consumption")
    battery_kwh = st.number_input("Average battery capacity (kWh)", 1.0, 2000.0, 500.0, 10.0, key="battery_kwh")
    start_soc = st.slider("Returning rate / Start SoC (0–1)", 0.0, 1.0, 0.2, 0.05, key="start_soc")
    target_soc = st.slider("Target SoC (0–1)", 0.0, 1.0, 1.0, 0.05, key="target_soc")
    ev_consumption = st.number_input("EV consumption (kWh/km) (Excel default ~1.6)", 0.1, 10.0, 1.6, 0.1, key="ev_consumption")

    st.subheader("Electricity price")
    avg_elec_price_mwh = st.number_input("Average electricity price (€/MWh)", 0.0, 2000.0, 200.0, 10.0, key="avg_elec_price_mwh")
    dynamic_share = st.slider("Dynamic price share (0–1)", 0.0, 1.0, 1.0, 0.05, key="dynamic_share")

    st.caption("Charging time window (for dynamic CO₂ & dynamic price weighting)")
    start_hour = st.slider("Start charging hour", 0, 23, 6, 1, key="start_hour")
    end_hour = st.slider("End charging hour", 0, 23, 20, 1, key="end_hour")
    charging_window_hours = st.number_input("Charging window per day (hours)", 0.5, 24.0, 14.0, 0.5, key="charging_window_hours")

    st.subheader("Site load / grid")
    existing_peak_kw = st.number_input("Existing site peak load (kW)", 0.0, 100000.0, 300.0, 10.0, key="existing_peak_kw")
    charger_power_kw = st.number_input("Charger power per truck (kW)", 0.0, 2000.0, 150.0, 10.0, key="charger_power_kw")
    site_capacity_kva = st.number_input("Site capacity limit (kVA)", 0.0, 100000.0, 630.0, 10.0, key="site_capacity_kva")

    st.subheader("Optional: Peak shaving (minimal)")
    desired_peak_limit_kw = st.number_input("Desired peak limit (kW) (0 = off)", 0.0, 200000.0, 0.0, 50.0, key="desired_peak_limit_kw")
    peak_duration_h = st.number_input("Peak duration (h)", 0.0, 24.0, 0.25, 0.05, key="peak_duration_h")

    st.subheader("Diesel baseline")
    diesel_price = st.number_input("Diesel price (€/L)", 0.0, 5.0, 1.8, 0.05, key="diesel_price")
    diesel_l_per_100 = st.number_input("Diesel consumption (L/100km)", 0.0, 200.0, 28.0, 1.0, key="diesel_l_per_100")

    st.subheader("Toll (optional)")
    toll_rate = st.number_input("Toll rate (€/km)", 0.0, 5.0, 0.25, 0.01, key="toll_rate")
    tolled_share = st.slider("% of distance tolled", 0.0, 1.0, 0.6, 0.05, key="tolled_share")
    ev_toll_exempt = st.checkbox("Assume EV toll exempt", value=True, key="ev_toll_exempt")

    run_button = st.button("Run calculation")

if st.session_state.get("model_results") is None:
    recalc_from_state()

if run_button:
    recalc_from_state()

results = st.session_state["model_results"]

st.subheader("Results")

if results is None:
    st.info("Click **Run calculation** in the sidebar to see results.")
else:
    ec = results["energy_cost"]
    co2 = results["co2"]
    dist = results["distance"]
    load = results["load"]
    dv = results["diesel_vs_ev"]
    inp = results["inputs"]
    prof = results["charging_profile"]
      
    def _fmt_eur(x): 
        return f"{x:,.0f} €"

    def _fmt_kg(x):  
        return f"{x:,.0f} kg"

    def _recommendation(kind: str, text: str):
        # kind: "success" | "warning" | "info" | "error"
        if kind == "success":
            st.success(text)
        elif kind == "warning":
            st.warning(text)
        elif kind == "error":
            st.error(text)
        else:
            st.info(text)

    total_savings = float(dv.get("total_savings_incl_toll_eur", 0.0) or 0.0)
    cost_savings  = float(dv.get("cost_savings_eur", 0.0) or 0.0)
    co2_savings   = float(dv.get("co2_savings_kg", 0.0) or 0.0)
    cap_ok        = bool(load.get("capacity_ok", True))
    dyn_share     = float(inp.get("dynamic_price_share", 1.0) or 0.0)

    annual_km = float(dist.get("annual_km_fleet", 0.0) or 0.0)
    annual_km_safe = annual_km if annual_km > 0 else 1.0

    # ---- TOP: Executive KPIs (clean + fast) ----
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total savings incl. toll (€ / year)", f"{dv['total_savings_incl_toll_eur']:,.0f}")
    with k2:
        st.metric("CO₂ savings vs diesel (kg / year)", f"{dv['co2_savings_kg']:,.0f}")
    with k3:
        st.metric("Annual energy (MWh)", f"{ec['annual_energy_mwh']:,.0f}")
    with k4:
        st.metric("Capacity check", "Adequate" if load["capacity_ok"] else "Exceeds")


    st.markdown("")

    
    # ---- TABS for structure ----
    tab_overview, tab_finance, tab_emissions, tab_grid, tab_hourly, tab_solutions, tab_details = st.tabs(
    ["Overview", "Finance", "CO₂", "Grid / Load", "Hourly profile", "Solutions", "Details"]
    )


    issues = detect_issues(results)
    solutions = generate_solution_set(results, issues)

    with tab_solutions:
        st.markdown("## Solution options")

        if not issues:
            st.success("No critical issues detected. Current configuration is technically and economically feasible.")
        else:
            st.markdown("### Detected issues")
            for i in issues:
                st.warning(i["description"])
    
            st.markdown("### Recommended Solution Paths")
            for idx, s in enumerate(solutions, start=1):
                with st.container(border=True):
                    st.markdown(f"### {idx}. {s['title']}")

                    st.markdown("**What it is?**")
                    st.write(s["definition"])

                    st.markdown("**How to implement?**")
                    for h in s["how_to"]:
                        st.write(f"• {h}")

                    st.markdown("**Advantages**")
                    for p in s["pros"]:
                        st.write(f"• {p}")

                    st.markdown("**Disadvantages**")
                    for c in s["cons"]:
                        st.write(f"• {c}")

                    st.markdown("**Quantitative impact**")
                    for k, v in s["quantitative"].items():
                        st.write(f"- {k.replace('_',' ')}: **{v}**")



    # =========================
    # TAB: OVERVIEW
    # =========================
    with tab_overview:
        if total_savings > 0 and co2_savings > 0:
            _recommendation(
                "success",
                f"Under these assumptions, the Electric Vehicle setup is favourable if compared with diesel (OPEX). "
                f"Total benefit including toll is {_fmt_eur(total_savings)} per year,"
                f"while the CO₂ reduction is {_fmt_kg(co2_savings)} per year."
                "Find more details on the subsequent tabs and recommendations/solutions for potential issues in 'Solutions' tab."
                )
        elif total_savings > 0 and co2_savings <= 0:
            _recommendation(
                "warning",
                "Operating costs are lower for EVs under current assumptions; however, CO₂ emissions are not reduced. "
                "This is driven by the assumed charging window and grid mix."
                "Find more details on the subsequent tabs and recommendations/solutions for potential issues in 'Solutions' tab."
            )
        else:
            _recommendation(
                "warning",
                "Under the current assumptions, diesel shows lower operating costs than Electric Vehicles. "
                "This result is driven by electricity price, energy consumption, charging profile, and toll assumptions."
                "Find more details on the subsequent tabs and recommendations/solutions for potential issues in 'Solutions' tab."
                )
        
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("#### Fleet distance")
            st.metric("km per truck per day", f"{dist['km_per_truck_per_day']:,.1f}")
            st.metric("annual km fleet", f"{dist['annual_km_fleet']:,.0f}")

        with c2:
            st.markdown("#### Cost per km (excl. capex)")
            diesel_cost_per_km = dv["diesel_cost_baseline_eur"] / annual_km_safe
            ev_cost_per_km = ec["annual_cost_eur"] / annual_km_safe
            st.metric("Diesel €/km", f"{diesel_cost_per_km:,.3f}")
            st.metric("EV electricity €/km", f"{ev_cost_per_km:,.3f}")

        with c3:
            st.markdown("#### Charging + load")
            st.metric("New theoretical peak (kW)", f"{load['new_theoretical_peak_kw']:,.0f}")
            st.metric("New avg load (kW)", f"{load['new_avg_load_kw']:,.0f}")

        df_km = pd.DataFrame({
            "Scenario": ["Diesel (fuel only)", "EV (electricity only)"],
            "€/km": [diesel_cost_per_km, ev_cost_per_km]
        }).set_index("Scenario")
        st.bar_chart(df_km)

    # =========================
    # TAB: FINANCE
    # =========================
    with tab_finance:
        if total_savings > 0:
            _recommendation(
                "success",
                f"Cost comparison: The Electric Vehicle scenario results in lower variable operating costs than diesel "
                f"Energy cost delta is {_fmt_eur(cost_savings)} per year, increasing to {_fmt_eur(total_savings)} per year when toll effects are included."
            )
        else:
            _recommendation(
                "warning",
                 "Cost comparison: Under the current inputs, EV operating costs exceed the diesel baseline. "
                "This outcome is sensitive to electricity price levels, charging timing, and vehicle energy consumption."
            )


        st.markdown("#### Annual costs")
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            st.metric("EV electricity cost (€)", f"{ec['annual_cost_eur']:,.0f}")
        with f2:
            st.metric("Diesel baseline cost (€)", f"{dv['diesel_cost_baseline_eur']:,.0f}")
        with f3:
            st.metric("Cost savings (€)", f"{dv['cost_savings_eur']:,.0f}")
        with f4:
            st.metric("Toll savings (€)", f"{dv['toll_savings_eur']:,.0f}")

        df_cost = pd.DataFrame({
            "Scenario": ["Diesel baseline", "EV electricity"],
            "Annual cost (€)": [dv["diesel_cost_baseline_eur"], ec["annual_cost_eur"]]
        }).set_index("Scenario")
        st.bar_chart(df_cost)

        st.markdown("#### Toll (optional)")
        df_toll = pd.DataFrame({
            "Item": ["Baseline toll cost", "EV toll cost (assumption)"],
            "€ / year": [dv["baseline_toll_cost_eur"], 0.0 if inp["ev_toll_exempt"] else dv["baseline_toll_cost_eur"]]
        }).set_index("Item")
        st.bar_chart(df_toll)

    # =========================
    # TAB: CO2
    # =========================
    with tab_emissions:
        if co2_savings > 0:
            _recommendation(
                "success",
                "Emissions comparison: The EV scenario reduces CO₂ emissions by "
                f"{_fmt_kg(co2_savings)} per year, based on the assumed grid intensity and charging window."
            )
        else:
            _recommendation(
                "warning",
                "Emissions comparison: Under the current grid mix and charging window, EV charging does not reduce CO₂ emissions compared to diesel."
            )


        st.markdown("#### Emissions comparison")
        e1, e2, e3 = st.columns(3)
        with e1:
            st.metric("Effective grid CO₂ (kg/kWh)", f"{co2['effective_grid_co2_kg_per_kwh']:,.3f}")
        with e2:
            st.metric("EV CO₂ (kg / year)", f"{dv['ev_co2_kg']:,.0f}")
        with e3:
            st.metric("Diesel CO₂ (kg / year)", f"{dv['diesel_co2_baseline_kg']:,.0f}")

        df_co2 = pd.DataFrame({
            "Scenario": ["Diesel baseline", "EV (grid mix)"],
            "CO₂ (kg/year)": [dv["diesel_co2_baseline_kg"], dv["ev_co2_kg"]]
        }).set_index("Scenario")
        st.bar_chart(df_co2)

    # =========================
    # TAB: GRID / LOAD
    # =========================
    with tab_grid:
        if cap_ok:
            _recommendation(
                "success",
                "Grid assessment: The site connection capacity is sufficient to accommodate EV charging under the assumed peak conditions."
            )
        else:
            _recommendation(
                "error",
                "Grid assessment: The combined site load exceeds the connection capacity under peak charging assumptions."
            )

        st.markdown("#### Site load and constraints")

        g1, g2, g3, g4 = st.columns(4)
        with g1:
            st.metric("Existing site peak (kW)", f"{inp['existing_site_peak_kw']:,.0f}")
        with g2:
            st.metric("Charging peak add (kW)", f"{load['total_charge_power_kw']:,.0f}")
        with g3:
            st.metric("New theoretical peak (kW)", f"{load['new_theoretical_peak_kw']:,.0f}")
        with g4:
            st.metric("Site capacity limit (kVA)", f"{inp['site_capacity_limit_kva']:,.0f}")

        df_load = pd.DataFrame({
            "Value (kW/kVA)": [
                inp["existing_site_peak_kw"],
                load["new_theoretical_peak_kw"],
                inp["site_capacity_limit_kva"],
            ]
        }, index=["Existing peak (kW)", "New peak (kW)", "Capacity limit (kVA)"])
        st.bar_chart(df_load)

        if load["required_shaving_kw"] > 0:
            st.markdown("#### Peak shaving (optional)")
            st.write(
                f"- required shaving power: **{load['required_shaving_kw']:.0f} kW**\n"
                f"- required battery energy: **{load['required_battery_energy_kwh']:.1f} kWh**"
            )

    # =========================
    # TAB: HOURLY PROFILE (CHARTS)
    # =========================
    with tab_hourly:
        st.markdown("#### Hourly profile (visual)")

        df_hour = pd.DataFrame({
            "hour": list(range(24)),
            "charging_flag": prof["flags"],
            "share": prof["shares"],
            "grid_co2_g_per_kwh": prof["grid_co2_g_per_kwh"],
            "tou_price_eur_per_kwh": prof["tou_price_eur_per_kwh"],
        }).set_index("hour")

        best_price_hours = df_hour["tou_price_eur_per_kwh"].nsmallest(5).index.tolist()
        best_co2_hours   = df_hour["grid_co2_g_per_kwh"].nsmallest(5).index.tolist()

        covered_price = sum(int(df_hour.loc[h, "charging_flag"] == 1) for h in best_price_hours)
        covered_co2   = sum(int(df_hour.loc[h, "charging_flag"] == 1) for h in best_co2_hours)

        if dyn_share <= 0.0:
            _recommendation(
                "info",
                "Charging window analysis: With dynamic pricing disabled, time-of-use effects have limited impact on energy cost. "
                "The charging window mainly affects CO₂ intensity."
            )
        else:
            _recommendation(
                "info",
                "Charging window analysis: The current charging window overlaps with "
                f"{covered_price}/5 of the lowest-price hours."
            )

        _recommendation(
            "info",
            "Charging window CO₂ alignment: The current charging window overlaps with "
            f"{covered_co2}/5 of the lowest-carbon hours. "
            f"Lowest-price hours: {best_price_hours} | Lowest-CO₂ hours: {best_co2_hours}"    
            )


        st.markdown("**Charging share by hour**")
        st.bar_chart(df_hour[["share"]])

        st.markdown("**Grid CO₂ intensity (g/kWh)**")
        st.line_chart(df_hour[["grid_co2_g_per_kwh"]])

        st.markdown("**Time-of-use price (€/kWh)**")
        st.line_chart(df_hour[["tou_price_eur_per_kwh"]])

        with st.expander("Show hourly table"):
            st.dataframe(df_hour.reset_index(), use_container_width=True)

    # =========================
    # TAB: DETAILS (everything for nerd mode)
    # =========================
    with tab_details:
        _recommendation(
            "info",
            "Recommendation: Use the details to sanity-check assumptions (consumption, mileage, diesel L/100, electricity price, charging window). "
            "Note: this model mainly compares OPEX (energy/toll) — CAPEX/maintenance/residual value are not included yet."
        )


        with st.expander("Energy / price details"):
            st.json(ec)
        with st.expander("Distance details"):
            st.json(dist)
        with st.expander("Diesel vs EV details"):
            st.json(dv)
        with st.expander("Load details"):
            st.json(load)


st.markdown("---")


# -----------------------------
# Assistant UI (FleetMate)
# -----------------------------

if "assistant_messages" not in st.session_state:
    st.session_state["assistant_messages"] = [
        {"role": "assistant", "content": "Hi — I’m FleetMate. Ask me about the results, or tell me what to change (e.g., 'set num_trucks to 50')."}
    ]

if "last_gemini_payload" not in st.session_state:
    st.session_state["last_gemini_payload"] = None

if "assistant_is_open" not in st.session_state:
    st.session_state["assistant_is_open"] = False

st.markdown("""
<style>
#st-key-assistant_fab {
  position: fixed;
  right: 18px;
  bottom: 18px;
  z-index: 999999;
}
#st-key-assistant_fab button {
  border-radius: 999px;
  width: 56px;
  height: 56px;
  padding: 0;
  font-weight: 700;
  font-size: 22px;
}
</style>
""", unsafe_allow_html=True)

def assistant_dialog():
    # top bar inside modal
    h1, h2 = st.columns([0.85, 0.15])
    with h1:
        st.subheader("⚡ FleetMate")
        st.caption("I reply in English. I can also update inputs and re-run the model.")
    with h2:
        if st.button("✖", key="assistant_close"):
            st.session_state["assistant_is_open"] = False
            try:
                del st.query_params["fleetmate"]
            except Exception:
                pass
            st.rerun()

    # chat history
    for m in st.session_state["assistant_messages"]:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    user_msg = st.chat_input("Message FleetMate…")
    if user_msg:
        st.session_state["assistant_messages"].append({"role": "user", "content": user_msg})

        lowered = user_msg.lower()
        if ("payload" in lowered) or ("api json" in lowered) or ("json you send" in lowered) or ("request json" in lowered):
            payload = st.session_state.get("last_gemini_payload")
            reply = "Here is the exact JSON payload I last sent to Gemini:" if payload else "No payload stored yet (ask something first)."
            st.session_state["assistant_messages"].append({"role": "assistant", "content": reply})
            st.rerun()

        if st.session_state.get("model_results") is None:
            st.session_state["assistant_messages"].append({"role": "assistant", "content": "Run the calculation first so I have results to work with."})
            st.rerun()

        out = call_gemini_assistant(user_msg, st.session_state["model_results"])
        st.session_state["assistant_messages"].append({"role": "assistant", "content": out["reply"]})

        if out.get("update_inputs"):
            for k, v in out["update_inputs"].items():
                st.session_state[k] = v
            recalc_from_state()
            st.session_state["assistant_messages"].append({"role": "assistant", "content": "✅ Updated inputs and recalculated the results."})

        if out.get("show_payload"):
            p = st.session_state.get("last_gemini_payload")
            if p:
                st.session_state["assistant_messages"].append({"role": "assistant", "content": "Here is the JSON request payload I used."})

        st.rerun()

# Sticky floating action button (icon)
if st.button("💬", key="assistant_fab", help="Open FleetMate"):
    st.session_state["assistant_is_open"] = True
    st.rerun()

# Keep it open once clicked
if st.session_state["assistant_is_open"]:
    try:
        st.dialog("FleetMate")(assistant_dialog)()
    except Exception:
        # fallback if dialog not available
        with st.expander("FleetMate", expanded=True):
            assistant_dialog()


# If requested, render payload below (only when asked)
if st.session_state.get("last_gemini_payload") and any(
    m["content"].lower().startswith("here is the exact json payload")
    for m in st.session_state["assistant_messages"][-3:]
):
    st.markdown("---")
    st.subheader("Last Gemini Request Payload")
    st.json(st.session_state["last_gemini_payload"])




