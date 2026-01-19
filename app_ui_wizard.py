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

    st.markdown("## 🔒 " + t("login_title"))
    pw = st.text_input(t("password"), type="password")

    c1, c2 = st.columns([0.35, 0.65])
    with c1:
        if st.button(t("sign_in"), use_container_width=True):
            if expected and hmac.compare_digest(pw, expected):
                st.session_state["authed"] = True
                st.rerun()
            else:
                st.error(t("wrong_password"))
    with c2:
        st.caption(t("password_tip"))


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

TRUCK_TYPES = [
  {
    "id": "urban_16t",
    "name": {"DE": "Urban/Verteiler (12–16t)", "EN": "Urban distribution (12–16t)"},
    "subtitle": {"DE": "Stop&Go, kurze Distanzen", "EN": "Stop&go, short distances"},
    "cons_kwh_km": 0.95,
    "battery_kwh": 300,
    "max_charge_kw": 150,
    "img": "assets/trucks/rigid_16t.png",
  },
  {
    "id": "regional_26t",
    "name": {"DE": "Regional-Verteiler (18–26t)", "EN": "Regional distribution (18–26t)"},
    "subtitle": {"DE": "Gemischt, regional", "EN": "Mixed, regional"},
    "cons_kwh_km": 1.05,
    "battery_kwh": 400,
    "max_charge_kw": 200,
    "img": "assets/trucks/rigid_26t.png",
  },
  {
    "id": "tractor_regional",
    "name": {"DE": "Sattelzug Regional (~40t)", "EN": "Tractor-trailer regional (~40t)"},
    "subtitle": {"DE": "Gemischt, Autobahn + Stopps", "EN": "Mixed, motorway + stops"},
    "cons_kwh_km": 1.15,
    "battery_kwh": 600,
    "max_charge_kw": 350,
    "img": "assets/trucks/tractor_regional.png",
  },
  {
    "id": "tractor_longhaul_375",
    "name": {"DE": "Sattelzug Long-Haul (375mi Klasse)", "EN": "Tractor-trailer long-haul (375mi class)"},
    "subtitle": {"DE": "Konservativer Default", "EN": "Conservative default"},
    "cons_kwh_km": 1.30,
    "battery_kwh": 800,
    "max_charge_kw": 750,
    "img": "assets/trucks/tractor_longhaul.png",
  },
]



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
    avg_price_eur_per_mwh: float,          # dynamic ENERGY day-average (€/MWh)
    dynamic_share: float,
    shares,
    spot_curve_avg_eur_per_kwh: float | None = None,
    spot_window_avg_eur_per_kwh: float | None = None,
    fixed_price_eur_per_mwh: float | None = None,  # fixed ENERGY price (€/MWh)
    grid_eur_per_kwh: float = 0.0,
    levies_eur_per_kwh: float = 0.0,
    vat_percent: float = 0.0,
):
    # --- ENERGY prices in €/kWh ---
    dyn_energy_dayavg_eur_per_kwh = max(0.0, float(avg_price_eur_per_mwh)) / 1000.0
    dynamic_share = min(max(float(dynamic_share), 0.0), 1.0)

    fixed_energy_eur_per_kwh = (
        (max(0.0, float(fixed_price_eur_per_mwh)) / 1000.0)
        if fixed_price_eur_per_mwh is not None
        else dyn_energy_dayavg_eur_per_kwh
    )

    # --- Base curve for REL factor (fallback) ---
    base_curve = TOU_PRICE_EUR_PER_KWH
    curve_avg = float(sum(base_curve)) / 24.0
    window_avg = float(sum(p * s for p, s in zip(base_curve, shares)))
    rel = (window_avg / curve_avg) if curve_avg > 0 else 1.0

    used_spot = False
    if spot_curve_avg_eur_per_kwh is not None and spot_window_avg_eur_per_kwh is not None:
        curve_avg = float(max(1e-9, spot_curve_avg_eur_per_kwh))
        window_avg = float(spot_window_avg_eur_per_kwh)
        rel = window_avg / curve_avg
        used_spot = True

    # --- Window-adjust only the DYNAMIC ENERGY part ---
    dyn_energy_window_eur_per_kwh = dyn_energy_dayavg_eur_per_kwh * rel

    # --- Mix ENERGY ---
    mixed_energy_eur_per_kwh = (
        (1.0 - dynamic_share) * fixed_energy_eur_per_kwh
        + dynamic_share * dyn_energy_window_eur_per_kwh
    )

    # --- Add FIXED add-ons (do NOT vary by hour) + VAT ---
    net_allin_eur_per_kwh = mixed_energy_eur_per_kwh + float(grid_eur_per_kwh) + float(levies_eur_per_kwh)
    gross_allin_eur_per_kwh = net_allin_eur_per_kwh * (1.0 + float(vat_percent) / 100.0)

    eff_eur_per_mwh = gross_allin_eur_per_kwh * 1000.0

    # --- Build curves for charts (only ENERGY varies) ---
    if curve_avg > 0:
        scale = dyn_energy_dayavg_eur_per_kwh / curve_avg
        dyn_energy_curve = [float(p) * scale for p in base_curve]
    else:
        dyn_energy_curve = [dyn_energy_dayavg_eur_per_kwh] * 24

    mixed_energy_curve = [
        (1.0 - dynamic_share) * fixed_energy_eur_per_kwh + dynamic_share * dyn_energy_curve[h]
        for h in range(24)
    ]
    mixed_allin_curve = [
        (mixed_energy_curve[h] + float(grid_eur_per_kwh) + float(levies_eur_per_kwh)) * (1.0 + float(vat_percent) / 100.0)
        for h in range(24)
    ]

    details = {
        # ENERGY components
        "fixed_energy_eur_per_kwh": float(fixed_energy_eur_per_kwh),
        "dynamic_energy_dayavg_eur_per_kwh": float(dyn_energy_dayavg_eur_per_kwh),
        "dynamic_energy_window_eur_per_kwh": float(dyn_energy_window_eur_per_kwh),

        # Add-ons
        "grid_eur_per_kwh": float(grid_eur_per_kwh),
        "levies_eur_per_kwh": float(levies_eur_per_kwh),
        "vat_percent": float(vat_percent),

        # REL
        "curve_avg_eur_per_kwh": float(curve_avg),
        "window_avg_eur_per_kwh": float(window_avg),
        "rel_factor": float(rel),
        "dynamic_share": float(dynamic_share),
        "used_spot_data": bool(used_spot),

        # All-in results
        "effective_allin_eur_per_kwh": float(gross_allin_eur_per_kwh),
        "mixed_curve_allin_eur_per_kwh": mixed_allin_curve,
    }
    return eff_eur_per_mwh, details



def run_model(
    num_trucks: int,
    operating_days: int,
    km_per_truck_per_day: float,
    battery_kwh: float,
    charge_loss_pct: float,
    avg_elec_price_eur_per_mwh: float,
    dynamic_price_share: float,
    start_hour: int,
    end_hour: int,
    charging_window_hours: float,
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
    num_chargers: int,
    power_factor: float,
    existing_site_peak_kw: float,          # Peak im Ladefenster (aus Profil)
    existing_site_avg_kw_in_window: float, # Baseline im Ladefenster (neu)
    baseline_site_peak_kw_overall: float,  # Peak gesamt für Leistungsentgelt (neu)
    net_demand_eur_per_kw_year: float,     # €/kW/Jahr (neu)
    spot_curve_avg_eur_per_kwh: float | None = None,
    spot_window_avg_eur_per_kwh: float | None = None,
    fixed_price_eur_per_mwh: float | None = None,   # FIXED ENERGY €/MWh
    fixed_monthly_eur: float = 0.0,
    grid_eur_per_kwh: float = 0.0,
    levies_eur_per_kwh: float = 0.0,
    vat_percent: float = 0.0,
):

    """
    Core business-case model.

    NOTE:
    - We keep start_soc/target_soc in the model for backward compatibility.
      In the UI, they are derived from km/day (not user inputs anymore).
    - We keep dynamic_price_share internally (0=fixed, 1=dynamic) for compatibility.
      In the UI, the user chooses a price mode instead of entering a share.
    """
    start_h = int(start_hour) % 24
    end_h = int(end_hour) % 24
    flags, shares = compute_flags_and_shares(start_h, end_h)

    num_trucks = max(1, int(num_trucks))
    operating_days = max(1, int(operating_days))

    km_per_truck_per_day = max(0.0, float(km_per_truck_per_day))
    battery_kwh = max(0.0, float(battery_kwh))
    ev_consumption_kwh_per_km = max(0.01, float(ev_consumption_kwh_per_km))

    charging_window_hours = max(0.1, float(charging_window_hours))
    dynamic_price_share = min(max(float(dynamic_price_share), 0.0), 1.0)
    tolled_share_0_1 = min(max(float(tolled_share_0_1), 0.0), 1.0)
    peak_duration_h = max(0.0, float(peak_duration_h))

    # Charge efficiency
    charge_loss_pct = min(max(float(charge_loss_pct), 0.0), 40.0)
    eta_charge = 1.0 - (charge_loss_pct / 100.0)
    eta_charge = max(0.60, eta_charge)

    # Energy per truck/day
    batt_kwh_truck_day = km_per_truck_per_day * ev_consumption_kwh_per_km
    grid_kwh_truck_day = batt_kwh_truck_day / eta_charge

    # Fleet energy (GRID!)
    total_daily_energy_mwh = (num_trucks * grid_kwh_truck_day) / 1000.0
    annual_energy_mwh = total_daily_energy_mwh * operating_days

    eff_price_eur_per_mwh, price_details = effective_energy_price_details(
        avg_elec_price_eur_per_mwh,
        dynamic_price_share,
        shares,
        spot_curve_avg_eur_per_kwh=spot_curve_avg_eur_per_kwh,
        spot_window_avg_eur_per_kwh=spot_window_avg_eur_per_kwh,
        fixed_price_eur_per_mwh=fixed_price_eur_per_mwh,
        grid_eur_per_kwh=grid_eur_per_kwh,
        levies_eur_per_kwh=levies_eur_per_kwh,
        vat_percent=vat_percent,
    )


    annual_variable_cost_eur = annual_energy_mwh * eff_price_eur_per_mwh

    # CO2
    eff_grid_co2_kg_per_kwh, co2_details = effective_grid_co2_details(shares)
    annual_energy_kwh = annual_energy_mwh * 1000.0
    annual_ev_co2_kg = annual_energy_kwh * eff_grid_co2_kg_per_kwh


    # Distance (derived from energy + consumption)
    daily_batt_kwh = num_trucks * batt_kwh_truck_day
    km_per_truck_per_day = (daily_batt_kwh / num_trucks) / ev_consumption_kwh_per_km
    annual_km_per_truck = km_per_truck_per_day * operating_days
    annual_km_fleet = annual_km_per_truck * num_trucks

    # Diesel baseline
    diesel_litres_baseline = (annual_km_fleet * diesel_l_per_100km) / 100.0
    diesel_cost_baseline = diesel_litres_baseline * diesel_price_eur_per_l
    diesel_co2_baseline_kg = diesel_litres_baseline * DIESEL_CO2_PER_L


    # Toll
    baseline_toll_cost = annual_km_fleet * tolled_share_0_1 * toll_rate_eur_per_km
    ev_toll_cost = 0.0 if ev_toll_exempt else baseline_toll_cost
    toll_savings = baseline_toll_cost - ev_toll_cost

    # kVA -> kW für physische Grenze
    site_capacity_kw = kva_to_kw(site_capacity_limit_kva, power_factor)

    num_chargers = max(0, int(num_chargers))
    simultaneous_trucks = min(num_trucks, num_chargers) if num_chargers > 0 else num_trucks

    total_charge_power_kw = simultaneous_trucks * charger_power_per_truck_kw

    # Worst-case Peak (für physische Machbarkeit)
    new_theoretical_peak_kw = existing_site_peak_kw + total_charge_power_kw

    # Durchschnittslast (für “wie viel Energie passt ins Fenster”)
    daily_grid_kwh_fleet = total_daily_energy_mwh * 1000.0
    avg_charging_power_kw = daily_grid_kwh_fleet / charging_window_hours
    new_avg_load_kw = existing_site_avg_kw_in_window + avg_charging_power_kw

    capacity_ok = (new_theoretical_peak_kw <= site_capacity_kw) if site_capacity_kw > 0 else True


    # Optional peak shaving
    desired_peak_limit_kw = max(0.0, float(desired_peak_limit_kw))
    required_shaving_kw = max(0.0, new_theoretical_peak_kw - desired_peak_limit_kw) if desired_peak_limit_kw > 0 else 0.0
    required_battery_energy_kwh = required_shaving_kw * peak_duration_h if peak_duration_h > 0 else 0.0


    # Leistungsentgelt-Mehrkosten (Peak > bisheriger Abrechnungs-Peak)
    # EV-Peak passiert im Ladefenster => Candidate peak = existing_in_window_peak + ev_peak_power
    candidate_peak_for_billing = existing_site_peak_kw + total_charge_power_kw
    new_billing_peak_kw = max(float(baseline_site_peak_kw_overall), float(candidate_peak_for_billing))

    delta_peak_kw = max(0.0, new_billing_peak_kw - float(baseline_site_peak_kw_overall))
    extra_demand_charge_eur_year = delta_peak_kw * max(0.0, float(net_demand_eur_per_kw_year))

    annual_fixed_cost_eur = float(max(0.0, fixed_monthly_eur)) * 12.0
    annual_fixed_cost_eur += extra_demand_charge_eur_year
    annual_cost_eur = annual_variable_cost_eur + annual_fixed_cost_eur

    # EV scenario
    ev_cost = annual_cost_eur
    ev_co2_kg = annual_ev_co2_kg

    cost_savings_eur = diesel_cost_baseline - ev_cost
    co2_savings_kg = diesel_co2_baseline_kg - ev_co2_kg

    total_savings_incl_toll = cost_savings_eur + toll_savings

    return {
        "inputs": {
            "num_trucks": num_trucks,
            "operating_days": operating_days,
            "battery_kwh": battery_kwh,
            "km_per_truck_per_day": km_per_truck_per_day,
            "charge_loss_pct": charge_loss_pct,
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
            "fixed_price_eur_per_mwh": fixed_price_eur_per_mwh,
            "fixed_monthly_eur": fixed_monthly_eur,
            "grid_eur_per_kwh": grid_eur_per_kwh,
            "levies_eur_per_kwh": levies_eur_per_kwh,
            "vat_percent": vat_percent,
        },
        "charging_profile": {
            "flags": flags,
            "shares": shares,
            "grid_co2_g_per_kwh": GRID_CO2_G_PER_KWH,
            "tou_price_eur_per_kwh": price_details.get("mixed_curve_allin_eur_per_kwh", TOU_PRICE_EUR_PER_KWH),
        },
        "energy_cost": {
            "charge_loss_pct": charge_loss_pct,
            "eta_charge": eta_charge,
            "battery_energy_kwh_per_truck_day": batt_kwh_truck_day,
            "grid_energy_kwh_per_truck_day": grid_kwh_truck_day,
            "total_daily_energy_mwh": total_daily_energy_mwh,  # GRID
            "annual_energy_mwh": annual_energy_mwh,            # GRID
            "effective_price_eur_per_mwh": eff_price_eur_per_mwh,
            "annual_cost_eur": annual_cost_eur,
            "price_details": price_details,
            "annual_variable_cost_eur": annual_variable_cost_eur,
            "annual_fixed_cost_eur": annual_fixed_cost_eur,
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
            "site_capacity_kw": site_capacity_kw,
            "simultaneous_trucks": simultaneous_trucks,
            "extra_demand_charge_eur_year": extra_demand_charge_eur_year,
            "delta_peak_kw_for_billing": delta_peak_kw,
            "new_billing_peak_kw": new_billing_peak_kw,
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
    operating_days=300,

    # Usage (core)
    km_per_truck_per_day=220.0,

    # Truck type selection (cards)
    truck_type_id=TRUCK_TYPES[0]["id"],

    # Battery / consumption (can be overridden)
    battery_kwh=500.0,
    ev_consumption=1.0,

    # Charging losses (net -> battery)
    charge_loss_pct=15,  # default 15%, adjustable in "Advanced"

    # Electricity
    price_mode="fixed",           # "fixed" | "dynamic"
    fixed_elec_price_mwh=86.79,
    dynamic_price_region="DE",

    # Internals (computed):
    avg_elec_price_mwh=86.79,
    dynamic_share=0.0,

    # Charging window
    start_hour=22,
    end_hour=6,
    charging_window_hours=8.0,

    # Site / capacity
    existing_peak_kw=3000.0,
    charger_power_kw=150.0,
    site_capacity_kva=4000.0,

    # Optional peak limit
    desired_peak_limit_kw=0.0,
    peak_duration_h=0.25,

    # Diesel baseline + toll
    market_region="AT",
    diesel_price=1.55,
    diesel_l_per_100=22.0,
    toll_rate=0.456,
    tolled_share=0.60,
    ev_toll_exempt=True,

    power_factor=0.90,     # cos φ
    num_chargers=10,       # Anzahl Ladepunkte / Säulen-Ports
    # Demand charge (Leistungsentgelt) separat:
    net_demand_eur_per_kw_year=0.0,
)


# Gemini may only update these (avoid letting the LLM switch profiles/regions or hidden internals)
HIDDEN_OR_CONTROLLED = {
    "market_region",
    "avg_elec_price_mwh", "dynamic_share",
    "charging_window_hours",
}
ALLOWED_INPUT_KEYS = set(DEFAULT_INPUTS.keys()) - HIDDEN_OR_CONTROLLED


def ensure_defaults():
    st.session_state.setdefault("intro_done", False)

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

    # window as HH:00 strings (synced from/to start_hour/end_hour)
    st.session_state.setdefault("charge_window_start", f"{int(get_inp('start_hour'))%24:02d}:00")
    st.session_state.setdefault("charge_window_end", f"{int(get_inp('end_hour'))%24:02d}:00")

    # ---- New Strompreis state (netto) ----
    st.session_state.setdefault("pricing_country", "AT")
    st.session_state.setdefault("pricing_country_prev", "AT")

    st.session_state.setdefault("fixed_energy_eur_per_kwh", eur_mwh_to_eur_kwh(86.79))
    st.session_state.setdefault("dyn_energy_eur_per_kwh",   eur_mwh_to_eur_kwh(86.79))
    st.session_state.setdefault("split_dynamic_share_pct", 30)

    st.session_state.setdefault("net_tariff_key", NETZ_DEFAULT_KEY_BY_COUNTRY["AT"])
    st.session_state.setdefault("net_fee_eur_per_kwh", 0.0)

    st.session_state.setdefault("power_tax_eur_per_kwh", ct_kwh_to_eur_kwh(TAX_DEFAULT_CT_BY_COUNTRY["AT"]))

    # netto erzwingen (kompatibel zum bestehenden Modell)
    st.session_state.setdefault("vat_percent", 0.0)
    st.session_state.setdefault("fixed_monthly_eur", 0.0)

    st.session_state.setdefault("net_fee_is_manual", False)
    st.session_state.setdefault("net_tariff_key_prev", st.session_state.get("net_tariff_key"))




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
def derive_energy_from_km(inputs: dict) -> dict:
    km_day = float(max(0.0, inputs.get("km_per_truck_per_day", 0.0)))
    cons = float(max(0.01, inputs.get("ev_consumption", 0.01)))

    loss_pct = float(inputs.get("charge_loss_pct", 15))
    loss_pct = min(max(loss_pct, 0.0), 40.0)
    eta_charge = 1.0 - (loss_pct / 100.0)
    eta_charge = max(0.60, eta_charge)

    e_batt_day = km_day * cons
    e_grid_day = e_batt_day / eta_charge

    battery_kwh = float(max(0.0, inputs.get("battery_kwh", 0.0)))

    # --- NEW: residual charge on arrival (assumption: start day at 100% SoC) ---
    residual_soc = None
    residual_kwh = None
    if battery_kwh > 0:
        residual_kwh = max(0.0, battery_kwh - e_batt_day)
        residual_soc = max(0.0, 1.0 - (e_batt_day / battery_kwh))

    warn = None
    if battery_kwh > 0 and e_batt_day > battery_kwh * 0.98:
        warn = "km/Tag liegen (nahezu) auf/über Batteriekapazität. Range/Verbrauch/Batterie prüfen oder Zwischenladen annehmen."

    return {
        "km_per_truck_per_day": km_day,
        "cons_kwh_per_km": cons,
        "charge_loss_pct": loss_pct,
        "eta_charge": eta_charge,  # keep for grid-energy calc (do NOT label as residual)
        "battery_energy_kwh_per_truck_day": e_batt_day,
        "grid_energy_kwh_per_truck_day": e_grid_day,
        "residual_soc_arrival": residual_soc,   # NEW (0..1)
        "residual_kwh_arrival": residual_kwh,   # NEW
        "warning": warn,
    }


def apply_pricing_split_from_state(inputs: dict) -> tuple[dict, dict]:
    def _hh_to_h(v: str, fallback: int) -> int:
        try:
            return int(str(v).split(":")[0]) % 24
        except Exception:
            return int(fallback) % 24

    start_h = _hh_to_h(st.session_state.get("charge_window_start"), int(inputs.get("start_hour", 22)))
    end_h   = _hh_to_h(st.session_state.get("charge_window_end"), int(inputs.get("end_hour", 6)))
    inputs["start_hour"] = start_h
    inputs["end_hour"] = end_h

    p_dyn = clamp01(float(st.session_state.get("split_dynamic_share_pct", 50)) / 100.0)

    fixed_energy = float(max(0.0, st.session_state.get("fixed_energy_eur_per_kwh", 0.0)))
    dyn_energy   = float(max(0.0, st.session_state.get("dyn_energy_eur_per_kwh", 0.0)))

    netz = float(max(0.0, st.session_state.get("net_fee_eur_per_kwh", 0.0)))
    tax  = float(max(0.0, st.session_state.get("power_tax_eur_per_kwh", 0.0)))

    inputs["net_demand_eur_per_kw_year"] = float(st.session_state.get("net_power_eur_kw_year", 0.0))

    # Feed model with ENERGY prices (€/MWh); add-ons separately (alles netto)
    inputs["fixed_price_eur_per_mwh"] = float(eur_kwh_to_eur_mwh(fixed_energy))
    inputs["avg_elec_price_mwh"]      = float(eur_kwh_to_eur_mwh(dyn_energy))
    inputs["dynamic_share"]           = float(p_dyn)

    # Mapping: Netz und Steuer/Abgabe als konstante €/kWh Add-ons
    inputs["grid_eur_per_kwh"]   = float(netz)
    inputs["levies_eur_per_kwh"] = float(tax)

    # explizit netto
    inputs["vat_percent"] = 0.0
    inputs["fixed_monthly_eur"] = 0.0

    pricing_meta = {
        "pricing_country": st.session_state.get("pricing_country"),
        "split_dynamic_share": float(p_dyn),
        "fixed_energy_eur_per_kwh": float(fixed_energy),
        "dynamic_energy_dayavg_eur_per_kwh": float(dyn_energy),
        "net_fee_eur_per_kwh": float(netz),
        "tax_eur_per_kwh": float(tax),
        "vat_percent": 0.0,
    }
    return inputs, pricing_meta




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


def _window_day(ts: pd.Timestamp, start_h: int, end_h: int) -> datetime.date:
    # Wenn Fenster über Mitternacht geht, zählen 00:00–end_h zur VORHERIGEN “Fenster-Nacht”
    d = ts.date()
    if start_h > end_h and ts.hour < end_h:
        d = (ts - pd.Timedelta(days=1)).date()
    return d

def compute_profile_metrics(
    df_raw: pd.DataFrame,
    timestamp_col: str,
    consumption_col: str,
    start_h: int,
    end_h: int,
    site_capacity_kw: float | None = None,
) -> dict:
    df = df_raw.copy()

    ts = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.loc[ts.notna()].copy()
    df["ts"] = ts.loc[ts.notna()].dt.tz_localize(None)

    df["site_kw"] = pd.to_numeric(df[consumption_col], errors="coerce")
    df = df.loc[df["site_kw"].notna()].copy()
    df = df.sort_values("ts")

    df["hour"] = df["ts"].dt.hour
    df["in_window"] = df["hour"].apply(lambda h: _in_window_hours(int(h), start_h, end_h))

    # dt (hours) robust für 1h / 15min / 1min / unregelmäßig
    dts = df["ts"].shift(-1) - df["ts"]
    dt_h = dts.dt.total_seconds() / 3600.0
    dt_pos = dt_h[(dt_h > 0) & (dt_h < 24)]
    dt_fill = float(dt_pos.median()) if len(dt_pos) else 1.0
    df["dt_h"] = dt_h.fillna(dt_fill).clip(lower=0.0, upper=24.0)

        # kWh je Zeile (robust bei 15min/1h/unregelmäßig)
    df["kwh"] = df["site_kw"] * df["dt_h"]

    # Overall peak (für Leistungsentgelt)
    peak_kw_overall = float(df["site_kw"].max()) if not df.empty else 0.0
    energy_kwh_total = float(df["kwh"].sum()) if not df.empty else 0.0

    # Window-only
    inw = df.loc[df["in_window"]].copy()
    if inw.empty:
        inw = df.copy()

    inw["window_day"] = inw["ts"].apply(lambda x: _window_day(x, start_h, end_h))

    # Tageswerte im Ladefenster: Ø kW, Peak kW, kWh
    daily = inw.groupby("window_day").agg(
        avg_kw_in_window=("site_kw", "mean"),
        peak_kw_in_window=("site_kw", "max"),
        energy_kwh_in_window=("kwh", "sum"),
        samples=("site_kw", "count"),
    ).reset_index()

    avg_kw_in_window = float(daily["avg_kw_in_window"].mean()) if not daily.empty else float(inw["site_kw"].mean())
    peak_kw_in_window = float(daily["peak_kw_in_window"].max()) if not daily.empty else float(inw["site_kw"].max())
    p95_kw_in_window = float(daily["peak_kw_in_window"].quantile(0.95)) if len(daily) >= 5 else peak_kw_in_window

    # kWh pro Window-Day (dein "Verbrauch pro Tag im Ladefenster")
    energy_kwh_in_window_avg_per_day = float(daily["energy_kwh_in_window"].mean()) if not daily.empty else float(inw["kwh"].sum())
    energy_kwh_in_window_min_per_day = float(daily["energy_kwh_in_window"].min()) if not daily.empty else energy_kwh_in_window_avg_per_day
    energy_kwh_in_window_max_per_day = float(daily["energy_kwh_in_window"].max()) if not daily.empty else energy_kwh_in_window_avg_per_day
    energy_kwh_in_window_total = float(inw["kwh"].sum()) if not inw.empty else 0.0

    # Hourly curves (Ø/Max kW über alle Daten)
    hourly_avg = df.groupby("hour")["site_kw"].mean().reindex(range(24)).fillna(0.0)
    hourly_max = df.groupby("hour")["site_kw"].max().reindex(range(24)).fillna(0.0)

    # Verbrauch pro Stunde im Ladefenster: Ø kWh pro (Window-Day, Stunde)
    # -> gruppiert nach window_day & hour, dann Mittel über Tage
    inw_hour = inw.groupby(["window_day", "hour"])["kwh"].sum().reset_index()
    hourly_kwh_in_window_avg = (
        inw_hour.groupby("hour")["kwh"].mean().reindex(range(24)).fillna(0.0)
        if not inw_hour.empty else pd.Series([0.0]*24, index=range(24))
    )

    # Sampling-Info
    sampling_minutes_median = float(dt_fill * 60.0)

    # Headroom-Energie (wie viel kWh passt zusätzlich ins Fenster) — optional
    energy_headroom_kwh_avg_per_day = None
    energy_headroom_kwh_min_per_day = None
    if site_capacity_kw is not None and site_capacity_kw > 0:
        inw["headroom_kw"] = (float(site_capacity_kw) - inw["site_kw"]).clip(lower=0.0)
        inw["headroom_kwh"] = inw["headroom_kw"] * inw["dt_h"]
        daily_headroom = inw.groupby("window_day")["headroom_kwh"].sum()
        if len(daily_headroom):
            energy_headroom_kwh_avg_per_day = float(daily_headroom.mean())
            energy_headroom_kwh_min_per_day = float(daily_headroom.min())

    return {
        "start_ts": str(df["ts"].iloc[0]) if not df.empty else None,
        "end_ts": str(df["ts"].iloc[-1]) if not df.empty else None,
        "records": int(len(df)),
        "window_days": int(daily["window_day"].nunique()) if not daily.empty else 0,
        "sampling_minutes_median": sampling_minutes_median,

        "peak_kw_overall": peak_kw_overall,
        "energy_kwh_total": energy_kwh_total,

        "avg_kw_in_window": avg_kw_in_window,
        "p95_kw_in_window": p95_kw_in_window,
        "peak_kw_in_window": peak_kw_in_window,

        "energy_kwh_in_window_avg_per_day": energy_kwh_in_window_avg_per_day,
        "energy_kwh_in_window_min_per_day": energy_kwh_in_window_min_per_day,
        "energy_kwh_in_window_max_per_day": energy_kwh_in_window_max_per_day,
        "energy_kwh_in_window_total": energy_kwh_in_window_total,

        "hourly_avg_kw": [float(x) for x in hourly_avg.values],
        "hourly_max_kw": [float(x) for x in hourly_max_kw.values] if "hourly_max_kw" in locals() else [float(x) for x in hourly_max.values],
        "hourly_kwh_in_window_avg": [float(x) for x in hourly_kwh_in_window_avg.values],

        "daily_window_records": daily.to_dict("records"),

        "energy_headroom_kwh_avg_per_window_day": energy_headroom_kwh_avg_per_day,
        "energy_headroom_kwh_min_per_window_day": energy_headroom_kwh_min_per_day,
    }




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
    prof_metrics = st.session_state.get("profile_cache", {}).get("metrics") or {}
    has_profile = bool(prof_metrics)

    # UI für existing_peak_kw soll genau EINMAL im Wizard/Sidebar stehen.
    # Hier nur lesen und ggf. durch Profil überschreiben.

    if prof_metrics and prof_metrics.get("peak_kw_in_window") is not None:
        site_peak_source = "load_profile"
        existing_peak_kw_in_window = float(max(0.0, prof_metrics.get("peak_kw_in_window", 0.0)))
        existing_avg_kw_in_window  = float(prof_metrics.get("avg_kw_in_window", existing_peak_kw_in_window))
        baseline_peak_overall_kw   = float(prof_metrics.get("peak_kw_overall", existing_peak_kw_in_window))
    else:
        site_peak_source = "manual"
        existing_peak_kw_in_window = float(max(0.0, inp.get("existing_peak_kw", 0.0)))
        existing_avg_kw_in_window  = float(max(0.0, inp.get("existing_peak_kw", 0.0)))
        baseline_peak_overall_kw   = float(max(0.0, inp.get("existing_peak_kw", 0.0)))



    # ---- Derive SoC from km/day (no manual SoC inputs anymore) ----
    derived = derive_energy_from_km(inp)

    # ---- Pricing: fixed + dynamic split (always on) ----
    inp, pricing_meta = apply_pricing_split_from_state(inp)

    # ---- Existing site peak: prefer upload peak in charging window (if present) ----
    existing_peak_kw = float(inp.get("existing_peak_kw", 0.0))
    site_peak_source = "manual"
    if prof_metrics and prof_metrics.get("peak_kw_in_window") is not None:
        existing_peak_kw = float(max(0.0, prof_metrics["peak_kw_in_window"]))
        site_peak_source = "load_profile"
        existing_peak_kw_in_window = float(prof_metrics.get("peak_kw_in_window", 0.0))
        existing_avg_kw_in_window  = float(prof_metrics.get("avg_kw_in_window", existing_peak_kw_in_window))
        baseline_peak_overall_kw   = float(prof_metrics.get("peak_kw_overall", existing_peak_kw_in_window))
    else:
        existing_peak_kw_in_window = float(inp.get("existing_peak_kw", 0.0))
        existing_avg_kw_in_window  = float(inp.get("existing_peak_kw", 0.0))
        baseline_peak_overall_kw   = float(inp.get("existing_peak_kw", 0.0))

    # Spot stats are intentionally disabled (dynamic pricing uses fixed averages + TOU curve)
    spot_curve_avg = None
    spot_window_avg = None


    inp["charging_window_hours"] = float(window_len_hours(int(inp["start_hour"]), int(inp["end_hour"])))

    # ---- Run model ----
    res = run_model(
        num_trucks=inp["num_trucks"],
        operating_days=inp["operating_days"],
        km_per_truck_per_day=inp["km_per_truck_per_day"],
        battery_kwh=inp["battery_kwh"],
        charge_loss_pct=inp["charge_loss_pct"],
        avg_elec_price_eur_per_mwh=inp["avg_elec_price_mwh"],
        dynamic_price_share=inp["dynamic_share"],
        start_hour=inp["start_hour"],
        end_hour=inp["end_hour"],
        charging_window_hours=inp["charging_window_hours"],
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
        spot_curve_avg_eur_per_kwh=None,
        spot_window_avg_eur_per_kwh=None,
        fixed_price_eur_per_mwh=inp.get("fixed_price_eur_per_mwh"),
        fixed_monthly_eur=inp.get("fixed_monthly_eur", 0.0),
        grid_eur_per_kwh=inp.get("grid_eur_per_kwh", 0.0),
        levies_eur_per_kwh=inp.get("levies_eur_per_kwh", 0.0),
        vat_percent=inp.get("vat_percent", 0.0),
        num_chargers=inp["num_chargers"],
        power_factor=inp["power_factor"],
        existing_site_peak_kw=existing_peak_kw_in_window,
        existing_site_avg_kw_in_window=existing_avg_kw_in_window,
        baseline_site_peak_kw_overall=baseline_peak_overall_kw,
        net_demand_eur_per_kw_year=inp.get("net_demand_eur_per_kw_year", 0.0),
    )

    res["energy_cost"].setdefault("price_details", {})
    res["energy_cost"]["price_details"]["pricing_meta"] = pricing_meta


    # ---- Attach profile metrics (serialisable) ----
    res.setdefault("profile", {})
    res["profile"].update({
        "site_peak_source": site_peak_source,
        "derived_energy": {
            "km_per_truck_per_day": float(inp.get("km_per_truck_per_day", 0.0)),
            "battery_energy_kwh_per_truck_day": float(derived["battery_energy_kwh_per_truck_day"]),
            "grid_energy_kwh_per_truck_day": float(derived["grid_energy_kwh_per_truck_day"]),
            "charge_loss_pct": float(derived["charge_loss_pct"]),
            "eta_charge": float(derived["eta_charge"]),
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
    cap_kw = kva_to_kw(float(inp.get("site_capacity_kva", 0.0)), float(inp.get("power_factor", 0.90)))

    peak_in_window = float(existing_peak_kw_in_window)  # Peak im Ladefenster aus Profil


    # Energy needed per truck per day (from km/day & consumption)
    e_truck_day = float(derived["grid_energy_kwh_per_truck_day"])

    wh = window_len_hours(int(inp["start_hour"]), int(inp["end_hour"]))

   # --- Headroom-Definitionen (klar getrennt) ---
    headroom_peak_kw = max(0.0, cap_kw - existing_peak_kw_in_window)
    headroom_baseline_kw = max(0.0, cap_kw - existing_avg_kw_in_window)


    energy_avail_kwh = None
    if prof_metrics and prof_metrics.get("energy_headroom_kwh_avg_per_window_day") is not None:
        energy_avail_kwh = float(prof_metrics["energy_headroom_kwh_avg_per_window_day"])
    else:
        energy_avail_kwh = headroom_baseline_kw * float(wh)

    max_trucks_energy_based = int(np.floor(energy_avail_kwh / e_truck_day)) if e_truck_day > 1e-9 else 0



    # "Naive" simultaneous charging at full power
    charger_kw = float(max(0.01, inp.get("charger_power_kw", 0.01)))
    max_trucks_simultaneous = int(np.floor(headroom_peak_kw / charger_kw)) if headroom_peak_kw > 0 else 0

    # Recommended average charging power per truck to fit into effective charging time
    eff_h = float(max(0.1, inp.get("charging_window_hours", 0.1)))
    recommended_kw_per_truck = (e_truck_day / eff_h) if eff_h > 0 else None

    res.setdefault("profile", {})
    res["profile"].update({
        "site_peak_source": site_peak_source,
        "derived_energy": {
            **derived
        }
    })
    if derived.get("warning"):
        res["profile"]["derived_energy"]["warning"] = derived["warning"]

    res.setdefault("capacity_analysis", {})
    res["capacity_analysis"].update({
        "charging_window_hours_total": int(wh),
        "energy_available_kwh_in_window": energy_avail_kwh,
        "energy_needed_kwh_per_truck_day": e_truck_day,
        "max_trucks_energy_based": max_trucks_energy_based,
        "max_trucks_simultaneous_at_full_power": max_trucks_simultaneous,
        "recommended_avg_kw_per_truck": float(recommended_kw_per_truck) if recommended_kw_per_truck is not None else None,
        "site_capacity_kw": cap_kw,
        "site_peak_kw_in_window": peak_in_window,
        "available_kw_at_peak": cap_kw - peak_in_window,
        "headroom_peak_kw": headroom_peak_kw,
        "headroom_baseline_kw": headroom_baseline_kw,
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
        "login_title": "FleetMate Login",
        "password": "Password",
        "sign_in": "Sign in",
        "wrong_password": "Wrong password.",
        "password_tip": "Tip: The password is the partner company name in lowercase. (7 letters – Austria)",

        "peak_in_window": "Peak in charging window",
        "headroom": "Headroom",
        "headroom_at_peak": "Headroom (at peak)",

        "kpi_ev_cost_year": "EV cost / year",
        "kpi_peak_window": "Peak in window",
        "kpi_headroom_peak": "Headroom (peak)",
        "kpi_max_trucks_energy": "Max trucks (energy)",
        "kpi_rec_avg_kw_truck": "Rec. avg kW/truck",

        "analysis_title": "Analysis & recommendation",
        "constraints_title": "Constraints & recommendations",
        "no_constraints": "No major constraints detected.",
        "next_steps_title": "Best next steps",
        "best_next_actions": "Best next actions",

        "reverse_calc_title": "Max. Trucks with Peak",      
        "reverse_calc_caption": (
            "We start from the **available site capacity** "
            "(site connection limit minus the measured peak load within the charging window). "
            "This capacity is then translated into the **maximum number of trucks** using two views: "
            "**energy-based feasibility** and a **simultaneous charging worst-case**."
        ),

        "toll_assumptions": "Toll assumptions",
        "apply_estimate": "Apply estimate",

        "interpretation": "Interpretation",

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
        "login_title": "FleetMate Login",
        "password": "Passwort",
        "sign_in": "Anmelden",
        "wrong_password": "Falsches Passwort.",
        "password_tip": "Tipp: Das Passwort ist die Partner-Firma klein geschrieben. (7 Buchstaben – Österreich)",

        "peak_in_window": "Peak im Ladefenster",
        "headroom": "Headroom",
        "headroom_at_peak": "Headroom (am Peak)",

        "kpi_ev_cost_year": "EV-Kosten / Jahr",
        "kpi_peak_window": "Peak im Ladefenster",
        "kpi_headroom_peak": "Headroom (Peak)",
        "kpi_max_trucks_energy": "Max. LKW (Energie)",
        "kpi_rec_avg_kw_truck": "Empf. Ø kW/LKW",

        "analysis_title": "Analyse & Empfehlung",
        "constraints_title": "Constraints & Empfehlungen",
        "no_constraints": "Keine großen Constraints erkannt.",
        "next_steps_title": "Nächste sinnvolle Schritte",
        "best_next_actions": "Best next actions",

        "reverse_calc_title": "Max. LKW mit Lastspitze",
        "reverse_calc_caption": (
            "Ausgangspunkt ist die **verfügbare Leistungsreserve am Standort** "
            "(Anschlusslimit abzüglich der gemessenen Peak-Last im Ladefenster). "
            "Diese Reserve wird anschließend in die **maximal ladbare Anzahl an LKW** übersetzt – "
            "einmal **energie-basiert** und einmal als **konservativer Simultan-Worst-Case**."
        ),

        "toll_assumptions": "Maut-Annahmen",
        "apply_estimate": "Schätzung übernehmen",

        "interpretation": "Interpretation",

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
            "km_per_truck_per_day_input": prof.get("derived_energy", {}).get("km_per_truck_per_day", None),
            "charge_loss_pct": prof.get("derived_energy", {}).get("charge_loss_pct", None),
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
        "Keep the reply concise (max ~600 characters) so the JSON never gets truncated.\n"
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
        "generationConfig": {"temperature": 0.25, "maxOutputTokens": 700},
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
        # salvage: show only the reply-ish part, not the whole JSON garbage
        return {"reply": normalize_assistant_text(text), "update_inputs": None, "show_payload": False}

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

    st.session_state["intro_done"] = True
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
    # Pricing split UI reset
    st.session_state["fixed_energy_eur_per_mwh"] = 86.79
    st.session_state["dyn_spot_avg_eur_per_mwh"] = 80.0
    st.session_state["dyn_markup_ct_per_kwh"] = 0.0
    st.session_state["charge_window_start"] = "22:00"
    st.session_state["charge_window_end"] = "06:00"
    st.session_state["pricing_country"] = "AT"
    st.session_state["pricing_country_prev"] = "AT"
    st.session_state["fixed_energy_eur_per_kwh"] = eur_mwh_to_eur_kwh(86.79)
    st.session_state["dyn_energy_eur_per_kwh"] = eur_mwh_to_eur_kwh(86.79)
    st.session_state["split_dynamic_share_pct"] = 30
    st.session_state["net_tariff_key"] = NETZ_DEFAULT_KEY_BY_COUNTRY["AT"]
    st.session_state["net_fee_eur_per_kwh"] = 0.0
    st.session_state["power_tax_eur_per_kwh"] = ct_kwh_to_eur_kwh(TAX_DEFAULT_CT_BY_COUNTRY["AT"])
    st.session_state["vat_percent"] = 0.0
    st.session_state["fixed_monthly_eur"] = 0.0




def apply_pending_flow():
    """Apply deferred flow changes BEFORE the flow radio is instantiated."""
    v = st.session_state.pop("pending_flow_switch", None)
    if v in ("wizard", "report"):
        st.session_state["flow"] = v
        st.session_state.pop("flow_switch", None)  # ensure UI can take the new default


def ensure_assistant_greeting():
    if st.session_state["assistant_messages"]:
        return
    lang = st.session_state.get("lang", DEFAULT_LANG)
    greeting = "Hi — I’m Eva. Ask me anything while you fill this in." if lang == "EN" else "Hi — ich bin Eva. Frag mich jederzeit während du das ausfüllst."
    st.session_state["assistant_messages"].append({"role": "assistant", "content": greeting})

def _coerce_like_default(field: str, value):
    """Coerce LLM-updated values to the expected type based on DEFAULT_INPUTS."""
    if field not in DEFAULT_INPUTS:
        return value

    d = DEFAULT_INPUTS[field]

    try:
        if isinstance(d, bool):
            # Gemini might return "true"/"false"/1/0
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "ja", "y")
            return bool(value)

        if isinstance(d, int):
            # hours / counts etc.
            return int(round(float(value)))

        if isinstance(d, float):
            return float(value)

        if isinstance(d, str):
            return str(value)

    except Exception:
        # fallback: keep original if coercion fails
        return value

    return value


def _sync_bound_widgets(field: str, value):
    """
    Sync all wizard-bound widget keys (w_<step>_<field>) so UI reflects assistant changes.
    """
    suffix = f"_{field}"
    for k in list(st.session_state.keys()):
        if k.startswith("w_") and k.endswith(suffix):
            st.session_state[k] = value


def _invalidate_report_caches():
    # keep it minimal but effective
    st.session_state["report_md"] = None
    st.session_state["report_meta"] = None
    st.session_state["report_pdf_bytes"] = None
    st.session_state["charts_md"] = None
    st.session_state.pop("pareto_df", None)
    st.session_state.pop("pareto_win_len", None)


def handle_user_chat(msg: str, current_step_name: str):
    if not msg:
        return

    ensure_assistant_greeting()
    st.session_state["assistant_messages"].append({"role": "user", "content": msg})

    # Always have fresh results for the assistant
    recalc_from_inputs()
    results = st.session_state.get("model_results")

    if not results:
        lang = st.session_state.get("lang", DEFAULT_LANG)
        st.session_state["assistant_messages"].append({
            "role": "assistant",
            "content": "Bitte zuerst den Fragebogen starten, damit ich Ergebnisse habe." if lang == "DE"
                      else "Please start the questionnaire so I have results to work with."
        })
        return

    out = call_gemini_assistant(msg, results, current_step_name)

    reply = normalize_assistant_text(out.get("reply", ""))
    st.session_state["assistant_messages"].append({
        "role": "assistant",
        "content": reply or ("OK." if st.session_state.get("lang") == "EN" else "Okay.")
    })

    upd = out.get("update_inputs") or None
    if isinstance(upd, dict) and upd:
        changed = {}

        for k, v in upd.items():
            if k not in ALLOWED_INPUT_KEYS:
                continue

            v2 = _coerce_like_default(k, v)
            st.session_state["inputs"][k] = v2
            _sync_bound_widgets(k, v2)
            changed[k] = v2

        if changed:
            _invalidate_report_caches()
            recalc_from_inputs()

            lang = st.session_state.get("lang", DEFAULT_LANG)
            st.session_state["assistant_messages"].append({
                "role": "assistant",
                "content": ("Inputs aktualisiert und neu berechnet." if lang == "DE"
                            else "Inputs updated and recalculated.")
            })

    # optional: if you want to honour show_payload
    if bool(out.get("show_payload", False)):
        st.session_state["debug_show_payload"] = True

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
    prof = res.get("profile", {}).get("derived_energy", {})

    annual_km = float(dist.get("annual_km_fleet", 0.0))
    annual_energy_mwh = float(ec.get("annual_energy_mwh", 0.0))
    annual_cost = float(ec.get("annual_cost_eur", 0.0))

    cost_per_km = (annual_cost / annual_km) if annual_km > 0 else np.nan
    ev_co2_per_km = (float(co2.get("annual_ev_co2_kg", 0.0)) / annual_km) if annual_km > 0 else np.nan
    diesel_co2_per_km = (float(dv.get("diesel_co2_baseline_kg", 0.0)) / annual_km) if annual_km > 0 else np.nan

    rows = [
        ("km/day input", "km_per_truck_per_day", prof.get("km_per_truck_per_day", np.nan), "km/day"),
        ("Battery energy per truck/day", "km/day * kWh/km", prof.get("battery_energy_kwh_per_truck_day", np.nan), "kWh/day"),
        ("Grid energy per truck/day", "battery_energy / eta", prof.get("grid_energy_kwh_per_truck_day", np.nan), "kWh/day"),
        ("Charge losses", "loss_pct", prof.get("charge_loss_pct", np.nan), "%"),
        ("Daily grid energy (fleet)", "num_trucks * grid_energy_truck_day", float(ec.get("total_daily_energy_mwh", 0.0)) * 1000.0, "kWh/day"),
        ("Grid energy per truck/day", "km/day * kWh/km / eta", prof.get("grid_energy_kwh_per_truck_day", np.nan), "kWh/day"),
        ("Annual energy", "daily_energy * operating_days", annual_energy_mwh, "MWh/year"),
        ("Effective electricity price", "…", float(ec.get("effective_price_eur_per_mwh", 0.0)) / 1000.0, "€/kWh"),
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
        pdets = res.get("energy_cost", {}).get("price_details", {}) or {}
        price_curve = pdets.get("dynamic_curve_eur_per_kwh")  # dynamic-only profile (scaled)
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

def normalize_assistant_text(raw) -> str:
    """
    Converts assistant output to plain text.
    Handles dict outputs and JSON-strings like {"reply":"..."}.
    Also salvages reply from *broken/truncated* JSON.
    """
    if raw is None:
        return ""

    # If the model already returned a dict
    if isinstance(raw, dict):
        for k in ("reply", "message", "content", "text", "answer", "response"):
            if k in raw and isinstance(raw[k], str) and raw[k].strip():
                return raw[k].strip()
        return str(raw)

    s = str(raw).strip()

    # remove code fences
    s = re.sub(r"^```(json)?\s*|\s*```$", "", s, flags=re.IGNORECASE).strip()

    # 1) Fast path: salvage "reply" even if JSON is truncated/invalid
    #    - tries a closed quote first
    m = re.search(r'"reply"\s*:\s*"((?:\\.|[^"\\])*)"', s, flags=re.S)
    if m:
        val = m.group(1)
        try:
            return json.loads('"' + val + '"')  # proper unescape
        except Exception:
            return val.replace("\\n", "\n").replace('\\"', '"').strip()

    #    - fallback: "reply":".... (no closing quote / truncated)
    m2 = re.search(r'"reply"\s*:\s*"(.*)$', s, flags=re.S)
    if m2:
        tail = m2.group(1)

        # cut off if update_inputs/show_payload begins
        tail = re.split(r'"\s*,\s*"(update_inputs|show_payload)"\s*:', tail, maxsplit=1)[0]
        tail = tail.rstrip().rstrip("}").rstrip()

        # best-effort unescape
        tail = tail.replace("\\n", "\n").replace('\\"', '"')
        return tail.strip()

    # 2) If it looks like JSON, try to parse it normally
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                for k in ("reply", "message", "content", "text", "answer", "response"):
                    if k in obj and isinstance(obj[k], str) and obj[k].strip():
                        return obj[k].strip()
                return json.dumps(obj, ensure_ascii=False)
        except Exception:
            pass

    return s

def estimate_range_km(battery_kwh: float, cons_kwh_km: float, soc_from: float = 1.0, soc_to: float = 0.0) -> float:
    battery_kwh = float(max(0.0, battery_kwh))
    cons_kwh_km = float(max(1e-6, cons_kwh_km))
    soc_from = min(max(float(soc_from), 0.0), 1.0)
    soc_to = min(max(float(soc_to), 0.0), 1.0)
    usable = max(0.0, soc_from - soc_to)
    return (battery_kwh * usable) / cons_kwh_km

import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# Strompreis-Helper (NEU: DE/AT, Fix+Dyn Split, Netzselect, Steuer/Abgabe; alles netto)
# =========================================================
import numpy as np
import streamlit as st

def eur_mwh_to_eur_kwh(eur_per_mwh: float) -> float:
    return float(eur_per_mwh) / 1000.0

def eur_kwh_to_eur_mwh(eur_per_kwh: float) -> float:
    return float(eur_per_kwh) * 1000.0

def ct_kwh_to_eur_kwh(ct_per_kwh: float) -> float:
    return float(ct_per_kwh) / 100.0

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def ensure_state(key: str, default):
    if key not in st.session_state:
        st.session_state[key] = default


# --- Netzentgelt-Tarife (aus deiner Tabelle) ---
NETZ_TARIFFE = [
    # AT — Wiener Netze
    {"country": "AT", "operator": "Wiener Netze", "level": "Netzebene 3", "power_eur_kw_year": 42.0, "energy_ct_kwh": 0.54, "loss_ct_kwh": 0.12,  "key": "Wiener Netze|Netzebene 3"},
    {"country": "AT", "operator": "Wiener Netze", "level": "Netzebene 4", "power_eur_kw_year": 47.4, "energy_ct_kwh": 0.79, "loss_ct_kwh": 0.158, "key": "Wiener Netze|Netzebene 4"},
    {"country": "AT", "operator": "Wiener Netze", "level": "Netzebene 5", "power_eur_kw_year": 57.0, "energy_ct_kwh": 1.35, "loss_ct_kwh": 0.193, "key": "Wiener Netze|Netzebene 5"},  # DEFAULT AT
    {"country": "AT", "operator": "Wiener Netze", "level": "Netzebene 6", "power_eur_kw_year": 60.6, "energy_ct_kwh": 1.97, "loss_ct_kwh": 0.339, "key": "Wiener Netze|Netzebene 6"},
    {"country": "AT", "operator": "Wiener Netze", "level": "Netzebene 7 (mit Leistungsmessung)", "power_eur_kw_year": 85.44, "energy_ct_kwh": 4.21, "loss_ct_kwh": 0.7, "key": "Wiener Netze|Netzebene 7 (mit Leistungsmessung)"},
    {"country": "AT", "operator": "Wiener Netze", "level": "Netzebene 7 (unterbrechbare Nutzung)", "power_eur_kw_year": 48.0, "energy_ct_kwh": 4.01, "loss_ct_kwh": 0.7, "key": "Wiener Netze|Netzebene 7 (unterbrechbare Nutzung)"},

    # DE — Bayernwerk
    {"country": "DE", "operator": "Bayernwerk", "level": "Umspannung Höchst-/Hochspannung <2500 h", "power_eur_kw_year": 17.18, "energy_ct_kwh": 2.81, "loss_ct_kwh": 0.0, "key": "Bayernwerk|Umspannung Höchst-/Hochspannung <2500 h"},
    {"country": "DE", "operator": "Bayernwerk", "level": "Umspannung Höchst-/Hochspannung ≥2500 h", "power_eur_kw_year": 76.7, "energy_ct_kwh": 0.43, "loss_ct_kwh": 0.0, "key": "Bayernwerk|Umspannung Höchst-/Hochspannung ≥2500 h"},
    {"country": "DE", "operator": "Bayernwerk", "level": "Hochspannung <2500 h", "power_eur_kw_year": 11.02, "energy_ct_kwh": 3.65, "loss_ct_kwh": 0.0, "key": "Bayernwerk|Hochspannung <2500 h"},
    {"country": "DE", "operator": "Bayernwerk", "level": "Hochspannung ≥2500 h", "power_eur_kw_year": 99.06, "energy_ct_kwh": 0.13, "loss_ct_kwh": 0.0, "key": "Bayernwerk|Hochspannung ≥2500 h"},
    {"country": "DE", "operator": "Bayernwerk", "level": "Umspannung Hoch-/Mittelspannung <2500 h", "power_eur_kw_year": 12.45, "energy_ct_kwh": 3.99, "loss_ct_kwh": 0.0, "key": "Bayernwerk|Umspannung Hoch-/Mittelspannung <2500 h"},
    {"country": "DE", "operator": "Bayernwerk", "level": "Umspannung Hoch-/Mittelspannung ≥2500 h", "power_eur_kw_year": 107.32, "energy_ct_kwh": 0.2, "loss_ct_kwh": 0.0, "key": "Bayernwerk|Umspannung Hoch-/Mittelspannung ≥2500 h"},
    {"country": "DE", "operator": "Bayernwerk", "level": "Mittelspannung <2500 h", "power_eur_kw_year": 15.86, "energy_ct_kwh": 4.01, "loss_ct_kwh": 0.0, "key": "Bayernwerk|Mittelspannung <2500 h"},
    {"country": "DE", "operator": "Bayernwerk", "level": "Mittelspannung ≥2500 h", "power_eur_kw_year": 99.17, "energy_ct_kwh": 0.68, "loss_ct_kwh": 0.0, "key": "Bayernwerk|Mittelspannung ≥2500 h"},
    {"country": "DE", "operator": "Bayernwerk", "level": "Umspannung Mittel-/Niederspannung <2500 h", "power_eur_kw_year": 17.61, "energy_ct_kwh": 5.88, "loss_ct_kwh": 0.0, "key": "Bayernwerk|Umspannung Mittel-/Niederspannung <2500 h"},
    {"country": "DE", "operator": "Bayernwerk", "level": "Umspannung Mittel-/Niederspannung ≥2500 h", "power_eur_kw_year": 159.96, "energy_ct_kwh": 0.18, "loss_ct_kwh": 0.0, "key": "Bayernwerk|Umspannung Mittel-/Niederspannung ≥2500 h"},  # DEFAULT DE
    {"country": "DE", "operator": "Bayernwerk", "level": "Niederspannung <2500 h", "power_eur_kw_year": 20.11, "energy_ct_kwh": 5.01, "loss_ct_kwh": 0.0, "key": "Bayernwerk|Niederspannung <2500 h"},
    {"country": "DE", "operator": "Bayernwerk", "level": "Niederspannung ≥2500 h", "power_eur_kw_year": 97.66, "energy_ct_kwh": 1.91, "loss_ct_kwh": 0.0, "key": "Bayernwerk|Niederspannung ≥2500 h"},
]

NETZ_DEFAULT_KEY_BY_COUNTRY = {
    "AT": "Wiener Netze|Netzebene 5",
    "DE": "Bayernwerk|Umspannung Mittel-/Niederspannung ≥2500 h",
}

# Steuer/Abgabe Defaults (netto; Richtwerte, kundenspezifisch/Entlastung möglich)
TAX_DEFAULT_CT_BY_COUNTRY = {
    "DE": 2.05,  # Stromsteuer: typischer Standardwert; viele B2B-Fälle haben Entlastung/Reduktionen. :contentReference[oaicite:0]{index=0}
    "AT": 1.50,  # Elektrizitätsabgabe (Richtwert); bitte kundenseitig prüfen
}


def _hhmm_list():
    return [f"{h:02d}:00" for h in range(0, 24)]

def _idx_or_default(options, value, fallback_idx):
    try:
        return options.index(value)
    except Exception:
        return fallback_idx

def _annual_kwh_from_inputs(inp: dict) -> float:
    # nutzt dieselbe Logik wie derive_energy_from_km/run_model (km/Tag * Verbrauch / eta)
    num_trucks = float(max(1, int(inp.get("num_trucks", 1))))
    operating_days = float(max(1, int(inp.get("operating_days", 1))))
    km_day = float(max(0.0, inp.get("km_per_truck_per_day", 0.0)))
    cons = float(max(0.01, inp.get("ev_consumption", 0.01)))

    loss_pct = float(inp.get("charge_loss_pct", 15.0))
    loss_pct = min(max(loss_pct, 0.0), 40.0)
    eta = 1.0 - loss_pct / 100.0
    eta = max(0.60, eta)

    kwh_truck_day_grid = (km_day * cons) / eta
    annual_kwh = num_trucks * kwh_truck_day_grid * operating_days
    return float(max(0.0, annual_kwh))

def _netztariff_by_key(key: str) -> dict | None:
    for r in NETZ_TARIFFE:
        if r["key"] == key:
            return r
    return None

def _netzentgelt_reco_eur_per_kwh(tariff: dict) -> float:
    # Nur Arbeitspreis + Verlust (ct/kWh -> €/kWh)
    return ct_kwh_to_eur_kwh(float(tariff["energy_ct_kwh"]) + float(tariff["loss_ct_kwh"]))


def render_pricing_split_ui(*, inputs: dict, default_country: str = "AT") -> dict:
    """
    Neue Strompreis-UI (netto):
    LAND, FIX, DYN, DYNANTEIL, NETZSELECT, NETZ, STEUER/ABGABE
    + Ladefenster wie bisher.
    """

    lang = st.session_state.get("lang", DEFAULT_LANG)

    def _L(de: str, en: str) -> str:
        return en if lang == "EN" else de


    st.subheader(_L("Strompreis (netto, ohne MwSt/USt)", "Electricity price (net, excl. VAT)"))

    # --- PROXIES für Transparenz-Block (damit kein NameError) ---
    site_cap_kw = kva_to_kw(
        float(inputs.get("site_capacity_kva", 0.0)),
        float(inputs.get("power_factor", 0.90)),
    )
    annual_kwh = _annual_kwh_from_inputs(inputs)

    ensure_state("pricing_country", default_country)
    ensure_state("pricing_country_prev", default_country)

    # Energiepreise netto (€/kWh)
    ensure_state("fixed_energy_eur_per_kwh", eur_mwh_to_eur_kwh(float(inputs.get("fixed_elec_price_mwh", 86.79))))
    ensure_state("dyn_energy_eur_per_kwh", eur_mwh_to_eur_kwh(float(inputs.get("avg_elec_price_mwh", 86.79))))
    ensure_state("split_dynamic_share_pct", 50)

    # Netzselect + abgeleiteter Netz-Input
    ensure_state("net_tariff_key", NETZ_DEFAULT_KEY_BY_COUNTRY.get(st.session_state["pricing_country"], NETZ_DEFAULT_KEY_BY_COUNTRY["AT"]))
    ensure_state("net_fee_eur_per_kwh", 0.0)

    # Steuer/Abgabe
    ensure_state("power_tax_eur_per_kwh", ct_kwh_to_eur_kwh(TAX_DEFAULT_CT_BY_COUNTRY.get(st.session_state["pricing_country"], 0.0)))

    # Ladefenster
    ensure_state("charge_window_start", "22:00")
    ensure_state("charge_window_end", "06:00")

    # --- Landwechsel: Defaults nachziehen ---
    top1, top2 = st.columns([1, 1])
    with top1:
        country = st.selectbox(
            _L("Land (Defaults; bitte Rechnung/DSO prüfen)", "Country (defaults; please verify with invoice/DSO)"),
            options=["AT", "DE"],
            index=["AT", "DE"].index(st.session_state["pricing_country"]),
            key="pricing_country",
        )

        prev = st.session_state.get("pricing_country_prev", None)
        if prev != country:
            st.session_state["net_tariff_key"] = NETZ_DEFAULT_KEY_BY_COUNTRY.get(country, st.session_state["net_tariff_key"])
            st.session_state["power_tax_eur_per_kwh"] = ct_kwh_to_eur_kwh(TAX_DEFAULT_CT_BY_COUNTRY.get(country, 0.0))
            st.session_state["pricing_country_prev"] = country

    with top2:
        st.slider(
            _L("Dynamik-Anteil (nur Energie, %)", "Dynamic share (energy only, %)"),
            min_value=0, max_value=100,
            value=int(st.session_state["split_dynamic_share_pct"]),
            step=5, key="split_dynamic_share_pct",
            help=_L(
                "Der Split wirkt nur auf den ENERGIEANTEIL. Netzentgelt & Abgaben bleiben fix pro kWh.",
                "The split affects ENERGY only. Grid fees & levies stay fixed per kWh.",
            ),
        )

    p_dyn = clamp01(float(st.session_state["split_dynamic_share_pct"]) / 100.0)
    p_fix = 1.0 - p_dyn

    # --- FIX / DYN Energie ---
    c_fix, c_dyn = st.columns(2)
    with c_fix:
        st.markdown(_L("**FIX (Energie)**", "**FIX (energy)**"))
        st.number_input(
            _L("FIX (€/kWh)", "FIX (€/kWh)"),
            min_value=0.0,
            value=float(st.session_state["fixed_energy_eur_per_kwh"]),
            format="%.3f",
            step=0.001,
            key="fixed_energy_eur_per_kwh",
        )

    with c_dyn:
        st.markdown(_L("**DYN (Energie)**", "**DYN (energy)**"))
        st.number_input(
            _L("DYN Ø (€/kWh)", "DYN avg (€/kWh)"),
            min_value=0.0,
            value=float(st.session_state["dyn_energy_eur_per_kwh"]),
            format="%.3f",
            step=0.001,
            key="dyn_energy_eur_per_kwh",
            help=_L(
                "Tages-/Periodenmittel. Das Ladefenster gewichtet später nur den DYN-Anteil.",
                "Daily/period average. The charging window later weights only the dynamic share.",
            ),
        )

    # --- Netzselect + Netzinput ---
    st.markdown("---")
    st.markdown(_L("**Netzentgelt (aus Tarif-Auswahl vorbefüllt, overridbar)**",
                "**Grid fees (prefilled from tariff, can be overridden)**"))

    rows_country = [r for r in NETZ_TARIFFE if r["country"] == country]
    keys_country = [r["key"] for r in rows_country]

    # Wenn ein Key aus anderem Land drin hängt, auf Default umstellen
    if st.session_state["net_tariff_key"] not in keys_country:
        st.session_state["net_tariff_key"] = NETZ_DEFAULT_KEY_BY_COUNTRY[country]

    def _format_key(k: str) -> str:
        r = _netztariff_by_key(k)
        if not r:
            return k
        return (
            f"{r['operator']} — {r['level']} "
            f"(LP {r['power_eur_kw_year']:.2f} €/kW/a | "
            f"AP {r['energy_ct_kwh']:.2f} ct/kWh | "
            f"Verlust {r['loss_ct_kwh']:.3f} ct/kWh)"
        )

    net_key = st.selectbox(
        _L("Tarif auswählen", "Select tariff"),
        options=keys_country,
        index=keys_country.index(st.session_state["net_tariff_key"]),
        format_func=_format_key,
        key="net_tariff_key",
    )

    tariff = _netztariff_by_key(net_key)

    # Leistungspreis separat für Peak-Mehrkosten (€/kW/Jahr)
    ensure_state("net_power_eur_kw_year", float(tariff["power_eur_kw_year"]) if tariff else 0.0)

    # Netzarbeitspreis (€/kWh) Empfehlung
    reco = _netzentgelt_reco_eur_per_kwh(tariff) if tariff else None

    # --- Tariff-Wechsel-Handling / Manual Override Logik ---
    prev_key = st.session_state.get("net_tariff_key_prev")
    key_changed = (prev_key != net_key)

    if key_changed:
        # bei Tarifwechsel: wieder "auto" und Empfehlung setzen
        st.session_state["net_fee_is_manual"] = False
        st.session_state["net_tariff_key_prev"] = net_key

    # Auto-set nur, wenn nicht manuell (oder wenn noch 0) und reco existiert
    if reco is not None:
        if (not st.session_state.get("net_fee_is_manual", False)) or float(st.session_state.get("net_fee_eur_per_kwh", 0.0)) <= 0.0:
            st.session_state["net_fee_eur_per_kwh"] = float(reco)

    def _mark_net_manual():
        st.session_state["net_fee_is_manual"] = True

    # Button: Empfehlung bewusst übernehmen (setzt wieder auto)
    btn_cols = st.columns([0.35, 0.65])
    with btn_cols[0]:
        if st.button(_L("Netzempfehlung übernehmen", "Apply recommended grid fee"), disabled=(reco is None)):
            st.session_state["net_fee_is_manual"] = False
            if reco is not None:
                st.session_state["net_fee_eur_per_kwh"] = float(reco)
            st.rerun()

    with btn_cols[1]:
        if reco is None:
            st.caption(_L("Kein Tarif gefunden – bitte Auswahl prüfen.", "No tariff found — please check selection."))
        else:
            st.caption(f"Empfehlung (Proxy): {reco*100:.2f} ct/kWh (inkl. Verlust + LP umgelegt)")

    # Transparenz
    if tariff and reco is not None:
        with st.expander(_L("Tarif-Details (Transparenz)", "Tariff details (transparency)"), expanded=False):
            ap_ct = float(tariff["energy_ct_kwh"]) + float(tariff["loss_ct_kwh"])
            lp_ct = (float(tariff["power_eur_kw_year"]) * site_cap_kw / annual_kwh * 100.0) if annual_kwh > 0 else 0.0
            reco = ct_kwh_to_eur_kwh(ap_ct + lp_ct)
            st.write(
                f"- Jahresarbeit (Proxy): {annual_kwh:,.0f} kWh/a\n"
                f"- Standortlimit (Proxy für Leistungspreis): {site_cap_kw:,.0f} kW\n"
                f"- Arbeitspreis + Verlust: {ap_ct:.3f} ct/kWh\n"
                f"- Empfehlung NETZ gesamt: {reco*100:.2f} ct/kWh"
            )



    # Manuelles Netzentgelt
    st.number_input(
        _L("Netz (€/kWh)", "Grid fee (€/kWh)"),
        min_value=0.0,
        value=float(st.session_state["net_fee_eur_per_kwh"]),
        step=0.001,
        format="%.3f",
        key="net_fee_eur_per_kwh",
        on_change=_mark_net_manual,
    )



    st.number_input(
        _L("Abgaben/Steuer (€/kWh)", "Levies/tax (€/kWh)"),
        min_value=0.0,
        value=float(st.session_state["power_tax_eur_per_kwh"]),
        step=0.001,
        format="%.3f",
        key="power_tax_eur_per_kwh",
    )

    gs_hr()

    st.number_input(
        _L(
            "Leistungsentgelt (€/kW/Jahr) – relevant bei Peak-Mehrkosten",
            "Capacity charge (€/kW/year) — relevant for peak-driven costs"
        ),
        min_value=0.0,
        value=float(st.session_state["net_power_eur_kw_year"]),
        step=1.0,
        key="net_power_eur_kw_year",
    )


    # --- KPI/„spannende“ Metrics (nicht Fix vs Dyn getrennt, sondern relevant) ---
    fixed_energy = float(st.session_state["fixed_energy_eur_per_kwh"])
    dyn_energy = float(st.session_state["dyn_energy_eur_per_kwh"])
    netz = float(st.session_state["net_fee_eur_per_kwh"])
    tax = float(st.session_state["power_tax_eur_per_kwh"])

    energy_mix = p_fix * fixed_energy + p_dyn * dyn_energy
    total_price = energy_mix + netz + tax

    # Anteil-Logik
    def _share(part, total):
        return (100.0 * part / total) if total > 1e-9 else 0.0

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(_L("Gesamtpreis (€/kWh)", "Total price (€/kWh)"), f"{total_price:,.3f}")
    m2.metric(_L("Energie (€/kWh)", "Energy (€/kWh)"), f"{energy_mix:,.3f}")
    m3.metric(_L("Netz (€/kWh)", "Grid (€/kWh)"), f"{netz:,.3f}")
    m4.metric(_L("Abgaben/Steuer (€/kWh)", "Levies/tax (€/kWh)"), f"{tax:,.3f}")

    with st.expander(_L("Komposition (Anteile)", "Composition (shares)"), expanded=False):
        st.write(
            f"- Energieanteil: {_share(energy_mix, total_price):.1f}%\n"
            f"- Netzentgelt: {_share(netz, total_price):.1f}%\n"
            f"- Steuer/Abgabe: {_share(tax, total_price):.1f}%\n"
            f"- Dynamik-Anteil (nur Energie): {p_dyn*100:.0f}%", "- Dynamic share (energy only): {p_dyn*100:.0f}%"
        )

    # --- Ladefenster (wie bisher) ---
    st.subheader(_L("Ladefenster", "Charging window"))

    options = _hhmm_list()
    w1, w2 = st.columns(2)
    with w1:
        st.selectbox(_L("Start", "Start"), options=options,
                     index=_idx_or_default(options, st.session_state["charge_window_start"], 22),
                     key="charge_window_start")
    with w2:
        st.selectbox(_L("Ende", "End"), options=options,
                     index=_idx_or_default(options, st.session_state["charge_window_end"], 6),
                     key="charge_window_end")


    def window_len_hours(start_h: int, end_h: int) -> int:
    # Mitternacht korrekt, Start==Ende als 24h interpretieren (optional)
        if start_h == end_h:
            return 24
        return (end_h - start_h) % 24

    try:
        sh = int(str(st.session_state["charge_window_start"]).split(":")[0]) % 24
        eh = int(str(st.session_state["charge_window_end"]).split(":")[0]) % 24
        hours = window_len_hours(sh, eh)

        st.caption(
            f"Fensterlänge: {hours} h "
            f"(Start {sh:02d}:00 → Ende {eh:02d}:00). "
            "Mitternacht wird korrekt behandelt."
        )
    except Exception:
        pass


    return {
        "country": country,
        "p_dyn": p_dyn,
        "fixed_energy_eur_per_kwh": fixed_energy,
        "dyn_energy_eur_per_kwh": dyn_energy,
        "net_fee_eur_per_kwh": netz,
        "tax_eur_per_kwh": tax,
        "total_price_eur_per_kwh": total_price,
        "charge_window_start": st.session_state["charge_window_start"],
        "charge_window_end": st.session_state["charge_window_end"],
    }


def ct_kwh_to_eur_mwh(ct_per_kwh: float) -> float:
    return float(ct_per_kwh) * 10.0

def eur_mwh_to_ct_kwh(eur_per_mwh: float) -> float:
    return float(eur_per_mwh) / 10.0

def kva_to_kw(kva: float, power_factor: float) -> float:
    pf = max(0.0, min(1.0, float(power_factor)))
    return max(0.0, float(kva)) * pf


def build_example_load_profile_csv(days: int = 7, freq: str = "15min", seed: int = 7) -> bytes:
    rng = np.random.default_rng(seed)

    start = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)
    ts = pd.date_range(start=start, periods=int((24*60/int(freq.replace("min","")))*days), freq=freq)

    # realistisches Muster: Nacht niedriger, Tag höher, leichte Peaks + Noise
    hours = ts.hour + ts.minute/60.0
    base = 420 + 180 * np.sin((hours - 7) / 24 * 2*np.pi)  # Tageswelle
    night_dip = np.where((hours >= 22) | (hours < 6), -120, 0)  # nachts weniger
    noise = rng.normal(0, 25, size=len(ts))
    occasional_peaks = (rng.random(len(ts)) < 0.015) * rng.uniform(150, 420, size=len(ts))

    kw = np.clip(base + night_dip + noise + occasional_peaks, 80, 1200)
    df = pd.DataFrame({
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "site_load_kw": np.round(kw, 1),
    })
    return df.to_csv(index=False).encode("utf-8")


# Helper Ende

# =========================================================
# Page config + styling (kept)
# =========================================================
st.set_page_config(page_title="FleetMate — Guided Intake", layout="wide")
require_login()


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

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
            
/* =========================
   Eva Chat: highlight + fixed height + scroll
   ========================= */
.eva-chat-card{
  border: 1.6px solid rgba(58,141,255,0.42);
  box-shadow: 0 18px 46px rgba(58,141,255,0.14), var(--shadow3);
}

.eva-chat-scroll{
  max-height: 300px;              /* <- “angemessene Höhe” */
  overflow-y: auto;
  padding-right: 6px;
}

/* optional: nicer scrollbar */
.eva-chat-scroll::-webkit-scrollbar{ width: 10px; }
.eva-chat-scroll::-webkit-scrollbar-thumb{
  background: rgba(255,255,255,0.18);
  border-radius: 999px;
}
.eva-chat-scroll::-webkit-scrollbar-track{
  background: rgba(255,255,255,0.06);
  border-radius: 999px;
}
            
/* applies to the container that holds Eva */
.eva-card {
  border: 1.6px solid rgba(58,141,255,0.42);
  box-shadow: 0 18px 46px rgba(58,141,255,0.14);
  border-radius: 16px;
}

/* Expander/Details – Theme-agnostisch */
div[data-testid="stExpander"] details{
  border-radius: 14px;
  border: 1px solid var(--border, rgba(0,0,0,0.12));
  background: var(--card, rgba(255,255,255,0.75));
}
div[data-testid="stExpander"] summary{
  color: var(--txt, rgba(10,12,18,0.92)) !important;
}
div[data-testid="stExpander"] .stMarkdown,
div[data-testid="stExpander"] p,
div[data-testid="stExpander"] li,
div[data-testid="stExpander"] span,
div[data-testid="stExpander"] label{
  color: var(--txt, rgba(10,12,18,0.92)) !important;
}
            
/* alles was dein Dark Theme sonst nicht einfärbt */
.stMarkdown, .stCaption, .stTextLabel, label, span {
  color: inherit;
}

</style>
""", unsafe_allow_html=True)

# Theme CSS injection
inject_theme_css(st.session_state.get("theme", "system"))

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

def render_intro_page():
    lang = st.session_state.get("lang", DEFAULT_LANG)

    # --- Top bar: Language + Theme (also on page 0) ---
    bar_l, bar_r = st.columns([0.70, 0.30], vertical_alignment="center")
    with bar_l:
        st.caption(" ")  # keeps spacing clean (optional)

    with bar_r:
        c_lang, c_theme = st.columns([0.55, 0.45], vertical_alignment="center")

        with c_lang:
            current_lang = st.session_state.get("lang", DEFAULT_LANG)
            lang_pick = st.radio(
                "",
                options=["DE", "EN"],
                index=0 if current_lang == "DE" else 1,
                horizontal=True,
                label_visibility="collapsed",
                key="lang_switch_intro",
            )
            if lang_pick != current_lang:
                st.session_state["lang"] = lang_pick
                st.rerun()

        with c_theme:
            current_theme = st.session_state.get("theme", DEFAULT_THEME)
            theme_pick = st.radio(
                "",
                options=["system", "dark", "light"],
                index=["system", "dark", "light"].index(current_theme),
                horizontal=True,
                format_func=lambda v: "🖥️" if v=="system" else ("🌙" if v=="dark" else "☀️"),
                label_visibility="collapsed",
                key="theme_switch_intro",
            )
            if theme_pick != current_theme:
                st.session_state["theme"] = theme_pick
                st.rerun()


    # --- copy (DE/EN) ---
    COPY = {
        "EN": {
            "kicker": "EV charging feasibility for logistics depots",
            "h1": "FleetMate",
            "sub": (
                "A guided intake that turns fleet reality into a defendable business case — "
                "energy demand, costs, CO₂, and grid headroom."
            ),

            "who_title": "Who it’s for",
            "who_body": (
                "FleetMate is designed for **mid-sized logistics operators** — and the people who must sign off decisions: "
                "**Operations**, **Fleet Management**, **Finance**, and **Site/Energy**.\n\n"
                "If you need a fast, honest answer to **“Can our depot support electrification — and what will it cost?”**, "
                "you’re in the right place."
            ),

            "what_title": "What FleetMate does",
            "what_points": [
                "Turn **km/day → kWh/day** using your fleet and utilisation.",
                "Compare **Diesel vs EV** with your own prices (**net values**, no VAT added).",
                "Check **depot feasibility**: how much **headroom** is left in your charging window.",
                "Expose risks early: **peak conflicts**, unrealistic windows, and a transparent **worst-case** view.",
            ],

            "assumptions_title": "Default assumptions",
            "assumptions_points": [
                "**JBD: 2,500 operating hours/year** (typical mid-sized logistics baseline — adjustable).",
                "**Electricity & diesel are treated as net prices** (no VAT/USt added by FleetMate).",
                "**Depot charging is the default** within your chosen window (public HPC isn’t the baseline model).",
            ],

            "load_title": "Load profile",
            "load_body": (
                "Uploading a site load profile is a big upgrade: FleetMate calculates **real headroom** by measuring the "
                "**peak inside your charging window** — much more reliable than a manual guess.\n\n"
                "You can use the **example CSV** as a template for your own export."
            ),

            "eva_title": "Meet EVA — your FleetMate copilot",
            "eva_body": (
                "EVA does more than explain: she can **propose adjustments** and help you **change inputs on the fly** "
                "(charging window, mileage, fleet size, pricing mode, etc.).\n\n"
                "She also flags inconsistencies — so you don’t build a case on shaky assumptions."
            ),

            "eva_chips_title": "Try asking EVA:",
            "eva_chips": [
                "“Change the charging window to 22:00–06:00 and show the impact.”",
                "“Which 3 inputs drive savings the most in my scenario?”",
                "“Increase the fleet by 20% and update the results.”",
                "“Do we still have headroom at night with this load profile?”",
            ],

            "cta_primary": "🚀 Start the tool",
            "cta_secondary": "Download example load profile (CSV)",
            "note": (
                "Note: FleetMate supports feasibility decisions. Final sizing requires DSO data, measurements, "
                "a charging/EMS concept and engineering."
            ),

            "qs_title": "Quick start (5 steps)",
            "qs_points": [
                "**Fleet & usage**: trucks, km/day, operating pattern",
                "**Battery & consumption**: realistic kWh/km (no SoC inputs)",
                "**Charging window**: when trucks are actually at the depot",
                "**Site capacity & load profile**: headroom during the window",
                "**Diesel baseline & toll**: fair comparison"
            ],
            "qs_hint": "Tip: If you’re not sure, EVA can suggest reasonable defaults and update inputs for you.",
            "cta_box_title": "Start or try with a template",
            "example_note": "Use the CSV template if you don’t have an export yet (it shows the expected format).",

            "clarity_title": "Mini glossary",
            "clarity_items": [
                ("Headroom", "The free power capacity at the site during your charging window (how much ‘room’ is left before you hit the limit)."),
                ("Charging window", "The hours when vehicles are actually at the depot and can charge — not a theoretical time range."),
                ("Worst-case “simultaneous max”", "An intentionally strict upper bound: assumes many vehicles charge at once. It shows the ceiling, not the typical day."),
            ],

        },

        "DE": {
            "kicker": "Machbarkeitscheck für E-Laden im Logistik-Depot",
            "h1": "FleetMate",
            "sub": (
                "Ein geführtes Intake, das Betriebsrealität in einen belastbaren Case übersetzt — "
                "Energiebedarf, Kosten, CO₂ und Netz-Headroom."
            ),

            "who_title": "Für wen ist das?",
            "who_body": (
                "FleetMate ist für **mittlere Logistikunternehmen** gebaut – und für die Rollen, die Entscheidungen tragen: "
                "**Operations**, **Fleet Management**, **Finance** und **Standort/Energie**.\n\n"
                "Wenn du schnell und ehrlich beantworten willst **„Geht das bei uns am Depot – und was kostet’s?“**, "
                "bist du hier richtig."
            ),

            "what_title": "Was macht FleetMate",
            "what_points": [
                "**km/Tag → kWh/Tag** aus deinen Flotten- und Nutzungsdaten ableiten.",
                "**Diesel vs. E** vergleichen – mit deinen Preisen (**netto**, ohne MwSt/USt-Zuschlag).",
                "**Standort-Machbarkeit** prüfen: wie viel **Headroom** im Ladefenster wirklich übrig bleibt.",
                "**Risiken früh** sichtbar machen: Peak-Konflikte, unrealistische Fenster, transparenter **Worst-Case**.",
            ],

            "assumptions_title": "Standardannahmen",
            "assumptions_points": [
                "**JBD: 2.500 h/Jahr** als typische Baseline für ein mittleres Logistikunternehmen (anpassbar).",
                "**Strom- und Dieselpreise werden netto behandelt** (FleetMate addiert keine MwSt/USt).",
                "**Laden am Standort** im definierten Fenster (öffentliches HPC ist nicht die Standardannahme).",
            ],

            "load_title": "Lastprofil",
            "load_body": (
                "Ein Lastprofil ist ein echter Hebel: FleetMate berechnet **realen Headroom**, indem der "
                "**Peak im Ladefenster** direkt aus Messwerten kommt — deutlich belastbarer als ein manueller Peak.\n\n"
                "Nutze das **Beispiel-CSV** als Vorlage für deinen Export."
            ),

            "eva_title": "EVA — dein FleetMate Co-Pilot",
            "eva_body": (
                "EVA erklärt nicht nur: sie kann **Änderungen vorschlagen** und dir helfen, **Werte direkt anzupassen** "
                "(Ladefenster, km/Tag, Flottengröße, Preis-Modus, etc.).\n\n"
                "Außerdem findet sie Ungereimtheiten — damit du keinen Case auf wackligen Annahmen baust."
            ),

            "eva_chips_title": "Frag EVA z. B.:",
            "eva_chips": [
                "„Stell das Ladefenster auf 22:00–06:00 und zeig den Effekt.“",
                "„Welche 3 Inputs treiben die Einsparung am stärksten?“",
                "„Mach +20% LKW und update die Ergebnisse.“",
                "„Haben wir nachts mit Lastprofil noch Headroom?“",
            ],

            "cta_primary": "🚀 Tool starten",
            "cta_secondary": "Beispiel-Lastprofil (CSV) downloaden",
            "note": (
                "Hinweis: FleetMate ist eine Entscheidungs-/Machbarkeitsunterstützung. Für finale Auslegung braucht es "
                "Netzbetreiber-Daten, Messwerte, Lade-/EMS-Konzept und ggf. Elektroplanung."
            ),

            "qs_title": "Schnellstart (5 Schritte)",
            "qs_points": [
                "**Flotte & Nutzung**: LKW, km/Tag, Einsatzmuster",
                "**Batterie & Verbrauch**: realistische kWh/km (keine SoC-Inputs)",
                "**Ladefenster**: wann die LKW wirklich am Depot sind",
                "**Standort & Lastprofil**: Headroom im Ladefenster",
                "**Diesel-Baseline & Maut**: fairer Vergleich"
            ],
            "qs_hint": "Tipp: Wenn du unsicher bist, kann EVA sinnvolle Defaults vorschlagen und Inputs direkt für dich anpassen.",
            "cta_box_title": "Starten oder mit Vorlage testen",
            "example_note": "Nutze das CSV als Vorlage, wenn du noch keinen Export hast (zeigt das erwartete Format).",

            "clarity_title": "Mini-Glossar",
            "clarity_items": [
                ("Leistungsreserve", "Freie Leistung am Standort im Ladefenster (wie viel ‘Luft’ bleibt, bevor das Anschlusslimit erreicht ist)."),
                ("Ladefenster", "Die Stunden, in denen Fahrzeuge wirklich am Depot stehen und laden können — nicht ein theoretisches Zeitintervall."),
                ("Worst-Case „simultan @ max“", "Absichtlich streng: nimmt an, dass viele Fahrzeuge gleichzeitig laden. Zeigt die Obergrenze, nicht den Normalfall."),
            ],
        },
    }


    C = COPY["DE"] if lang == "DE" else COPY["EN"]

    # --- layout ---
    left, right = st.columns([0.62, 0.38], gap="large")

    with left:
        st.markdown(
            f"""
            <div class="gs-pill">⚡ {C["kicker"]}</div>
            <h1 class="gs-title" style="margin-top:.55rem;">{C["h1"]}</h1>
            <p class="gs-sub" style="font-size:1.05rem; max-width: 56rem;">{C["sub"]}</p>
            """,
            unsafe_allow_html=True
        )

        with st.container(border=True):
            st.markdown("<span class='gs-card-marker'></span>", unsafe_allow_html=True)
            st.markdown(f"### {C['who_title']}")
            st.write(C["who_body"])

        with st.container(border=True):
            st.markdown("<span class='gs-card-marker'></span>", unsafe_allow_html=True)
            st.markdown(f"### {C['what_title']}")
            for p in C["what_points"]:
                st.markdown(f"- {p}")

        with st.container(border=True):
            st.markdown("<span class='gs-card-marker'></span>", unsafe_allow_html=True)
            st.markdown(f"### {C['assumptions_title']}")
            for p in C["assumptions_points"]:
                st.markdown(f"- {p}")

        with st.container(border=True):
            st.markdown("<span class='gs-card-marker'></span>", unsafe_allow_html=True)
            st.markdown(f"### {C['load_title']}")
            st.write(C["load_body"])

        with st.container(border=True):
            st.markdown("<span class='gs-card-marker'></span>", unsafe_allow_html=True)
            st.markdown(f"### {C['clarity_title']}")
            for term, desc in C["clarity_items"]:
                st.markdown(f"**{term}:** {desc}")


    with right:
        with st.container(border=True):
            st.markdown("<span class='gs-card-marker'></span>", unsafe_allow_html=True)
            st.markdown(f"### {C['eva_title']}")
            st.write(C["eva_body"])
            st.markdown(f"**{C['eva_chips_title']}**")
            for chip in C["eva_chips"]:
                st.markdown(f"- {chip}")

        with st.container(border=True):
            st.markdown("<span class='gs-card-marker'></span>", unsafe_allow_html=True)
            st.markdown(f"### {C['qs_title']}")
            for p in C["qs_points"]:
                st.markdown(f"- {p}")
            st.caption(C["qs_hint"])

        with st.container(border=True):
            st.markdown("<span class='gs-card-marker'></span>", unsafe_allow_html=True)

            st.markdown(f"### {C['cta_box_title']}")
            st.caption(C["example_note"])

            example_bytes = build_example_load_profile_csv(days=7, freq="15min", seed=7)
            st.download_button(
                label=C["cta_secondary"],
                data=example_bytes,
                file_name="fleetmate_example_load_profile.csv",
                mime="text/csv",
                use_container_width=True,
            )

            if st.button(C["cta_primary"], use_container_width=True):
                st.session_state["intro_done"] = True
                st.rerun()

            st.caption(C["note"])


# =========================================================
# Render
# =========================================================
ensure_defaults()
apply_pending_flow()
ensure_assistant_greeting()
inject_theme_css(st.session_state.get("theme", DEFAULT_THEME))

# --- Page 0: Intro / Onboarding ---
if not st.session_state.get("intro_done", False):
    render_intro_page()
    st.stop()


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
                    f"<span class='gs-pill'>🏭 {t('peak_in_window')}: {fmt_num(cap.get('site_peak_kw_in_window'))} kW</span> "
                    f"&nbsp; <span class='gs-pill'>🧩 {t('headroom')}: {fmt_num(cap.get('available_kw_at_peak'))} kW</span>",
                    unsafe_allow_html=True
                )


    with st.container(border=True):
        st.markdown("### 💬 " + t("assistant_title"))
        st.caption(t("assistant_hint"))

        msg_box = st.container(height=300)
        with msg_box:
            for m in st.session_state["assistant_messages"][-80:]:
                with st.chat_message(m["role"]):
                    st.write(m["content"])

        user_msg = st.chat_input(t("ask_placeholder"))
        if user_msg:
            handle_user_chat(user_msg, step_name)
            st.rerun()


# ---------- Left: Wizard or Report ----------
with left:
    step_idx, step = current_step()
    pct = int(((step_idx + 1) / len(STEPS)) * 100)

    mode_l, mode_r = st.columns([0.55, 0.45], vertical_alignment="center")
    with mode_l:
        flow_key = "flow_switch"
        radio_kwargs = dict(
            options=["wizard", "report"],
            horizontal=True,
            format_func=lambda v: t("questionnaire") if v == "wizard" else t("report"),
            label_visibility="collapsed",
            key=flow_key,
        )
        if flow_key not in st.session_state:
            radio_kwargs["index"] = 0 if st.session_state.get("flow") == "wizard" else 1

        mode = st.radio("", **radio_kwargs)

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

            elif sid == "battery":
                st.subheader("Fahrprofil als LKW-Typ" if lang == "DE" else "Truck type as driving profile")

                # ensure selection exists
                if not get_inp("truck_type_id"):
                    set_inp("truck_type_id", TRUCK_TYPES[0]["id"])

                cols = st.columns(len(TRUCK_TYPES))
                for j, tt in enumerate(TRUCK_TYPES):
                    with cols[j]:
                        try:
                            st.image(tt["img"], use_container_width=True)
                        except Exception:
                            pass

                        def _tt_txt(obj, k: str, lang: str) -> str:
                            v = obj.get(k)
                            if isinstance(v, dict):
                                return v.get(lang) or v.get("EN") or v.get("DE") or ""
                            return str(v) if v is not None else ""

                        st.markdown(f"**{_tt_txt(tt, 'name', lang)}**  \n{_tt_txt(tt, 'subtitle', lang)}")

                        full_range = estimate_range_km(tt["battery_kwh"], tt["cons_kwh_km"], 1.0, 0.0)
                        practical_range = estimate_range_km(tt["battery_kwh"], tt["cons_kwh_km"], 0.80, 0.10)

                        st.caption(
                            (
                                f"Consumption: {tt['cons_kwh_km']:.2f} kWh/km · "
                                f"Battery: {int(tt['battery_kwh'])} kWh · "
                                f"Range: {full_range:,.0f} km (100–0) / {practical_range:,.0f} km (80–10) · "
                                f"Max charging: {int(tt['max_charge_kw'])} kW"
                            )
                            if lang == "EN"
                            else
                            (
                                f"Verbrauch: {tt['cons_kwh_km']:.2f} kWh/km · "
                                f"Batterie: {int(tt['battery_kwh'])} kWh · "
                                f"Reichweite: {full_range:,.0f} km (100–0) / {practical_range:,.0f} km (80–10) · "
                                f"Max. Laden: {int(tt['max_charge_kw'])} kW"
                            )
                        )


                        if st.button("Auswählen" if lang == "DE" else "Select", key=f"pick_{tt['id']}", use_container_width=True):
                            set_inp("truck_type_id", tt["id"])
                            # apply defaults on pick (customer-friendly)
                            set_inp("ev_consumption", float(tt["cons_kwh_km"]))
                            set_inp("battery_kwh", float(tt["battery_kwh"]))
                            # optional: also prefill charger power
                            set_inp("charger_power_kw", float(tt["max_charge_kw"]))
                            _sync_bound_widgets("truck_type_id", tt["id"])
                            _sync_bound_widgets("ev_consumption", float(tt["cons_kwh_km"]))
                            _sync_bound_widgets("battery_kwh", float(tt["battery_kwh"]))
                            _sync_bound_widgets("charger_power_kw", float(tt["max_charge_kw"]))
                            st.rerun()

                gs_hr()

                bind_number(
                    sid, "km_per_truck_per_day",
                    "Ø km pro LKW und Tag" if lang == "DE" else "Average km per truck per day",
                    min_value=0.0, max_value=2000.0, step=10.0,
                )

                with st.expander("Erweitert: Ladeverluste & Werte überschreiben" if lang == "DE" else "Advanced: charging losses & override values", expanded=False):
                    bind_slider(
                        sid, "charge_loss_pct",
                        "Ladeverluste Netz→Batterie (%)" if lang == "DE" else "Charging losses grid→battery (%)",
                        5, 25, 1,
                        help=("Default 15% ist ein robuster Modellwert. 20–25% eher Worst-Case (kalt/ineffizient)."
                              if lang == "DE" else
                              "Default 15% is robust. 20–25% is more worst-case (cold/inefficient chain).")
                    )

                    override = st.checkbox("Defaults überschreiben" if lang == "DE" else "Override defaults", value=False, key="override_truck_defaults")
                    if override:
                        bind_number(
                            sid, "ev_consumption",
                            "Verbrauch (kWh/km)" if lang == "DE" else "Consumption (kWh/km)",
                            min_value=0.4, max_value=3.0, step=0.05,
                        )
                        bind_number(
                            sid, "battery_kwh",
                            "Batterie (kWh)" if lang == "DE" else "Battery (kWh)",
                            min_value=100.0, max_value=1500.0, step=25.0,
                        )

                # Derived daily energy (battery + grid)
                d = derive_energy_from_km(st.session_state["inputs"])

                st.markdown("**Tagesenergie (abgeleitet)**" if lang == "DE" else "**Daily energy (derived)**")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Batterieenergie / Tag" if lang == "DE" else "Battery energy / day",
                              f"{d['battery_energy_kwh_per_truck_day']:,.0f} kWh")
                with c2:
                    st.metric("Netzenergie / Tag" if lang == "DE" else "Grid energy / day",
                              f"{d['grid_energy_kwh_per_truck_day']:,.0f} kWh")
                with c3:
                    st.metric("Ladeverluste" if lang == "DE" else "Charging losses",
                              f"{d['charge_loss_pct']:,.0f}%")
                with c4:
                    rs = d.get("residual_soc_arrival", None)
                    rk = d.get("residual_kwh_arrival", None)

                    if rs is None:
                        st.metric("Restladung bei Ankunft" if lang == "DE" else "Residual charge on arrival", "—")
                    else:
                        st.metric(
                            "Restladung bei Ankunft" if lang == "DE" else "Residual charge on arrival",
                            f"{rs*100:,.0f}%",
                            f"{rk:,.0f} kWh" if rk is not None else None
                        )

                if d.get("warning"):
                    st.warning(d["warning"])



            elif sid == "electricity":
                # Sync initial window if missing
                if "charge_window_start" not in st.session_state:
                    st.session_state["charge_window_start"] = f"{int(get_inp('start_hour'))%24:02d}:00"
                if "charge_window_end" not in st.session_state:
                    st.session_state["charge_window_end"] = f"{int(get_inp('end_hour'))%24:02d}:00"

                pricing_out = render_pricing_split_ui(inputs=st.session_state["inputs"], default_country="AT")

                # Energiepreise aus UI kommen in €/kWh, Modell erwartet €/MWh (ENERGIE-Anteil)
                set_inp("fixed_price_eur_per_mwh", eur_kwh_to_eur_mwh(pricing_out["fixed_energy_eur_per_kwh"]))
                set_inp("avg_elec_price_mwh",      eur_kwh_to_eur_mwh(pricing_out["dyn_energy_eur_per_kwh"]))
                set_inp("dynamic_share", float(pricing_out["p_dyn"]))

                # Netz + Steuer/Abgabe als Add-ons
                set_inp("grid_eur_per_kwh",   float(pricing_out["net_fee_eur_per_kwh"]))
                set_inp("levies_eur_per_kwh", float(pricing_out["tax_eur_per_kwh"]))

                # Netto fix
                set_inp("vat_percent", 0.0)
                set_inp("fixed_monthly_eur", 0.0)

                # Ladefenster
                set_inp("start_hour", int(str(pricing_out["charge_window_start"]).split(":")[0]) % 24)
                set_inp("end_hour",   int(str(pricing_out["charge_window_end"]).split(":")[0]) % 24)

                gs_hr()



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

                def example_load_profile_csv() -> bytes:
                    idx = pd.date_range("2026-01-01 00:00", "2026-01-03 00:00", freq="15min", inclusive="left")
                    # simple shape: day higher, night lower
                    base = []
                    for ts in idx:
                        h = ts.hour
                        v = 1200 if 7 <= h < 18 else 800
                        base.append(v)
                    df_ex = pd.DataFrame({"timestamp": idx, "site_kw": base})
                    return df_ex.to_csv(index=False).encode("utf-8")

                st.download_button(
                    ("Beispiel-Lastprofil (CSV) herunterladen" if lang == "DE" else "Download example load profile (CSV)"),
                    data=build_example_load_profile_csv(days=7, freq="15min"),
                    file_name="fleetmate_example_load_profile_15min.csv",
                    mime="text/csv",
                )
                st.caption(
                    ("Spalten: timestamp, site_load_kw. 15-min Raster, 7 Tage."
                    if lang == "DE" else
                    "Columns: timestamp, site_load_kw. 15-min resolution, 7 days.")
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
                        m = st.session_state.get("profile_cache", {}).get("metrics") or {}
                        if m.get("peak_kw_in_window") is not None:
                            st.markdown("**Kapazität im Ladefenster**" if lang == "DE" else "**Capacity in charging window**")

                            # Always read inputs via get_inp() (handles wizard keys correctly)
                            site_kva = float(get_inp("site_capacity_kva"))
                            pf = float(get_inp("power_factor"))
                            cap_kw = kva_to_kw(site_kva, pf)

                            peak_window = float(m.get("peak_kw_in_window", 0.0))
                            peak_global = float(m.get("peak_kw_overall", 0.0))
                            avg_kwh_day_window = float(m.get("energy_kwh_in_window_avg_per_day", 0.0))
                            avg_kw_window = float(m.get("avg_kw_in_window", 0.0))

                            headroom_peak = max(0.0, cap_kw - peak_window)

                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Peak gesamt (kW)", f"{peak_global:,.0f}")
                            c2.metric("Peak im Ladefenster (kW)", f"{peak_window:,.0f}")
                            c3.metric("Ø kWh/Tag im Ladefenster", f"{avg_kwh_day_window:,.0f}")
                            c4.metric(("Ø Last im Ladefenster" if lang == "DE" else "Average load in charging window (kW)"), f"{avg_kw_window:,.0f}")
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
                    "Site capacity limit in kVA (kw will be calculated with cos φ 0.9)" if lang == "EN" else "Standort-Limit in kVA (kw wird mit cos φ 0.9 berechnet)",
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
                    st.markdown("### Max. Trucks with Peak" if lang == "EN"
                                else "### Max. LKW mit Lastspitze")
                    st.caption(
                        (
                            "We start from the **available site capacity** "
                            "(site connection limit minus the measured peak load within the charging window). "
                            "This capacity is then translated into the **maximum number of trucks** using two views: "
                            "**energy-based feasibility** and a **simultaneous charging worst-case**."
                        )
                        if lang == "EN" else
                        (
                            "Ausgangspunkt ist die **verfügbare Leistungsreserve am Standort** "
                            "(Anschlusslimit abzüglich der gemessenen Peak-Last im Ladefenster). "
                            "Diese Reserve wird anschließend in die **maximal ladbare Anzahl an LKW** übersetzt – "
                            "einmal **energie-basiert** und einmal als **konservativer Simultan-Worst-Case**."
                        )
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
                    "AT": {"diesel_price": 1.55, "toll_rate": 0.456, "tolled_share": 0.60},
                    "DE": {"diesel_price": 1.50, "toll_rate": 0.21, "tolled_share": 0.55},
                    "EU": {"diesel_price": 1.55, "toll_rate": 0.23, "tolled_share": 0.55},
                }

                bind_select(sid, "market_region", "Region" if lang == "EN" else "Region", options=["AT", "DE", "EU"])

                if st.button(t("apply_estimate"), use_container_width=True, key="apply_market_estimate"):
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

                st.markdown("**" + t("toll_assumptions") + "**")
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
                _invalidate_report_caches()
                st.session_state["pending_flow_switch"] = "report"
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
                with b: st.metric(t("kpi_ev_cost_year"), fmt_eur(ec.get("annual_cost_eur")))
                with c: st.metric(t("kpi_peak"), f"{fmt_num(load.get('new_theoretical_peak_kw'))} kW")

                # Capacity window KPIs (new)
                if cap:
                    k1, k2, k3, k4 = st.columns(4)
                    with k1:
                        st.metric(t("kpi_peak_window"), f"{float(cap.get('site_peak_kw_in_window',0.0)):,.0f} kW")
                    with k2:
                        st.metric(t("kpi_headroom_peak"), f"{float(cap.get('available_kw_at_peak',0.0)):,.0f} kW")
                    with k3:
                        st.metric(t("kpi_max_trucks_energy"), f"{int(cap.get('max_trucks_energy_based',0))}")
                    with k4:
                        v = cap.get("recommended_avg_kw_per_truck")
                        st.metric(t("kpi_rec_avg_kw_truck"), f"{float(v):,.0f} kW" if v else "—")
                
                gs_hr()

                with st.container(border=True):
                    st.markdown("<span class='gs-card-marker'></span>", unsafe_allow_html=True)

                    st.markdown("## " + t("analysis_title"))
                    st.markdown(report_analysis_markdown(res))

                    st.markdown("### " + t("constraints_title"))
                    constraints = report_constraints(res)
                    if not constraints:
                        st.success(t("no_constraints"))
                    else:
                        for lvl, txt in constraints:
                            if lvl == "ok":
                                st.success(txt)
                            elif lvl == "bad":
                                st.error(txt)
                            else:
                                st.warning(txt)

                    st.markdown("### " + t("next_steps_title"))
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

                    st.markdown("#### " + t("best_next_actions"))
                    for idx, s in enumerate(solutions, 1):
                        st.markdown(f"**{idx}. {s.get('title','')}** — {s.get('definition','')}")

                gs_hr()

                lang = st.session_state.get("lang", "DE")

                # --- Reverse Calculation (robust, ohne Duplikate, KPI-Style bleibt) ---
                if cap:
                    with st.container(border=True):
                        st.markdown("<span class='gs-card-marker'></span>", unsafe_allow_html=True)

                        st.markdown("### " + t("reverse_calc_title"))

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
                            st.caption(t("reverse_calc_caption"))
                            st.markdown(
                                (
                                    f"- Ladefenster: **{int(cap.get('charging_window_hours_total', 0))} h**\n"
                                    f"- Peak im Ladefenster: **{float(cap.get('site_peak_kw_in_window', 0.0)):,.0f} kW**\n"
                                    f"- Headroom (Peak): **{float(cap.get('available_kw_at_peak', 0.0)):,.0f} kW**\n"
                                ) if lang == "DE" else (
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

                m = st.session_state["profile_cache"]["metrics"]

                c1, c2, c3, c4 = st.columns(4)
                peak_global = float(m.get("peak_kw_overall", m.get("peak_kw_overall_kw", m.get("peak_kw", 0.0))))
                peak_window = float(m.get("peak_kw_in_window", 0.0))
                avg_kwh_day_window = float(m.get("energy_kwh_in_window_avg_per_day", 0.0))
                avg_kw_window = float(m.get("avg_kw_in_window", 0.0))

                c1.metric("Peak gesamt (kW)", f"{peak_global:,.0f}")
                c2.metric("Peak im Ladefenster (kW)", f"{peak_window:,.0f}")
                c3.metric("Ø kWh/Tag im Ladefenster", f"{avg_kwh_day_window:,.0f}")
                c4.metric("Sampling (Median)", f"{m['sampling_minutes_median']:.0f} min")

                # Stunden im Ladefenster als Tabelle
                sh = int(str(st.session_state["charge_window_start"]).split(":")[0]) % 24
                eh = int(str(st.session_state["charge_window_end"]).split(":")[0]) % 24

                rows = []
                for h in range(24):
                    if _in_window_hours(h, sh, eh):
                        rows.append({"Stunde": f"{h:02d}:00", "Ø kWh in Stunde": m["hourly_kwh_in_window_avg"][h]})
                st.dataframe(pd.DataFrame(rows), use_container_width=True)


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
                    ("Events / truck / day", "km_per_truck_per_day", 0.20),
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
                    "km_per_truck_per_day": 0.25,
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
                    elif key == "km_per_truck_per_day":
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
                st.markdown("#### " + t("interpretation"))
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
