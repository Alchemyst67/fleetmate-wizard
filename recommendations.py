# ISSUE DETECTION
def detect_issues(results):
    issues = []

    load = results["load"]
    dv = results["diesel_vs_ev"]
    ec = results["energy_cost"]
    prof = results["charging_profile"]
    
    # Grid capacity exceeded
    if not load["capacity_ok"]:
        issues.append({
            "id": "capacity_exceeded",
            "severity": "high",
            "description": "Site connection / grid capacity is exceeded by EV charging load."
        })

    # High peak concentration (even if capacity OK)
    if load["new_theoretical_peak_kw"] > 1.5 * load["new_avg_load_kw"]:
        issues.append({
            "id": "high_peak_concentration",
            "severity": "medium",
            "description": "Charging demand is highly concentrated, creating peak stress."
        })

    # Financial issue
    if dv["total_savings_incl_toll_eur"] <= 0:
        issues.append({
            "id": "negative_business_case",
            "severity": "medium",
            "description": "EV operating costs are not lower than diesel under current assumptions."
        })

    return issues

# SOLUTION GENERATION
def generate_solution_set(results, issues):
    load = results["load"]
    inp = results["inputs"]
    dv = results["diesel_vs_ev"]

    issue_ids = {i["id"] for i in issues}
    solutions = []

    # SOLUTION 1 — SMART CHARGING
    if issue_ids & {"capacity_exceeded", "high_peak_concentration"}:
        solutions.append({
            "title": "Smart charging / load management",
            "definition": (
                "Software-controlled charging that dynamically limits total site power "
                "to avoid exceeding grid capacity."
            ),
            "how_to": [
                "Install OCPP-compatible smart chargers",
                "Configure site-level power cap (kW)",
                "Apply staggered or priority-based charging rules"
            ],
            "pros": [
                "Lowest CAPEX solution",
                "Fast to deploy",
                "No grid upgrade required"
            ],
            "cons": [
                "May increase charging time",
                "Requires backend control system"
            ],
            "quantitative": {
                "peak_reduction_kw": round(load["required_shaving_kw"], 1),
                "overload_kw": round(load["new_theoretical_peak_kw"] - inp["site_capacity_limit_kva"], 1),
                "capex_level": "low"
            },
            "rank_score": 90
        })

    # SOLUTION 2 — BATTERY ENERGY STORAGE
    if issue_ids & {"capacity_exceeded", "high_peak_concentration"}:
        solutions.append({
            "title": "Battery energy storage (peak shaving)",
            "definition": (
                "A stationary battery supplies power during peak charging periods, "
                "reducing grid draw."
            ),
            "how_to": [
                "Install on-site battery system",
                "Charge battery during off-peak hours",
                "Discharge battery during EV charging peaks"
            ],
            "pros": [
                "Physically reduces peak load",
                "Improves site resilience",
                "Future-proof for expansion"
            ],
            "cons": [
                "High CAPEX",
                "Efficiency losses"
            ],
            "quantitative": {
                "required_battery_kwh": round(load["required_battery_energy_kwh"], 1),
                "required_power_kw": round(load["required_shaving_kw"], 1),
                "capex_level": "high"
            },
            "rank_score": 70
        })

    # SOLUTION 3 — REDUCE CHARGER POWER
    if issue_ids & {"capacity_exceeded", "high_peak_concentration"}:
        reduced_power = max(inp["charger_power_per_truck_kw"] * 0.5, 50)

        solutions.append({
            "title": "Reduce charger power rating",
            "definition": (
                "Lower the per-charger power to reduce simultaneous peak demand."
            ),
            "how_to": [
                "Install lower-power chargers",
                "Or apply software power caps per charger"
            ],
            "pros": [
                "Very low CAPEX",
                "Simple to implement"
            ],
            "cons": [
                "Longer charging times",
                "Less operational flexibility"
            ],
            "quantitative": {
                "current_charger_kw": inp["charger_power_per_truck_kw"],
                "recommended_charger_kw": round(reduced_power, 0),
                "capex_level": "low"
            },
            "rank_score": 65
        })

    # SOLUTION 4 — GRID / TRANSFORMER UPGRADE
    if "capacity_exceeded" in issue_ids:
        solutions.append({
            "title": "Grid connection / transformer upgrade",
            "definition": (
                "Permanent increase of grid or transformer capacity to support EV load."
            ),
            "how_to": [
                "Apply for grid upgrade with utility",
                "Upgrade transformer and protection equipment",
                "Recommission site connection"
            ],
            "pros": [
                "Permanent solution",
                "No operational constraints"
            ],
            "cons": [
                "Very high CAPEX",
                "Long lead time",
                "Permitting required"
            ],
            "quantitative": {
                "required_capacity_kva": round(load["new_theoretical_peak_kw"], 0),
                "capex_level": "very high"
            },
            "rank_score": 40
        })

    # SOLUTION 5 — COST OPTIMISATION (BUSINESS CASE)
    prof = results["charging_profile"]
    hours = list(range(24))
    prices = prof["tou_price_eur_per_kwh"]
    co2 = prof["grid_co2_g_per_kwh"]
    flags = prof["flags"]

    cheapest_hours = sorted(range(24), key=lambda h: prices[h])[:5]
    lowest_co2_hours = sorted(range(24), key=lambda h: co2[h])[:5]

    covered_cheapest = sum(flags[h] for h in cheapest_hours)
    covered_co2 = sum(flags[h] for h in lowest_co2_hours)

    solutions.append({
        "title": "Shift charging to cheaper / lower-CO₂ hours",
        "category": "Energy cost / CO₂",
        "priority": "high",
        "rank_score": 95,
        "applicable_if": [
            "negative_business_case",
            "cost_optimisation_opportunity"
        ],
        "definition": (
            "Adjust the charging start and end times so that EV charging happens "
            "during hours with lower electricity prices and/or lower grid CO₂ intensity."
        ),
        "how_to": [
            "Change charging start and end hours in the input panel",
            "Align charging window with lowest TOU price hours",
            "Prefer night-time or early-morning charging where possible"
        ],
        "pros": [
            "No CAPEX required",
            "Immediate cost savings",
            "Reduces CO₂ footprint"
        ],
        "cons": [
            "Requires operational flexibility",
            "May conflict with vehicle availability"
        ],
        "quantitative": {
            "current_effective_price_eur_per_mwh": round(
                results["energy_cost"]["effective_price_eur_per_mwh"], 1
            ),
            "cheapest_tou_price_eur_per_kwh": min(
                results["charging_profile"]["tou_price_eur_per_kwh"]
            )
        },
        "when_to_use": (
            "Best when EV is more expensive than diesel and site capacity is sufficient."
        )
    })

    # Sorting best to words
    solutions.sort(key=lambda x: x["rank_score"], reverse=True)

    return solutions[:3]  # max 3 shown

def _g(d, *path, default=None):
    """safe nested get: _g(res,'a','b')"""
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def _fmt(x, unit="", digits=0):
    try:
        v = float(x)
        if digits == 0:
            s = f"{v:,.0f}"
        else:
            s = f"{v:,.{digits}f}"
        return f"{s} {unit}".strip()
    except Exception:
        return "—"

def _fmt_eur(x): return _fmt(x, "€", 0)
def _fmt_kw(x):  return _fmt(x, "kW", 0)
def _fmt_kg(x):  return _fmt(x, "kg", 0)

def report_analysis_markdown(res: dict) -> str:
    # Core KPIs
    savings_eur = _g(res, "diesel_vs_ev", "total_savings_incl_toll_eur")
    ev_cost_eur = _g(res, "diesel_vs_ev", "ev_cost_eur") or _g(res, "energy_cost", "annual_cost_eur")
    co2_kg      = _g(res, "co2", "annual_savings_kg")

    peak_new_kw     = _g(res, "load", "new_theoretical_peak_kw")
    peak_window_kw  = _g(res, "capacity_analysis", "site_peak_kw_in_window")
    headroom_kw     = _g(res, "capacity_analysis", "available_kw_at_peak")
    cap_limit_kw    = _g(res, "capacity_analysis", "site_capacity_kw")

    max_trucks_energy = _g(res, "capacity_analysis", "max_trucks_by_energy") or _g(res, "load", "max_trucks_energy")
    rec_kw_per_truck  = _g(res, "capacity_analysis", "recommended_kw_per_truck") or _g(res, "load", "recommended_kw_per_truck")

    # Data quality flags
    used_spot = _g(res, "energy_cost", "price_details", "used_spot_data")
    used_co2  = _g(res, "co2", "co2_details", "used_grid_data")

    # Plausibility nudge (keine falsche Sicherheit)
    plausi_hint = ""
    try:
        if savings_eur is not None and float(savings_eur) > 200000:
            plausi_hint = (
                "⚠️ **Plausibilitätscheck:** Die Einsparung ist sehr hoch. "
                "Bitte prüfe besonders **Jahres-km**, **Dieselverbrauch**, **Dieselpreis** und **Mautannahmen** "
                "(sonst wirkt das Ergebnis „zu gut um wahr zu sein“)."
            )
    except Exception:
        pass

    # Capacity status
    cap_status = ""
    try:
        if headroom_kw is not None and float(headroom_kw) < 0:
            cap_status = "🔴 **Kapazität kritisch:** Peak im Ladefenster über Standortlimit → ohne Lastmanagement/Netzausbau riskant."
        else:
            cap_status = "🟢 **Kapazität aktuell ok:** Peak im Ladefenster bleibt unter dem Standortlimit (mit Headroom)."
    except Exception:
        cap_status = "🟡 **Kapazität unklar:** Bitte Lastprofil/Limit prüfen."

    # Build markdown (kurz, aber substanziell)
    md = f"""
### Executive Summary (aus deinen Inputs gerechnet)
- **Einsparung (inkl. Maut):** **{_fmt_eur(savings_eur)} pro Jahr**
- **CO₂-Ersparnis:** **{_fmt_kg(co2_kg)} pro Jahr**
- **EV-Stromkosten:** **{_fmt_eur(ev_cost_eur)} pro Jahr**
- **Netz/Last:** neuer theoretischer Peak **{_fmt_kw(peak_new_kw)}**, Peak im Ladefenster **{_fmt_kw(peak_window_kw)}**, Headroom **{_fmt_kw(headroom_kw)}** (Limit **{_fmt_kw(cap_limit_kw)}**)

{plausi_hint}

### Was treibt das Ergebnis wirklich?
- Haupthebel sind typischerweise **Dieselpreis × Jahres-km × Verbrauch** vs. **effektiver Strompreis im Ladefenster**.
- Das Ladefenster entscheidet, ob du **billige/CO₂-ärmere Stunden** triffst – oder teuer/dirty lädst.
- Datenqualität: Strompreis = **{"Spot/Upload" if used_spot else "Fallback/Annahme"}**, CO₂ = **{"Grid/Upload" if used_co2 else "Fallback/Annahme"}**.

### Infrastruktur-Realität (nicht Marketing)
- {cap_status}
- Energetisch schaffst du in deinem Setup ungefähr **{_fmt(max_trucks_energy, 'Lkw')}** (Energie-Kriterium).
- Richtwert Ladeleistung: **{_fmt_kw(rec_kw_per_truck)} pro Lkw** (für dein Fenster/Profil).

### Empfehlung (konkret)
1) **Lastgang + Limit verifizieren** (Netzbetreiber/Smart Meter): Peak im Ladefenster ist der Showstopper – nicht „Durchschnitt“.
2) **Ladefenster optimieren** (Start/Ende so legen, dass Preis & CO₂ im Fenster niedrig sind).
3) **Lastmanagement definieren**: Priorisierung (frühe Abfahrten), maximale Standort-kW, dynamische Drosselung.
4) **Business-Check**: Die Top-4 Inputs (km/Jahr, l/100km, Dieselpreis, Maut) als „Proof“ dokumentieren → erhöht Glaubwürdigkeit massiv.
"""
    return md.strip()

def report_constraints(res: dict) -> list[tuple[str, str]]:
    """returns list of (level, text) where level in {'ok','warn','bad'}"""
    out = []
    headroom_kw = _g(res, "capacity_analysis", "available_kw_at_peak")
    used_spot = _g(res, "energy_cost", "price_details", "used_spot_data")
    used_co2  = _g(res, "co2", "co2_details", "used_grid_data")

    try:
        if headroom_kw is not None and float(headroom_kw) < 0:
            out.append(("bad", "Kapazität im Ladefenster überschritten → ohne Lastmanagement/Netzausbau nicht belastbar."))
        else:
            out.append(("ok", "Kapazität im Ladefenster aktuell ok (Headroom vorhanden)."))
    except Exception:
        out.append(("warn", "Kapazität konnte nicht sauber bewertet werden."))

    if not used_spot:
        out.append(("warn", "Strompreis basiert auf Annahmen/Fallback → Upload von Spot/TOU-Daten erhöht Genauigkeit."))
    if not used_co2:
        out.append(("warn", "CO₂ basiert auf Annahmen/Fallback → Upload von Grid-CO₂-Profil verbessert Aussage."))

    return out

def report_next_steps(res: dict) -> list[str]:
    headroom_kw = _g(res, "capacity_analysis", "available_kw_at_peak")
    steps = []
    steps.append("**Heute (30 min):** Inputs plausibilisieren (km/Jahr, Verbrauch, Dieselpreis, Maut) – sonst sind Top-Zahlen wertlos.")
    steps.append("**Diese Woche:** Lastprofil/Limit fixieren (Messdaten/Netzbetreiber) + Ladefenster/Abfahrtszeiten als harte Constraints definieren.")
    try:
        if headroom_kw is not None and float(headroom_kw) < 0:
            steps.append("**Priorität A:** Lastmanagement/Peak-Shaving planen (Drosselung, Staffelung, ggf. Batterie/PV/Netzausbau).")
        else:
            steps.append("**Priorität A:** Smart-Charging-Regeln implementieren (max kW am Standort + Prioritäten nach Abfahrt).")
    except Exception:
        steps.append("**Priorität A:** Kapazitätslage klären (Headroom/Peak).")

    steps.append("**30–90 Tage:** Charger-Plan (Anzahl/Leistung/Redundanz) + CAPEX/OPEX inkl. Förderungen/Netzkosten grob rechnen.")
    return steps
