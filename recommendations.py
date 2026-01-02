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
