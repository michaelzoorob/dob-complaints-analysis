"""
Configuration for the inspector leniency causal analysis.

Defines outcome variables, fixed effect structures, sample restrictions,
and joins across the five linked datasets.
"""

# ── Borough code mappings ───────────────────────────────────────────────

# Our bis_scrape uses full names; PLUTO uses numeric codes;
# permits/violations use single-digit strings
BOROUGH_NAME_TO_CODE = {
    "MANHATTAN": "1", "BRONX": "2", "BROOKLYN": "3",
    "QUEENS": "4", "STATEN ISLAND": "5",
}
BOROUGH_CODE_TO_NAME = {v: k for k, v in BOROUGH_NAME_TO_CODE.items()}

# Permits use abbreviations
BOROUGH_CODE_TO_PERMIT = {
    "1": "MANHATTAN", "2": "BRONX", "3": "BROOKLYN",
    "4": "QUEENS", "5": "STATEN ISLAND",
}


# ── BBL construction ────────────────────────────────────────────────────

def make_bbl(boro: str, block: str, lot: str) -> str:
    """Construct a 10-digit BBL from boro (1-5), block, and lot."""
    if not boro or not block or not lot:
        return ""
    try:
        return f"{int(boro)}{int(block):05d}{int(lot):04d}"
    except (ValueError, TypeError):
        return ""


# ── Fixed effect cell definitions ───────────────────────────────────────

# The identification strategy relies on within-cell variation.
# Within a cell, inspector assignment should be quasi-random.

FE_CELLS = {
    "baseline": ["category_description", "assigned_to", "year_month"],
    "tight": ["category_description", "assigned_to", "year_month", "time_block"],
    "saturated": ["category_description", "assigned_to", "year_month",
                   "time_block", "priority", "day_of_week"],
    "spatial": ["category_description", "assigned_to", "year_month",
                "community_board"],
}


# ── Outcome windows for compliance analysis ─────────────────────────────

# After an inspection, look for permits/violations within these windows
COMPLIANCE_WINDOWS_DAYS = [30, 60, 90, 180, 365]


# ── Spatial spillover definitions ───────────────────────────────────────

# "Neighbors" for spatial analysis
NEIGHBOR_DEFINITIONS = {
    "same_block": "Same census block (boro + block)",
    "radius_100m": "Within 100m radius (Haversine)",
    "radius_250m": "Within 250m radius",
}


# ── Minimum sample thresholds ──────────────────────────────────────────

MIN_INSPECTOR_CASES = 30  # Minimum cases per inspector
MIN_CELL_SIZE = 5         # Minimum observations per FE cell


# ── Proactive-enforcement category families ─────────────────────────────

# 2-char complaint-category prefixes (substr(category_code, 1, 2)) grouped
# by the kind of agency-initiated activity they represent. Fixed here once
# per the Wave-1 proactive-enforcement inventory (2026-07); see
# data/analysis/blog_posts/proactive_enforcement_plan.md.

_PROACTIVE_FAMILY_PREFIXES = {
    # Cyclical / legally mandated compliance tracking (LL188, facade/FISP,
    # tenant-protection, demo tracking, structural-monitoring roster,
    # shelters, sign registration, retaining walls)
    "statutory_periodic": (
        "7K", "7F", "6V", "2E", "2F", "2L", "2P", "6B", "6C", "6D",
        "4S", "6Z", "6Y", "7Q", "8P",
    ),
    # Sweeps, compliance checks, and unit-initiated field work where DOB
    # chooses the target (8A compliance, 7G sweeps, EWO, worker
    # endangerment, padlock/quality-of-life)
    "discretionary_field": (
        "8A", "7G", "1X", "1Y", "1V", "1U", "2Y", "5G", "6X", "91",
        "4J", "4B", "5H",
    ),
    # Re-inspections keyed to earlier enforcement actions
    "followup": ("7R", "4G", "1L", "2H"),
    # Incident-type categories carrying both caller and agency streams
    # (unstable buildings, falling debris, unsafe demo, excavation,
    # scaffold accidents, adjacent-construction damage)
    "mixed_incident": ("30", "10", "12", "14", "03", "1E", "23", "67"),
}

# Flat map: 2-char prefix -> family name
PROACTIVE_FAMILIES = {
    prefix: family
    for family, prefixes in _PROACTIVE_FAMILY_PREFIXES.items()
    for prefix in prefixes
}

# BIS parser artifact: category_description reads "Date" for a slice of
# rows in several categories; 73 and 7R have no clean description anywhere
# in the scrape (7R is 100% "Date"), so hand-map them from the DOB
# complaint-category code table.
CATEGORY_DESC_OVERRIDES = {
    "73": "FAILURE TO MAINTAIN",
    "7R": "KEY CLASS 1 HAZARDOUS VIOLATION FOLLOW-UP",
}
