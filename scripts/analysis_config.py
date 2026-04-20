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
