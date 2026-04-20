"""
Official BIS Complaint Disposition Codes (Rev. 09/21)
Source: https://www.nyc.gov/assets/buildings/pdf/bis_complaint_disposition_codes.pdf

Grouped by analytical category for the inspector leniency analysis.
"""

# ── Outcome taxonomy for inspector leniency analysis ────────────────────

# ENFORCEMENT ACTION TAKEN — inspector found a problem and took action
# These reflect inspector discretion to issue a violation/penalty
VIOLATION_CODES = {
    # Direct violations & summonses
    "A1": "Buildings Violation(s) Served",
    "A2": "Criminal Court Summons Served",
    "A3": "Full Stop Work Order Served",
    "A4": "Buildings Violation(s) and Criminal Court Summons Served",
    "A6": "Vacant/Open/Unguarded Structure - Violation(s) Issued",
    "A8": "OATH Violation Served",
    "A9": "OATH and DOB Violations Served",
    # HRP (Housing Recovery Program) enforcement
    "AA": "HRP: Request for Corrective Action Issued",
    "AC": "HRP: Request for Corrective Action Issued - Not Corrected",
    # SWO violations
    "H3": "Building Violation Issued for Failure to Obey Stop Work Order",
    "H4": "Criminal Court Summons Served Due to Failure to Obey SWO",
    "H5": "Stop All Work/No TCAO",
    # Stop work orders (new)
    "L1": "Partial Stop Work Order",
    # Vacate orders (most severe)
    "Y1": "Full Vacate Order Served",
    "Y3": "Partial Vacate Order Served",
    "W1": "Violation Served for Disobeying a Vacate Order",
    # CSC violations
    "U3": "CSC: Violation Issued (No OATH) - On Action Complaint",
    "U4": "CSC: Full Stop Work Order Issued - On Action Complaint",
    "U5": "CSC: Partial SWO Issued - On Action Complaint",
    "U6": "CSC: Violation Issued (No OATH) - On Closed Site",
    "V2": "FAIL Inspection",
    "V3": "Stop Work Order Violation Served (Non-Compliant After-Hours Work)",
    # MARCH enforcement
    "MB": "MARCH: Failure to Maintain Building/OATH Violation Issued",
    "MC": "MARCH: Contrary to Approved Plans/OATH Violation Issued",
    "MD": "MARCH: Exit Passage Obstructed/OATH Violation Issued",
    "ME": "MARCH: Exit Passage Obstructed/OATH Violation & Full Vacate Issued",
    "MF": "MARCH: Exit Passage Obstructed/OATH Violation & Partial Vacate Issued",
    "MG": "MARCH: Occupancy Contrary to C of O/OATH Violation Issued",
    "MH": "MARCH: No PA Permit/OATH Violation & Full Vacate Issued",
    "MI": "MARCH: No PA Permit/OATH Violation & Partial Vacate Issued",
    "MJ": "MARCH: Work Without a Permit/OATH Violation Issued",
    "MK": "MARCH: No PA Permit/OATH Violation Issued",
    # Unsafe building
    "RK": "Unsafe Building: Violation Issued",
    # POPS
    "X2": "Work Does Not Conform to TPP - OATH Summons Issued",
    "X7": "POPS Unauthorized Partial Closure - OATH Summons Issued",
    "X8": "POPS Unauthorized Full Closure - OATH Summons Issued",
    # Closures
    "P3": "Closure/Padlock Order Issued",
    # Other enforcement
    "B1": "Buildings Violation(s) Prepared - Attempt to Serve",
    "B2": "OATH Summons Prepared - Attempt to Serve",
    "K5": "Letter of Deficiency Issued",
    "K6": "Letter of Deficiency Issued with Partial SWO",
    "ND": "Notice of Deficiency Issued",
    "R5": "Inspection: Class 1 OATH Written/Order to Correct",
}

# NO VIOLATION FOUND — inspector inspected and found no actionable condition
# This is the key "no enforcement" outcome reflecting inspector discretion
NO_VIOLATION_CODES = {
    "I1": "Complaint Unsubstantiated Based on Department Records",
    "I2": "No Violation Warranted for Complaint at Time of Inspection",
    "I3": "Compliance Inspection Performed",
    "MA": "MARCH: No Enforcement Action Taken",
    "R1": "Inspection: No Immediate Action/No Follow-Up Required",
    "U2": "CSC: Construction Safety in Compliance",
    "V1": "PASS Inspection",
    "WA": "Weather Related: No Action Necessary",
    "X1": "Compliance Inspection Completed - No Enforcement Action Required",
    "X3": "All Work Substantially Conforms to Plans",
    # Corrected/resolved
    "AB": "HRP: Request for Corrective Action Issued - Corrected",
    "AF": "Class 1 Condition Resolved",  # from our data - not in PDF but appears
    "K7": "Notification of Correction Received",
    "K8": "Correction Verified by DOB",
    "Q4": "Compromised Structure: Condition Remedied",
    "RG": "Commissioner's Order: Owner Remediation Completed",
    "RL": "Unsafe Building: Action Completed",
    "S0": "All Work Completed",
    "L2": "Stop Work Order Fully Rescinded",
    "L3": "Stop Work Order Partially Rescinded",
    "Y2": "Vacate Order Fully Rescinded",
    "Y4": "Vacate Order Partially Rescinded",
    "P4": "Closure/Padlock Order Rescinded",
}

# NO ACCESS — inspector could not inspect (NOT about discretion)
# Exclude from the main analysis of inspector strictness
NO_ACCESS_CODES = {
    "C1": "Inspector Unable to Gain Access - 1st Attempt",
    "C2": "Inspector Unable to Gain Access - 2nd Final Attempt",
    "C3": "Access Denied - 1st Attempt",
    "C4": "Access Denied - 2nd Attempt",
    "C5": "AW: No Access - 1st Attempt",
    "C6": "AW: Access Denied - 1st Attempt",
    "C7": "AW: No Access - 2nd Attempt",
    "C8": "AW: Access Denied - 2nd Attempt",
    "U1": "CSC: Unable to Gain Access",
    "WB": "Weather Related: No Access",
}

# REFERRAL/REASSIGNMENT — complaint routed elsewhere (not inspector discretion)
# Exclude from the main analysis
REFERRAL_CODES = {
    "D1": "Assigned to Construction Enforcement",
    "D2": "Assigned to Plumbing Unit",
    "D3": "Assigned to Elevator Unit",
    "D4": "Assigned to Construction Safety Compliance Unit",
    "D5": "Assigned to Emergency Response Team",
    "D6": "Assigned to Boiler Unit",
    "D7": "Assigned to Cranes and Derricks Unit",
    "D9": "Assigned to Electrical Unit",
    "EB": "Assigned to Facade Inspection Safety Program",
    "EC": "Assigned to Structurally Compromised Buildings Unit",
    "ED": "Assigned to Retaining Wall Unit",
    "EF": "Assigned to Mayor's Office of Special Enforcement",
    "EG": "Assigned to Quality of Life Unit",
    "EH": "Assigned to Concrete Enforcement Unit",
    "EJ": "Assigned to Construction Safety Enforcement (CSE)",
    "EK": "Assigned to Real-Time Enforcement Unit",
    "EZ": "Assigned to Department of Investigation (DOI)",
    "E1": "Assigned to Building Marshal's Office",
    "E2": "Assigned to Legal Affairs/Padlock Unit",
    "E3": "Assigned to Borough Office for Final Inspection",
    "E6": "Assigned to Special Operations Unit",
    "E9": "Assigned to Stalled Sites Unit",
    "F1": "Referred to DEP",
    "F2": "Referred to DHCR",
    "F3": "Referred to DOHMH",
    "F5": "Referred to DOS",
    "F6": "Referred to DOT",
    "F7": "Referred to NYS Real Properties",
    "F8": "Referred to HPD",
    "F9": "Referred to HUD",
    "G1": "Referred to Inspector General",
    "G2": "Referred to Parks",
    "G3": "Referred to TLC",
    "G4": "Referred to DCA",
    "G5": "Referred to NYPD",
    "G6": "Referred to FDNY",
    "G7": "Referred to MOSE",
    "G8": "Referred to NYCHA",
    "G9": "Referred to DCAS",
    "H1": "Please See Complaint Number (cross-reference)",
    "H2": "Previously Inspected - Pre-BIS Complaint Number",
    "XX": "Administrative Closure",
    "K1": "Insufficient Information/Unable to Locate Address",
    "K2": "Address Invalid",
}

# PENDING/FOLLOW-UP — not a final outcome
PENDING_CODES = {
    "J1": "Follow-Up Inspection to be Scheduled",
    "J2": "Resolved by Periodic Inspection",
    "J3": "Reviewed - Inspection to Be Scheduled",
    "J4": "Follow-Up Inspection Scheduled for Hazardous Condition",
    "J5": "Sign Field Inspection Conducted - Under Review",
    "J6": "Sign Moratorium 2019",
    "P1": "Job Vested",
    "P2": "Follow Up Inspection Required Pending Adoption",
    "R2": "Inspection: No Immediate Action/Weekly Inspection",
    "R3": "Inspection: No Immediate Action/Monthly Inspection",
    "R4": "Inspection: Engineering Assessment Required",
}


def classify_disposition(code: str) -> str:
    """Classify a disposition code into an analytical category.

    Returns one of: 'violation', 'no_violation', 'no_access', 'referral',
    'pending', or 'other'.
    """
    if not code:
        return "other"
    code = code.strip().upper()
    if code in VIOLATION_CODES:
        return "violation"
    if code in NO_VIOLATION_CODES:
        return "no_violation"
    if code in NO_ACCESS_CODES:
        return "no_access"
    if code in REFERRAL_CODES:
        return "referral"
    if code in PENDING_CODES:
        return "pending"
    return "other"


def get_analysis_sample_condition() -> str:
    """SQL WHERE clause fragment for the inspector leniency analysis sample.

    Includes only complaints with a substantive outcome where the inspector
    exercised discretion: violation found OR no violation found.
    Excludes no-access, referrals, pending, and other non-discretionary outcomes.
    """
    violation = list(VIOLATION_CODES.keys())
    no_violation = list(NO_VIOLATION_CODES.keys())
    all_codes = violation + no_violation
    placeholders = ",".join(f"'{c}'" for c in all_codes)
    return f"o.disposition_code IN ({placeholders})"


# ── Severity scale for secondary analysis ───────────────────────────────

SEVERITY_LEVELS = {
    # Level 0: No violation
    "no_violation": 0,
    # Level 1: Corrective action / letter of deficiency
    "corrective": 1,  # AA, K5, K6, ND
    # Level 2: OATH (administrative tribunal) violation
    "oath_violation": 2,  # A8, X2
    # Level 3: Buildings violation (more serious)
    "buildings_violation": 3,  # A1, A4
    # Level 4: Stop work order
    "stop_work": 4,  # A3, L1
    # Level 5: Criminal summons
    "criminal": 5,  # A2, A4, H4
    # Level 6: Vacate / closure order (most severe)
    "vacate_closure": 6,  # Y1, Y3, P3
}


def classify_severity(code: str) -> int:
    """Return a severity level (0-6) for a disposition code."""
    if not code:
        return -1
    code = code.strip().upper()
    if code in NO_VIOLATION_CODES:
        return 0
    if code in ("AA", "AC", "K5", "K6", "ND"):
        return 1
    if code in ("A8", "A9", "X2", "X7", "X8", "MB", "MC", "MD", "MG", "MJ", "MK"):
        return 2
    if code in ("A1", "A4", "A6", "H3", "RK", "U3", "U6", "V3"):
        return 3
    if code in ("A3", "L1", "U4", "U5"):
        return 4
    if code in ("A2", "H4", "W1"):
        return 5
    if code in ("Y1", "Y3", "ME", "MF", "MH", "MI", "P3"):
        return 6
    return -1
