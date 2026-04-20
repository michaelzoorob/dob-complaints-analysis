"""
Prototype: Validate that we can look up DOB complaint details by complaint number
and extract inspector comments from BIS Web.

Steps:
1. Fetch a small batch of recent complaints from NYC Open Data (Socrata API)
2. For each, fetch the BIS Web detail page using the complaint number
3. Parse the HTML to extract comments and other fields not in Open Data
4. Print results to validate the join
"""

import requests
import time
import re
from html.parser import HTMLParser

# ── Config ──────────────────────────────────────────────────────────────────

OPEN_DATA_URL = "https://data.cityofnewyork.us/resource/eabe-havv.json"
BIS_WEB_URL = "https://a810-bisweb.nyc.gov/bisweb/OverviewForComplaintServlet"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}
REQUEST_TIMEOUT = 120  # BIS Web is very slow


# ── Step 1: Fetch complaints from Open Data ─────────────────────────────────

def fetch_open_data_complaints(limit=5):
    """Fetch recent closed complaints with inspection dates from Socrata API."""
    params = {
        "$limit": limit,
        "$order": "date_entered DESC",
        "$where": "status='CLOSED' AND inspection_date IS NOT NULL",
    }
    resp = requests.get(OPEN_DATA_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ── Step 2: Parse BIS Web HTML ──────────────────────────────────────────────

def clean(text: str) -> str:
    """Strip HTML tags and clean whitespace from extracted text."""
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace('&nbsp;', ' ').replace('&amp;', '&')
    return ' '.join(text.split()).strip()


def parse_bis_detail(html: str) -> dict:
    """Extract key fields from BIS Web complaint detail HTML.

    The BIS Web HTML uses <b>Label:</b> inside <td class="content"> cells,
    with values either in the same cell or in adjacent sibling <td> cells.
    """
    result = {}

    # Complaint number + status from title tag
    m = re.search(r'<title>Overview for Complaint #:(\S+)\s*=\s*(\w+)</title>', html)
    if m:
        result["bis_complaint_number"] = m.group(1)
        result["bis_status"] = m.group(2)

    # Subject (Re:) — in a <td> with colspan="6" after "Re:&nbsp;&nbsp;"
    m = re.search(r'Re:&nbsp;&nbsp;(.*?)</td>', html, re.DOTALL)
    if m:
        result["subject"] = clean(m.group(1))

    # Category Code — <b>Category Code:</b> then value in same or next <td>
    m = re.search(r'<b>Category Code:</b>\s*</td>\s*<td[^>]*>(.*?)</td>', html, re.DOTALL)
    if m:
        result["category_code"] = clean(m.group(1))

    # Category description — next <tr> after category code
    m = re.search(
        r'<b>Category Code:</b>.*?</tr>\s*<tr>\s*<td[^>]*></td>\s*<td[^>]*>(.*?)</td>',
        html, re.DOTALL
    )
    if m:
        result["category_description"] = clean(m.group(1))

    # Assigned To
    m = re.search(r'<b>Assigned To:</b>\s*</td>\s*<td[^>]*>(.*?)</td>', html, re.DOTALL)
    if m:
        result["assigned_to"] = clean(m.group(1))

    # Priority — inline in same <td>: <b>Priority:</b>&nbsp;&nbsp;C
    m = re.search(r'<b>Priority:</b>&nbsp;&nbsp;(\w)', html)
    if m:
        result["priority"] = m.group(1)

    # 311 Reference Number
    m = re.search(r'<b>311 Reference Number:</b>&nbsp;&nbsp;([\w-]+)', html)
    if m:
        result["ref_311"] = m.group(1)

    # Received date
    m = re.search(r'<b>Received:</b>\s*</td>\s*<td[^>]*>&nbsp;&nbsp;([\d/]+)', html)
    if m:
        result["received_date"] = m.group(1)

    # Block
    m = re.search(r'<b>Block:</b>&nbsp;&nbsp;(\w+)', html)
    if m:
        result["block"] = m.group(1)

    # Lot
    m = re.search(r'<b>Lot:</b>&nbsp;&nbsp;(\w+)', html)
    if m:
        result["lot"] = m.group(1)

    # Community Board
    m = re.search(r'<b>Community Board:</b>&nbsp;&nbsp;(\w+)', html)
    if m:
        result["community_board"] = m.group(1)

    # Owner — <b>Owner:</b></td> then <td> with &nbsp;&nbsp;VALUE
    m = re.search(r'<b>Owner:</b>\s*</td>\s*<td[^>]*>(.*?)</td>', html, re.DOTALL)
    if m:
        result["owner"] = clean(m.group(1))

    # Last Inspection
    m = re.search(
        r'<b>Last Inspection:</b>.*?</td>\s*<td[^>]*>(.*?)</td>',
        html, re.DOTALL
    )
    if m:
        text = clean(m.group(1))
        result["last_inspection"] = text
        badge = re.search(r'BADGE\s*#\s*(\d+)', text, re.IGNORECASE)
        if badge:
            result["inspector_badge"] = badge.group(1)

    # Disposition — <b>Disposition:</b> then value in next <td>
    m = re.search(
        r'<b>Disposition:</b>.*?</td>\s*<td[^>]*>(.*?)</td>',
        html, re.DOTALL
    )
    if m:
        result["disposition"] = clean(m.group(1))

    # ECB Violation #
    m = re.search(
        r'<b>ECB Violation.*?:</b>.*?</td>\s*<td[^>]*>(.*?)</td>',
        html, re.DOTALL
    )
    if m:
        text = clean(m.group(1))
        if text:
            result["ecb_violation"] = text

    # Comments — <b>Comments:</b> then value in next <td>
    m = re.search(
        r'<b>Comments:</b>.*?</td>\s*<td[^>]*>(.*?)</td>',
        html, re.DOTALL
    )
    if m:
        result["comments"] = clean(m.group(1))

    # BIN from property profile link
    m = re.search(r'bin=(\d+)', html)
    if m:
        result["bin"] = m.group(1)

    # Address
    m = re.search(r'Complaint at:&nbsp;&nbsp;(.*?)</td>', html)
    if m:
        result["address"] = clean(m.group(1))

    # Borough
    m = re.search(r'Borough:&nbsp;(\w[\w\s]*\w)', html)
    if m:
        result["borough"] = m.group(1).strip()

    # vlcompdetlkey in any links
    keys = re.findall(r'vlcompdetlkey=(\d+)', html)
    if keys:
        result["vlcompdetlkey_found"] = keys

    return result


# ── Step 3: Fetch BIS Web detail page ───────────────────────────────────────

def fetch_bis_detail(complaint_number: str) -> dict:
    """Fetch and parse a single BIS Web complaint detail page."""
    params = {"requestid": "0", "complaintno": complaint_number}
    try:
        resp = requests.get(
            BIS_WEB_URL, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
        return parse_bis_detail(resp.text), resp.text
    except requests.exceptions.Timeout:
        return {"error": "timeout"}, ""
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}, ""


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PROTOTYPE: DOB Complaint Direct Lookup Validation")
    print("=" * 70)

    # Step 1: Get some complaints from Open Data
    print("\n[1] Fetching recent complaints from NYC Open Data...")
    complaints = fetch_open_data_complaints(limit=5)
    print(f"    Got {len(complaints)} complaints\n")

    for i, c in enumerate(complaints):
        print(f"  Open Data row {i+1}:")
        print(f"    complaint_number: {c['complaint_number']}")
        print(f"    address: {c.get('house_number', '')} {c.get('house_street', '')}")
        print(f"    date_entered: {c.get('date_entered', '')}")
        print(f"    category: {c.get('complaint_category', '')}")
        print(f"    disposition_code: {c.get('disposition_code', '')}")
        print(f"    inspection_date: {c.get('inspection_date', '')}")
        print()

    # Step 2: Look up each on BIS Web
    print("[2] Looking up each complaint on BIS Web...")
    print("    (BIS Web is slow, ~60-120s per request)\n")

    for i, c in enumerate(complaints):
        cnum = c["complaint_number"]
        print(f"  --- Complaint {cnum} (request {i+1}/{len(complaints)}) ---")
        print(f"  Fetching BIS Web detail page...")

        start = time.time()
        detail, raw_html = fetch_bis_detail(cnum)
        elapsed = time.time() - start
        print(f"  Response time: {elapsed:.1f}s")

        if "error" in detail:
            print(f"  ERROR: {detail['error']}")
        else:
            print(f"  BIS complaint #: {detail.get('bis_complaint_number', 'NOT FOUND')}")
            print(f"  Status: {detail.get('bis_status', 'NOT FOUND')}")
            print(f"  Address: {detail.get('address', 'NOT FOUND')}")
            print(f"  Borough: {detail.get('borough', 'NOT FOUND')}")
            print(f"  BIN: {detail.get('bin', 'NOT FOUND')}")
            print(f"  Subject (Re:): {detail.get('subject', 'NOT FOUND')}")
            print(f"  Category: {detail.get('category_code', 'NOT FOUND')}")
            print(f"  Category desc: {detail.get('category_description', 'NOT FOUND')}")
            print(f"  Assigned to: {detail.get('assigned_to', 'NOT FOUND')}")
            print(f"  Priority: {detail.get('priority', 'NOT FOUND')}")
            print(f"  311 Ref #: {detail.get('ref_311', 'NOT FOUND')}")
            print(f"  Received: {detail.get('received_date', 'NOT FOUND')}")
            print(f"  Owner: {detail.get('owner', 'NOT FOUND')}")
            print(f"  Block/Lot: {detail.get('block', '?')}/{detail.get('lot', '?')}")
            print(f"  Community Board: {detail.get('community_board', 'NOT FOUND')}")
            print(f"  Last inspection: {detail.get('last_inspection', 'NOT FOUND')}")
            print(f"  Inspector badge: {detail.get('inspector_badge', 'NOT FOUND')}")
            print(f"  Disposition: {detail.get('disposition', 'NOT FOUND')}")
            print(f"  ECB violation: {detail.get('ecb_violation', 'NOT FOUND')}")
            print()
            print(f"  *** COMMENTS: {detail.get('comments', 'NOT FOUND')} ***")

        print()

        # Save first raw HTML for debugging
        if i == 0 and raw_html:
            with open("debug_bis_response.html", "w") as f:
                f.write(raw_html)
            print("  (Saved raw HTML to debug_bis_response.html)")
            print()

        # Be polite to the server
        if i < len(complaints) - 1:
            print("  Waiting 5s before next request...")
            time.sleep(5)

    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
