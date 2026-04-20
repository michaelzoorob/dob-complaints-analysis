"""
Verify that Open Data complaint_number joins correctly to BIS Web
by cross-checking ALL overlapping fields between the two sources.
"""

import requests
import time
import re
import sys

sys.path.insert(0, ".")
from prototype_scrape import fetch_open_data_complaints, fetch_bis_detail

def extract_date(text):
    """Pull first MM/DD/YYYY date from a string."""
    m = re.search(r'(\d{2}/\d{2}/\d{4})', text or "")
    return m.group(1) if m else None

def extract_disp_code(text):
    """Pull disposition code like 'I2' or 'A8' from BIS disposition string."""
    m = re.search(r'- (\w\d) -', text or "")
    return m.group(1) if m else None

def main():
    print("Fetching 10 recent closed complaints from Open Data...\n")
    complaints = fetch_open_data_complaints(limit=10)

    print(f"{'COMPLAINT #':<14} {'FIELD':<20} {'OPEN DATA':<20} {'BIS WEB':<20} {'MATCH?'}")
    print("-" * 90)

    match_count = 0
    total_checks = 0

    for c in complaints:
        cnum = c["complaint_number"]
        detail, _ = fetch_bis_detail(cnum)

        if "error" in detail:
            print(f"{cnum:<14} ** ERROR: {detail['error']} **")
            continue

        # Check each overlapping field
        checks = [
            ("complaint_number", cnum, detail.get("bis_complaint_number")),
            ("bin", c.get("bin"), detail.get("bin")),
            ("community_board", c.get("community_board"), detail.get("community_board")),
            ("inspection_date", c.get("inspection_date"), extract_date(detail.get("last_inspection", ""))),
            ("disposition_code", c.get("disposition_code"), extract_disp_code(detail.get("disposition", ""))),
            ("disposition_date", c.get("disposition_date"), extract_date(detail.get("disposition", ""))),
        ]

        for field, od_val, bis_val in checks:
            match = od_val == bis_val
            marker = "YES" if match else "NO"
            if match:
                match_count += 1
            total_checks += 1
            print(f"{cnum:<14} {field:<20} {str(od_val):<20} {str(bis_val):<20} {marker}")

        # Show the bonus fields we're getting
        comments = detail.get("comments", "")
        subject = detail.get("subject", "")
        print(f"{cnum:<14} {'>> comments':<20} {comments[:60]}...")
        print(f"{cnum:<14} {'>> subject':<20} {subject[:60]}..." if subject else "")
        print()

        time.sleep(2)

    print("=" * 90)
    print(f"TOTAL FIELD CHECKS: {total_checks}")
    print(f"MATCHES: {match_count}/{total_checks} ({100*match_count/total_checks:.0f}%)")

if __name__ == "__main__":
    main()
