"""Parse BIS Web complaint detail HTML into structured data."""

import re


def clean(text: str) -> str:
    """Strip HTML tags, decode entities, and normalize whitespace."""
    text = re.sub(r'<[^>]+>', '', text)
    text = (text
            .replace('&nbsp;', ' ')
            .replace('&amp;', '&')
            .replace('&#039;', "'")
            .replace('&lt;', '<')
            .replace('&gt;', '>'))
    return ' '.join(text.split()).strip()


def parse_bis_detail(html: str) -> dict:
    """Extract all fields from a BIS Web complaint detail page.

    Returns a dict of parsed fields. Always includes '_parse_status':
      - 'ok' if a complaint was found and parsed
      - 'not_found' if the page indicates no record
      - 'error' with '_parse_error' explaining the issue
    """
    # Detect error/empty pages
    if 'ALL KEYS CANNOT BE BLANK' in html:
        return {"_parse_status": "not_found", "_parse_error": "blank keys error"}
    if 'No record found' in html or 'no record found' in html.lower():
        return {"_parse_status": "not_found", "_parse_error": "no record found"}
    if '<title>' not in html or 'Overview for Complaint' not in html:
        return {"_parse_status": "error", "_parse_error": "unexpected page format"}

    result = {"_parse_status": "ok"}

    # Complaint number + status from title tag
    m = re.search(r'<title>Overview for Complaint #:(\S+)\s*=\s*(\w+)</title>', html)
    if m:
        result["bis_complaint_number"] = m.group(1)
        result["bis_status"] = m.group(2)
    else:
        result["_parse_status"] = "error"
        result["_parse_error"] = "could not parse complaint number from title"
        return result

    # Subject (Re:) — the original complaint narrative
    m = re.search(r'Re:&nbsp;&nbsp;(.*?)</td>', html, re.DOTALL)
    if m:
        result["subject"] = clean(m.group(1))

    # Category Code + description
    m = re.search(r'<b>Category Code:</b>\s*</td>\s*<td[^>]*>(.*?)</td>', html, re.DOTALL)
    if m:
        result["category_code"] = clean(m.group(1))

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

    # Priority (inline: <b>Priority:</b>&nbsp;&nbsp;C)
    m = re.search(r'<b>Priority:</b>&nbsp;&nbsp;(\w)', html)
    if m:
        result["priority"] = m.group(1)

    # 311 Reference Number
    m = re.search(r'<b>311 Reference Number:</b>&nbsp;&nbsp;([\w-]+)', html)
    if m:
        result["ref_311"] = m.group(1)

    # Received date + time
    m = re.search(r'<b>Received:</b>\s*</td>\s*<td[^>]*>&nbsp;&nbsp;([\d/]+)(?:&nbsp;)*\s*([\d:]+)?', html)
    if m:
        result["received_date"] = m.group(1)
        if m.group(2):
            result["received_time"] = m.group(2)

    # Block / Lot (inline)
    m = re.search(r'<b>Block:</b>&nbsp;&nbsp;(\w+)', html)
    if m:
        result["block"] = m.group(1)

    m = re.search(r'<b>Lot:</b>&nbsp;&nbsp;(\w+)', html)
    if m:
        result["lot"] = m.group(1)

    # Owner
    m = re.search(r'<b>Owner:</b>\s*</td>\s*<td[^>]*>(.*?)</td>', html, re.DOTALL)
    if m:
        result["owner"] = clean(m.group(1))

    # Last Inspection + inspector badge
    m = re.search(
        r'<b>Last Inspection:</b>.*?</td>\s*<td[^>]*>(.*?)</td>',
        html, re.DOTALL
    )
    if m:
        text = clean(m.group(1))
        result["last_inspection"] = text
        badge = re.search(r'BADGE\s*#\s*(\d+)', text)
        if badge:
            result["inspector_badge"] = badge.group(1)

    # Disposition (full text: "03/11/2026 - I2 - NO VIOLATION WARRANTED...")
    m = re.search(
        r'<b>Disposition:</b>.*?</td>\s*<td[^>]*>(.*?)</td>',
        html, re.DOTALL
    )
    if m:
        result["disposition"] = clean(m.group(1))

    # ECB Violation number(s)
    m = re.search(
        r'<b>ECB Violation.*?:</b>.*?</td>\s*<td[^>]*>(.*?)</td>',
        html, re.DOTALL
    )
    if m:
        text = clean(m.group(1))
        if text:
            result["ecb_violation"] = text

    # Comments — inspector's narrative
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

    return result
