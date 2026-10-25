"""Deterministic date/time resolution used by support_calendar as a validator and fallback."""
import re
from datetime import date, timedelta
from typing import Optional

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def resolve_date(text: str, today: date) -> Optional[date]:
    """Resolve an ISO date, 'today', 'tomorrow', '<weekday>' or 'next <weekday>'."""
    lowered = text.lower()
    iso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", lowered)
    if iso:
        try:
            return date(int(iso[1]), int(iso[2]), int(iso[3]))
        except ValueError:
            return None
    if re.search(r"\bday after tomorrow\b", lowered):
        return today + timedelta(days=2)
    if re.search(r"\btomorrow\b", lowered):
        return today + timedelta(days=1)
    if re.search(r"\btoday\b", lowered):
        return today
    named = re.search(r"\b(?:(next)\s+)?(" + "|".join(WEEKDAYS) + r")\b", lowered)
    if named:
        ahead = (WEEKDAYS.index(named[2]) - today.weekday()) % 7
        if ahead == 0 or named[1]:
            ahead = ahead or 7
        return today + timedelta(days=ahead)
    return None


def resolve_time(text: str) -> Optional[str]:
    """Resolve '15:00', '3 pm', '3:30 pm' or '9 am' to 24-hour HH:MM."""
    lowered = text.lower()
    match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)\b", lowered)
    if match:
        hour, minute = int(match[1]), int(match[2] or 0)
        if not 1 <= hour <= 12 or minute > 59:
            return None
        if match[3].startswith("p") and hour != 12:
            hour += 12
        if match[3].startswith("a") and hour == 12:
            hour = 0
        return f"{hour:02d}:{minute:02d}"
    match = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", lowered)
    if match:
        return f"{int(match[1]):02d}:{match[2]}"
    return None
