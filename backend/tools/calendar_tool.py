"""support_calendar tool: local reminders and support appointment slots. Notifies nobody."""
import asyncio
import json
import re
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from backend.config import CALENDAR_FILE
from .base import Tool, ToolContext, ToolResult
from .dates import resolve_date, resolve_time

OPEN_TIME, CLOSE_TIME = "09:00", "17:00"
MAX_DAYS_AHEAD = 90
_lock = threading.Lock()


class CalendarArgs(BaseModel):
    action: Literal["schedule", "list", "cancel"]
    title: Optional[str] = Field(default=None, max_length=80, description="Short event title")
    date: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="YYYY-MM-DD")
    time: Optional[str] = Field(default=None, pattern=r"^\d{2}:\d{2}$", description="HH:MM 24-hour")
    duration_minutes: int = Field(default=15, ge=15, le=60)
    event_id: Optional[str] = Field(default=None, max_length=20)
    related_order_id: Optional[str] = Field(default=None, pattern=r"^ORD-\d{5}$")
    notes: Optional[str] = Field(default=None, max_length=200)


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("events", [])
    except (json.JSONDecodeError, OSError):
        return []


def _write(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"events": events}, indent=2), encoding="utf-8")
    for attempt in range(5):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.01 * (attempt + 1))


def _minutes(clock: str) -> int:
    return int(clock[:2]) * 60 + int(clock[3:])


def _error(code: str, message: str) -> ToolResult:
    return ToolResult(ok=False, error_code=code, user_message=message)


def _public(event: dict) -> dict:
    keys = ("event_id", "title", "date", "time", "duration_minutes", "related_order_id", "notes")
    return {key: event.get(key) for key in keys if event.get(key) is not None}


def _schedule(path: Path, args: CalendarArgs, ctx: ToolContext) -> ToolResult:
    if not args.date or not args.time:
        return _error("missing_details", "I need both a date and a time to record a slot.")
    try:
        start = datetime.strptime(f"{args.date} {args.time}", "%Y-%m-%d %H:%M")
    except ValueError:
        return _error("invalid_time", "That date or time is not valid.")
    if start < ctx.now:
        return _error("past_date", "That date and time is in the past.")
    if start.date() > ctx.now.date() + timedelta(days=MAX_DAYS_AHEAD):
        return _error("too_far", f"Slots can only be recorded up to {MAX_DAYS_AHEAD} days ahead.")
    end_minutes = _minutes(args.time) + args.duration_minutes
    if start.weekday() > 4 or _minutes(args.time) < _minutes(OPEN_TIME) or end_minutes > _minutes(CLOSE_TIME):
        return _error("outside_hours", "Support slots are Monday to Friday, 09:00 to 17:00.")
    with _lock:
        events = _read(path)
        for other in events:
            if other["owner_id"] == ctx.owner_id and other["date"] == args.date:
                other_start = _minutes(other["time"])
                if _minutes(args.time) < other_start + other["duration_minutes"] and other_start < end_minutes:
                    return _error("conflict", "That overlaps with another event you already have.")
        related = args.related_order_id
        event = {"event_id": "EVT-" + uuid.uuid4().hex[:6].upper(), "owner_id": ctx.owner_id,
                 "title": args.title or (f"Support callback for {related}" if related else "Support appointment"),
                 "date": args.date, "time": args.time, "duration_minutes": args.duration_minutes,
                 "related_order_id": related, "notes": args.notes}
        _write(path, events + [event])
    return ToolResult(ok=True, data={
        "recorded": _public(event),
        "note": "Recorded locally only. Nobody was contacted; a human must confirm the slot."})


def _list(path: Path, ctx: ToolContext) -> ToolResult:
    today = ctx.now.date().isoformat()
    with _lock:
        mine = [e for e in _read(path) if e["owner_id"] == ctx.owner_id and e["date"] >= today]
    mine.sort(key=lambda e: (e["date"], e["time"]))
    return ToolResult(ok=True, data={"count": len(mine), "events": [_public(e) for e in mine[:10]]})


def _cancel(path: Path, args: CalendarArgs, ctx: ToolContext) -> ToolResult:
    today = ctx.now.date().isoformat()
    with _lock:
        events = _read(path)
        mine = [e for e in events if e["owner_id"] == ctx.owner_id and e["date"] >= today]
        if args.event_id:
            target = [e for e in mine if e["event_id"] == args.event_id.upper()]
        elif args.date and args.time:
            target = [e for e in mine if e["date"] == args.date and e["time"] == args.time]
        else:
            target = mine if len(mine) == 1 else []
        if not target:
            return _error("not_found", "I could not find a matching event of yours to cancel.")
        _write(path, [e for e in events if e["event_id"] != target[0]["event_id"]])
    return ToolResult(ok=True, data={"cancelled": _public(target[0])})


class CalendarTool(Tool):
    name = "support_calendar"
    description = ("Records and lists reminders and support appointment slots in a local calendar file "
                   "for the current customer. It does not notify or contact anyone. "
                   "Use for remind me / schedule / callback / appointment / cancel / what is on my calendar.")
    args_model = CalendarArgs
    timeout_s = 2.0
    timeout_message = "The calendar did not respond in time. Please try again."
    gate_pattern = re.compile(
        r"\b(?:remind(?:er)?|schedule|appointment|callback|call back|book a (?:slot|call)|calendar|"
        r"cancel (?:my |the )?(?:appointment|reminder|slot|callback|event))\b", re.I)
    examples = [
        ("schedule a support callback tomorrow at 3 pm for order ORD-10002",
         {"action": "schedule", "title": "Support callback for ORD-10002", "date": "<tomorrow>",
          "time": "15:00", "related_order_id": "ORD-10002"}),
        ("what is on my calendar", {"action": "list"}),
    ]

    def __init__(self, path: Path = CALENDAR_FILE):
        self.path = Path(path)

    def fallback_arguments(self, text: str, ctx: ToolContext) -> Optional[dict]:
        lowered = text.lower()
        if re.search(r"\bcancel\b", lowered):
            args: dict = {"action": "cancel"}
        elif re.search(r"\b(?:what(?:'s| is)? on|list|show|upcoming|my calendar)\b", lowered) \
                and not re.search(r"\b(?:schedule|remind|book)\b", lowered):
            return {"action": "list"}
        else:
            args = {"action": "schedule"}
            order = re.search(r"\bORD-?\s?(\d{5})\b", text, re.I)
            if order:
                args["related_order_id"] = f"ORD-{order[1]}"
            kind = "Support callback" if re.search(r"call", lowered) else (
                "Reminder" if "remind" in lowered else "Support appointment")
            args["title"] = f"{kind} for {args['related_order_id']}" if order else kind
        day = resolve_date(text, ctx.now.date())
        clock = resolve_time(text)
        if day:
            args["date"] = day.isoformat()
        if clock:
            args["time"] = clock
        return args

    async def run(self, args: CalendarArgs, ctx: ToolContext) -> ToolResult:
        if args.action == "schedule":
            return await asyncio.to_thread(_schedule, self.path, args, ctx)
        if args.action == "cancel":
            return await asyncio.to_thread(_cancel, self.path, args, ctx)
        return await asyncio.to_thread(_list, self.path, ctx)
