import pytest
from datetime import datetime, timedelta
from backend.tools.base import ToolContext
from backend.tools.calendar_tool import CalendarTool, CalendarArgs


@pytest.fixture
def calendar_file(tmp_path):
    return tmp_path / "calendar_test.json"


@pytest.mark.asyncio
async def test_calendar_schedule_and_list(calendar_file):
    tool = CalendarTool(path=calendar_file)
    # Target next Wednesday at 14:00
    now = datetime(2026, 10, 26, 10, 0)  # Monday
    target_date = (now + timedelta(days=2)).strftime("%Y-%m-%d")  # Wednesday
    ctx1 = ToolContext(owner_id="user_a", session_id="s1", now=now)

    res = await tool.run(CalendarArgs(action="schedule", title="Callback for order", date=target_date, time="14:00"), ctx1)
    assert res.ok is True
    assert "recorded" in res.data
    event_id = res.data["recorded"]["event_id"]

    # List appointments
    list_res = await tool.run(CalendarArgs(action="list"), ctx1)
    assert list_res.ok is True
    assert list_res.data["count"] == 1
    assert list_res.data["events"][0]["event_id"] == event_id


@pytest.mark.asyncio
async def test_calendar_owner_isolation(calendar_file):
    tool = CalendarTool(path=calendar_file)
    now = datetime(2026, 10, 26, 10, 0)
    target_date = (now + timedelta(days=2)).strftime("%Y-%m-%d")
    ctx_a = ToolContext(owner_id="user_a", session_id="sa", now=now)
    ctx_b = ToolContext(owner_id="user_b", session_id="sb", now=now)

    # User A schedules
    res = await tool.run(CalendarArgs(action="schedule", date=target_date, time="11:00"), ctx_a)
    event_id = res.data["recorded"]["event_id"]

    # User B lists - sees 0
    list_b = await tool.run(CalendarArgs(action="list"), ctx_b)
    assert list_b.data["count"] == 0

    # User B cannot cancel User A's event
    cancel_b = await tool.run(CalendarArgs(action="cancel", event_id=event_id), ctx_b)
    assert cancel_b.ok is False
    assert cancel_b.error_code == "not_found"


@pytest.mark.asyncio
async def test_calendar_outside_hours(calendar_file):
    tool = CalendarTool(path=calendar_file)
    now = datetime(2026, 10, 26, 10, 0)
    target_date = (now + timedelta(days=2)).strftime("%Y-%m-%d")
    ctx = ToolContext(owner_id="user_a", session_id="s1", now=now)

    # 18:00 is outside 09:00-17:00
    res = await tool.run(CalendarArgs(action="schedule", date=target_date, time="18:00"), ctx)
    assert res.ok is False
    assert res.error_code == "outside_hours"


@pytest.mark.asyncio
async def test_calendar_past_date(calendar_file):
    tool = CalendarTool(path=calendar_file)
    now = datetime(2026, 10, 26, 10, 0)
    past_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    ctx = ToolContext(owner_id="user_a", session_id="s1", now=now)

    res = await tool.run(CalendarArgs(action="schedule", date=past_date, time="10:00"), ctx)
    assert res.ok is False
    assert res.error_code == "past_date"


@pytest.mark.asyncio
async def test_calendar_conflict(calendar_file):
    tool = CalendarTool(path=calendar_file)
    now = datetime(2026, 10, 26, 10, 0)
    target_date = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    ctx = ToolContext(owner_id="user_a", session_id="s1", now=now)

    res1 = await tool.run(CalendarArgs(action="schedule", date=target_date, time="10:00", duration_minutes=30), ctx)
    assert res1.ok is True

    # Try overlapping slot (10:15)
    res2 = await tool.run(CalendarArgs(action="schedule", date=target_date, time="10:15", duration_minutes=30), ctx)
    assert res2.ok is False
    assert res2.error_code == "conflict"
