import pytest
from backend.crm_store import CRMStore, CRMValidationError, validate_fields, welcome_line
from backend.tools.base import ToolContext
from backend.tools.crm import CRMTool, CRMArgs


@pytest.fixture
def crm_store(tmp_path):
    return CRMStore(db_path=tmp_path / "crm_test.db")


def test_crm_validate_fields_ok():
    fields = validate_fields(name="Sheharyar Ali", email="test@example.com", phone="+1234567890",
                             preference_key="preferred_carrier", preference_value="FedEx")
    assert fields["name"] == "Sheharyar Ali"
    assert fields["email"] == "test@example.com"
    assert fields["preference"] == ("preferred_carrier", "FedEx")


def test_crm_reject_credit_card():
    with pytest.raises(CRMValidationError) as exc:
        validate_fields(name="John Doe", preference_value="My card is 4532 1234 5678 9010")
    assert exc.value.code == "sensitive_data"


def test_crm_reject_password():
    with pytest.raises(CRMValidationError) as exc:
        validate_fields(name="password is secret123")
    assert exc.value.code == "sensitive_data"


def test_crm_owner_isolation(crm_store):
    crm_store.update("user_1", {"name": "Alice"})
    crm_store.update("user_2", {"name": "Bob"})

    p1 = crm_store.get("user_1")
    p2 = crm_store.get("user_2")
    assert p1["name"] == "Alice"
    assert p2["name"] == "Bob"


def test_welcome_line():
    assert welcome_line(None) == ""
    assert welcome_line({}) == ""

    profile1 = {"name": "Sheharyar", "last_interaction": {"order_id": "ORD-10002"}}
    line1 = welcome_line(profile1)
    assert "Welcome back, Sheharyar!" in line1
    assert "ORD-10002" in line1

    profile2 = {"name": "Qadeer", "last_interaction": {"sku": "ELEC-001"}}
    line2 = welcome_line(profile2)
    assert "Welcome back, Qadeer!" in line2
    assert "ELEC-001" in line2


@pytest.mark.asyncio
async def test_crm_tool_run(crm_store):
    tool = CRMTool(crm_store)
    ctx = ToolContext(owner_id="user_test", session_id="s1")

    # Update profile
    up_res = await tool.run(CRMArgs(action="update", name="Sheharyar", preference_key="preferred_carrier",
                                    preference_value="FedEx"), ctx)
    assert up_res.ok is True

    # Get profile
    get_res = await tool.run(CRMArgs(action="get"), ctx)
    assert get_res.ok is True
    assert get_res.data["found"] is True
    assert get_res.data["name"] == "Sheharyar"
    assert get_res.data["preferences"]["preferred_carrier"] == "FedEx"
