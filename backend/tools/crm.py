"""crm_profile tool (bonus): cross-session customer memory exposed to the LLM."""
import asyncio
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field

from backend.crm_store import CRMStore, CRMValidationError, PREFERENCE_KEYS, validate_fields
from .base import Tool, ToolContext, ToolResult


class CRMArgs(BaseModel):
    action: Literal["get", "update"]
    name: Optional[str] = Field(default=None, max_length=60)
    email: Optional[str] = Field(default=None, max_length=100)
    phone: Optional[str] = Field(default=None, max_length=30)
    preference_key: Optional[Literal["preferred_carrier", "favorite_category", "preferred_color",
                                     "preferred_size", "shipping_speed", "language"]] = None
    preference_value: Optional[str] = Field(default=None, max_length=60)
    note: Optional[str] = Field(default=None, max_length=200)


class CRMTool(Tool):
    name = "crm_profile"
    description = ("Reads or updates the current customer's saved profile (name, email, phone, preferences). "
                   "Use update when the customer states their name, contact details or a preference; "
                   "use get for 'what do you know about me'. Allowed preference_key values: "
                   + ", ".join(PREFERENCE_KEYS) + ".")
    args_model = CRMArgs
    timeout_s = 2.0
    timeout_message = "My customer notes did not respond in time. Please try again."
    gate_pattern = re.compile(
        r"\bmy name is\b|\bcall me\b|\bmy (?:e-?mail|phone|number)\b|\bi (?:prefer|like|love)\b|"
        r"\bwhat do you (?:know|have)\b.*\b(?:me|file)\b|\bon file\b|\bremember (?:me|my)\b", re.I)
    examples = [
        ("my name is Sheharyar and I prefer FedEx",
         {"action": "update", "name": "Sheharyar", "preference_key": "preferred_carrier",
          "preference_value": "FedEx"}),
        ("what do you have on file for me", {"action": "get"}),
    ]

    def __init__(self, store: CRMStore):
        self.store = store

    def fallback_arguments(self, text: str, ctx: ToolContext) -> Optional[dict]:
        if re.search(r"\bwhat do you (?:know|have)\b|\bon file\b", text, re.I):
            return {"action": "get"}
        args: dict = {"action": "update"}
        name = re.search(r"\b(?:my name is|call me)\s+([A-Za-z][A-Za-z'-]*(?: [A-Z][A-Za-z'-]*)?)", text)
        email = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
        phone = re.search(r"\+?\d[\d ()-]{6,18}\d", text)
        carrier = re.search(r"\bprefer\s+(fedex|ups|usps|dhl)\b", text, re.I)
        if name:
            args["name"] = name.group(1)
        if email:
            args["email"] = email.group(0)
        if phone:
            args["phone"] = phone.group(0)
        if carrier:
            args.update(preference_key="preferred_carrier", preference_value=carrier.group(1).upper())
        return args if len(args) > 1 else None

    async def run(self, args: CRMArgs, ctx: ToolContext) -> ToolResult:
        if args.action == "get":
            profile = await asyncio.to_thread(self.store.get, ctx.owner_id)
            return ToolResult(ok=True, data={"found": profile is not None, **(profile or {})})
        try:
            fields = validate_fields(args.name, args.email, args.phone, args.preference_key, args.preference_value)
        except CRMValidationError as error:
            return ToolResult(ok=False, error_code=error.code, user_message=error.message)
        if not fields:
            return ToolResult(ok=False, error_code="nothing_to_save", user_message="I didn't catch anything to save.")
        saved = await asyncio.to_thread(self.store.update, ctx.owner_id, fields)
        return ToolResult(ok=True, data={"saved": saved})
