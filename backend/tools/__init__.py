"""Tool framework: tools, registry and orchestrator."""
from backend.crm_store import CRMStore
from .calculator import CalculatorTool
from .calendar_tool import CalendarTool
from .catalog import CatalogTool
from .crm import CRMTool
from .orchestrator import ToolOrchestrator
from .registry import ToolRegistry


def build_orchestrator(llm, crm_store: CRMStore) -> ToolOrchestrator:
    """Create the orchestrator with the three required tools plus the bonus CRM tool."""
    registry = ToolRegistry([CalculatorTool(), CatalogTool(), CalendarTool(), CRMTool(crm_store)])
    return ToolOrchestrator(registry, llm)
