"""Research agent state definitions for LangGraph."""

from typing import TypedDict, List, Dict, Any, Literal, Annotated
from langchain_core.messages import BaseMessage

from langgraph.graph import add_messages


class SubQuestion(TypedDict):
    """A sub-question for research."""
    id: str
    question: str
    status: Literal["pending", "researching", "completed"]
    answer: str
    sources: List[str]


class Source(TypedDict):
    """A source for research."""
    url: str
    title: str
    content: str
    citations: List[str]


class ResearchState(TypedDict):
    """State for the ReAct research agent.

    This is a simplified state compared to the original 19-field state.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    sub_questions: List[SubQuestion]
    sources: List[Source]
    final_report: str
    current_query: str


# Legacy state definition (for backward compatibility during migration)
class LegacyResearchAgentState(TypedDict):
    """Original state definition - kept for migration reference."""
    project_id: str
    user_query: str
    research_plan: str
    search_queries: List[str]
    extracted_urls_from_search: List[str]
    scraped_data: Dict[str, str]
    processed_data: Dict[str, str]
    citations: Dict[str, Any]
    quantitative_data: List[Dict[str, Any]]
    data_validation: Dict[str, Any]
    statistical_results: Dict[str, Any]
    comparative_analysis: Dict[str, Any]
    qualitative_insights: str
    charts_and_tables: Dict[str, List[str]]
    final_report_markdown: str
    current_project_dir: str
    current_node_message: str
    messages: List[Dict[str, Any]]
    start_time: float
