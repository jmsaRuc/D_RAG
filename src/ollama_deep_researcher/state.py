import operator
from dataclasses import dataclass, field
from typing_extensions import Annotated

@dataclass(kw_only=True)
class SummaryState:
    research_topic: str = field(default=None) # Report topic
    research_topic_en: str = field(default=None)
    research_topic_da: str = field(default=None) 
    search_query_en: str = field(default=None) # Search query
    search_results_da: str = field(default=None)
    web_research_results_en: Annotated[list, operator.add] = field(default_factory=list) 
    sources_gathered: Annotated[list, operator.add] = field(default_factory=list)
    saved_frist_result_de: Annotated[list, operator.add] = field(default_factory=list)
    research_loop_count: int = field(default=0) # Research loop count
    running_summary: str = field(default=None) # Final report

@dataclass(kw_only=True)
class SummaryStateInput:
    research_topic: str = field(default=None) # Report topic     

@dataclass(kw_only=True)
class SummaryStateOutput:
    running_summary: str = field(default=None) # Final report