import operator
from dataclasses import dataclass, field
from typing_extensions import Annotated

@dataclass(kw_only=True)
class SummaryState:
    question_da: str = field(default=None) # Report topic in Danish
    question_en: str = field(default=None) # Report topic in English
    research_topic_en: str = field(default=None) 
    search_query_da: str = field(default=None) # Search query
    search_results_da: Annotated[list, operator.add] = field(default_factory=list)
    web_research_results_en: Annotated[list, operator.add] = field(default_factory=list) 
    sources_gathered: Annotated[list, operator.add] = field(default_factory=list)
    saved_frist_result_en: Annotated[list, operator.add] = field(default_factory=list)
    follow_up_query_en: str = field(default=None)
    research_loop_count: int = field(default=0) # Research loop count
    running_summary_en: str = field(default=None)  # summary in English
    question_answered_en: str = field(default=None)  # Final answer in English
    question_answered_da: str = field(default=None)  # Final answer in Danish

@dataclass(kw_only=True)
class SummaryStateInput:
    question_da: str = field(default=None) # Report topic     

@dataclass(kw_only=True)
class SummaryStateOutput:
    question_answered_da: str = field(default=None) # Final report