import operator
from dataclasses import dataclass, field
from typing import List

from typing_extensions import Annotated, Any, Dict


@dataclass(kw_only=True)
class SummaryState:
    """
    A container for managing and tracking the state of a research summary.

    Including queries, research topics, results.
    from both Danish and English sources, and other related information.
    """

    # Report topic in Danish and English
    question_da: str | None = field(default=None)
    question_en: str | None = field(default=None)

    # Research topic in English
    research_topic_en: str | None = field(default=None)

    # Research query in English
    search_query_da: str | None = field(default=None)

    # Danish results from retsinfo crawling
    search_results_da: Annotated[List[Dict[str, Any]], operator.add] = field(
        default_factory=List,
    )

    # crawled retsinfo results, translated to English
    web_research_results_en: Annotated[List[str], operator.add] = field(
        default_factory=List,
    )

    # List of sources gathered during the research
    sources_gathered: Annotated[List[str], operator.add] = field(default_factory=List)

    # fist saved results in Danish
    saved_frist_result_da: Annotated[List[str], operator.add] = field(
        default_factory=List,
    )

    # follow up query in English
    follow_up_query_en: str | None = field(default=None)

    # running summary in English
    running_summary_en: str | None = field(default=None)

    # question answered in English and Danish
    question_answered_en: str | None = field(default=None)
    question_answered_da: str | None = field(default=None)

    # research loop count
    research_loop_count: int = field(default=0)


@dataclass(kw_only=True)
class SummaryStateInput:
    """
    A minimal data structure to store the input question in Danish.

    Attributes:
        question_da (str | None): The input question provided in Danish.
    """

    # input question in Danish
    question_da: str | None = field(default=None)


@dataclass(kw_only=True)
class SummaryStateOutput:
    """A minimal data structure to store the the output answer in Danish."""

    # output question in Danish
    question_answered_da: str | None = field(default=None)
