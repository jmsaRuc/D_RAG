import json

from typing_extensions import Literal, Optional, Union
from typing import Dict, List, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph

from ollama_deep_researcher.configuration import Configuration, SearchAPI
from ollama_deep_researcher.utils import strip_thinking_tokens, format_sources
from ollama_deep_researcher.translate_async import (
    deduplicate_translate_and_format_sources,
)
from ollama_deep_researcher.retsinfo_crawl import (
    retsinfo_search_and_crawl,
)
from ollama_deep_researcher.state import (
    SummaryState,
    SummaryStateInput,
    SummaryStateOutput,
)
from ollama_deep_researcher.prompts import (
    summarizer_instructions,
    reflection_instructions,
    query_writer_instructions_with_tag,
    translate_texts_whith_ex_instructions,
    research_topic_write_instructions,
    translate_qustion_instructions,
    final_answer_instructions,
    get_current_date,
)


# Nodes
async def translate_question(
    state: SummaryState, config: RunnableConfig
) -> SummaryState:
    """
    Translate a Danish research topic into English using the configured LLM provider.
    Parameters:
        state (SummaryState): An object containing the Danish research topic.
        config (RunnableConfig): Configuration for the LLM, including provider, model parameters, and settings.
    Returns:
        Dict[str, str]: A dictionary with:
            - "research_topic_en": The original Danish research topic.
            - "research_topic_da": The translated English research topic.
    """

    # Insert the Danish reasearch topic into the prompt
    human_message_content = f"Translate the following qustion from Danish to English. \n <User Input> \n {state.question_da} \n <User Input>\n\n"

    # Configure
    configurable = Configuration.from_runnable_config(config)

    # Choose the appropriate LLM based on the provider
    if configurable.llm_provider == "groq":
        llm_translate_q = ChatGroq(
            model=configurable.groq_llm,
            temperature=0,
            max_tokens=12000,
            service_tier="auto",
        )
    else:  # Default to Ollama
        llm_translate_q = ChatOllama(
            model=configurable.local_llm,
            temperature=0,
            num_predict=12000,
        )

    result = await llm_translate_q.ainvoke(
        [
            SystemMessage(content=translate_qustion_instructions),
            HumanMessage(content=human_message_content),
        ]
    )

    # Strip thinking tokens
    if configurable.strip_thinking_tokens:
        translated_question = await strip_thinking_tokens(result.content)

    return {
        "question_da": state.question_da,
        "question_en": translated_question,
    }


async def generate_research_topic(
    state: SummaryState, config: RunnableConfig
) -> SummaryState:

    formattet_prompt = research_topic_write_instructions.format(
        question=state.question_en,
    )

    # Configure
    configurable = Configuration.from_runnable_config(config)

    # Choose the appropriate LLM based on the provider
    if configurable.llm_provider == "groq":
        llm_genrate_research_topic = ChatGroq(
            model=configurable.groq_llm,
            temperature=0,
            max_tokens=34000,
            service_tier="auto",
            response_format={"type": "json_object"},
        )
    else:  # Default to Ollama
        llm_genrate_research_topic = ChatOllama(
            base_url=configurable.ollama_base_url,
            model=configurable.local_llm,
            temperature=0,
            num_predict=34000,
            format="json",
        )

    result = await llm_genrate_research_topic.ainvoke(
        [
            SystemMessage(content=formattet_prompt),
            HumanMessage(
                content=f"Reflect on the question, identify the main topic, and knowledge gaps, and generate questions for further research (Format your response as a JSON object):"
            ),
        ]
    )

    # Get the content
    content = result.content

    # Parse the JSON response and get the query
    try:
        follow_up_questions = json.loads(content)
        research_topic_en = follow_up_questions["follow_up_questions"]
        if isinstance(research_topic_en, list):
            # If the response is a list, join it into a single string
            research_topic_en = " ".join(research_topic_en)
    except (json.JSONDecodeError, KeyError):
        # If parsing fails or the key is not found, use a fallback query
        print(f"Failed to parse JSON: {content}")
        if configurable.strip_thinking_tokens:
            content = await strip_thinking_tokens(content)
        research_topic_en = content
    return {"research_topic_en": research_topic_en}


async def generate_query(state: SummaryState, config: RunnableConfig) -> SummaryState:
    """LangGraph node that generates a search query based on the research topic.

    Uses an LLM to create an optimized search query for web research based on
    the user's research topic. Supports both Groq and Ollama as LLM providers.

    Args:
        state: Current graph state containing the research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query_en key containing the generated query
    """

    # Format the prompt
    current_date = await get_current_date()
    formatted_prompt = query_writer_instructions_with_tag.format(
        current_date=current_date,
        research_topic_en=state.research_topic_en,
        question_en=state.question_en,
        question_da=state.question_da,
    )

    # Configure
    configurable = Configuration.from_runnable_config(config)

    # Choose the appropriate LLM based on the provider
    if configurable.llm_provider == "groq":
        llm_json_mode = ChatGroq(
            model=configurable.groq_llm,
            temperature=0,
            max_tokens=12000,
            service_tier="auto",
            response_format={"type": "json_object"},
        )
    else:  # Default to Ollama
        llm_json_mode = ChatOllama(
            base_url=configurable.ollama_base_url,
            model=configurable.local_llm,
            temperature=0,
            num_predict=12000,
            format="json",
        )
    # Generate a query
    result = await llm_json_mode.ainvoke(
        [
            SystemMessage(content=formatted_prompt),
            HumanMessage(
                content=f"Generate a query in Danish for searching a law database:"
            ),
        ]
    )

    # Get the content
    content = result.content

    # Parse the JSON response and get the query
    try:
        query = json.loads(content)
        search_query_da = query["query"]
    except (json.JSONDecodeError, KeyError):
        # If parsing fails or the key is not found, use a fallback query
        print(f"Failed to parse JSON: {content}")
        if configurable.strip_thinking_tokens:
            content = await strip_thinking_tokens(content)
        search_query_da = content
    return {"search_query_da": search_query_da}


async def web_research(state: SummaryState) -> SummaryState:
    """LangGraph node that performs web research using the generated search query.

       Perform web research by executing a search and crawl using the 'retsinfo' service.
    Parameters:
        state (SummaryState): An object containing the search query and related state.
    Returns:
        Dict[str, list[Dict[str, Any]]]: A dictionary with the key 'search_results_en' mapping to a list containing the search results.
    """

    # Use the configured retsinfo search and crawl
    search_results_da = await retsinfo_search_and_crawl(state.search_query_da, 3, state)

    return {
        "search_results_da": [search_results_da],
    }


async def translate_search_results(
    state: SummaryState, config: RunnableConfig
) -> Dict[str, List[Any]]:

    latest_search_results_da = state.search_results_da[-1]
    configurable = Configuration.from_runnable_config(config)

    if configurable.llm_provider == "groq":
        llm_translate_s = ChatGroq(
            model=configurable.groq_llm,
            temperature=0,
            max_tokens=66000,
            service_tier="auto",
        )
    else:  # Default to Ollama
        llm_translate_s = ChatOllama(
            model=configurable.local_llm,
            temperature=0,
            num_predict=66000,
        )

    search_str_en, saved_first_en = await deduplicate_translate_and_format_sources(
        latest_search_results_da, llm_translate_s, 64000
    )

    return {
        "sources_gathered": [await format_sources(latest_search_results_da)],
        "research_loop_count": state.research_loop_count + 1,
        "web_research_results_en": [search_str_en],
        "saved_frist_result_en": [saved_first_en],
    }


async def summarize_sources(
    state: SummaryState, config: RunnableConfig
) -> SummaryState:
    """LangGraph node that summarizes web research results.

    Uses an LLM to create or update a running summary based on the newest web research
    results, integrating them with any existing summary.

    Args:
        state: Current graph state containing research topic, running summary,
              and web research results
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including running_summary_en key containing the updated summary
    """

    # Existing summary
    existing_summary = state.running_summary_en

    # Most recent web research
    most_recent_web_research = state.web_research_results_en[-1]

    # Build the human message
    if existing_summary:
        human_message_content = (
            f"<Existing Summary> \n {existing_summary} \n <Existing Summary>\n\n"
            f"<New Context> \n {most_recent_web_research} \n <New Context>"
            f"Update the Existing Summary with the New Context on this topic: \n <User Input> \n {state.research_topic_en} \n <User Input>\n\n"
        )
    else:
        human_message_content = (
            f"<Context> \n {most_recent_web_research} \n <Context>"
            f"Create a Summary using the Context on this topic: \n <User Input> \n {state.research_topic_en} \n <User Input>\n\n"
        )

    # Run the LLM
    configurable = Configuration.from_runnable_config(config)

    # Choose the appropriate LLM based on the provider
    if configurable.llm_provider == "groq":
        summarize_llm = ChatGroq(
            model=configurable.groq_llm,
            temperature=0,
            max_tokens=131072,
            service_tier="auto",
        )
    else:  # Default to Ollama
        summarize_llm = ChatOllama(
            base_url=configurable.ollama_base_url,
            model=configurable.local_llm,
            temperature=0,
            num_predict=64000,
        )

    result = await summarize_llm.ainvoke(
        [
            SystemMessage(content=summarizer_instructions),
            HumanMessage(content=human_message_content),
        ]
    )
    # check if the new summary is longer than the previous one

    # Strip thinking tokens if configured
    running_summary_en = result.content
    if configurable.strip_thinking_tokens:
        running_summary_en = await strip_thinking_tokens(running_summary_en)
    # check if the new summary is longer than the previous one
    if existing_summary and len(running_summary_en) < len(existing_summary):
        # If the new summary is shorter, keep the existing summary
        running_summary_en = existing_summary

    return {"running_summary_en": running_summary_en}


async def reflect_on_summary(
    state: SummaryState, config: RunnableConfig
) -> SummaryState:
    """LangGraph node that identifies knowledge gaps and generates follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    a new search query to address those gaps. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """

    # Generate a query
    configurable = Configuration.from_runnable_config(config)

    # Choose the appropriate LLM based on the provider
    if configurable.llm_provider == "groq":
        llm_json_mode_34k = ChatGroq(
            model=configurable.groq_llm,
            temperature=0,
            max_tokens=34000,
            response_format={"type": "json_object"},
        )
    else:  # Default to Ollama
        llm_json_mode_34k = ChatOllama(
            base_url=configurable.ollama_base_url,
            model=configurable.local_llm,
            temperature=0,
            num_predict=34000,
            format="json",
        )

    result = await llm_json_mode_34k.ainvoke(
        [
            SystemMessage(
                content=reflection_instructions.format(
                    research_topic=state.research_topic_en,
                    summery=state.running_summary_en,
                )
            ),
            HumanMessage(
                content=f"Reflect on our existing knowledge, identify a knowledge gap and generate a follow-up database search query:"
            ),
        ]
    )

    # Strip thinking tokens if configured
    try:
        # Try to parse as JSON first
        reflection_content = json.loads(result.content)
        # Get the follow-up query
        query = reflection_content.get("follow_up_query")
        # Check if query is None or empty
        if not query:
            # Use a fallback query
            raise ValueError("LLM failed to make a qurry")
        return {"follow_up_query_en": query}
    except (json.JSONDecodeError, KeyError, AttributeError) as e:
        # If parsing fails or the key is not found, use a fallback query
        print(f"{str(e)}")


async def translate_content_follow_up(
    state: SummaryState, config: RunnableConfig
) -> SummaryState:

    # set the char limit for the translation
    char_limit = 12000 * 3
    char_limit = round(char_limit)

    # Get the first search result in English
    try:
        recent_saved_first_en = state.saved_frist_result_en[0]
    except IndexError:
        recent_saved_first_en = state.question_en

    # Check if the saved first result is over char_limit
    test_content_len = recent_saved_first_en
    print(len(test_content_len))
    print(char_limit)
    if len(test_content_len) > char_limit:
        recent_saved_first_en = test_content_len[0:char_limit] + "... [truncated]"
        print(len(recent_saved_first_en))

    # Get the first search result in Danish
    try:
        first_search_result_da = state.search_results_da[0]["results"][0]["content"]
    except IndexError:
        first_search_result_da = state.question_da

    test_content_len = first_search_result_da
    print(len(test_content_len))
    print(char_limit)
    if len(test_content_len) > char_limit:
        first_search_result_d = test_content_len[0:char_limit] + "... [truncated]"
        print(len(first_search_result_d))

    # Format the prompt
    formatted_prompt = translate_texts_whith_ex_instructions.format(
        english_text_example=recent_saved_first_en,
        translated_to_danish_text_example=first_search_result_da,
    )

    # Prepare the human message content
    human_message_content = f"Translate the following text from English to Danish. \n <User Input> \n {state.follow_up_query_en} \n <User Input>\n\n"

    # Configure the LLM based on the provider
    configurable = Configuration.from_runnable_config(config)

    # Choose the appropriate LLM based on the provider
    if configurable.llm_provider == "groq":
        translate_follow_up_llm = ChatGroq(
            model=configurable.groq_llm,
            temperature=0,
            max_tokens=26000,
            service_tier="auto",
        )
    else:  # Default to Ollama
        translate_follow_up_llm = ChatOllama(
            base_url=configurable.ollama_base_url,
            model=configurable.local_llm,
            temperature=0,
            num_predict=26000,
        )

    result = await translate_follow_up_llm.ainvoke(
        [
            SystemMessage(content=formatted_prompt),
            HumanMessage(content=human_message_content),
        ]
    )

    if configurable.strip_thinking_tokens:
        translated_follow_up_qurry = await strip_thinking_tokens(result.content)

    return {"search_query_da": translated_follow_up_qurry}


async def generate_final_answer(
    state: SummaryState, config: RunnableConfig
) -> SummaryState:

    # Final summery in English
    final_summary_en = state.running_summary_en

    # original question in English
    question_en = state.question_en

    # Format the promt
    formatted_prompt = final_answer_instructions.format(
        summary=final_summary_en,
    )

    # human message content
    human_message_content = f"Think carefully about the provided Context first. Then generate a final answer to the user's question based on the summary provided, when using sources from the summary to answer, use title, chapter, paragraph, and clause (§=paragraph, stk.=clause), as citation: \n <User Input> \n {question_en} \n <User Input>\n\n"

    # Configure
    configurable = Configuration.from_runnable_config(config)

    # Choose the appropriate LLM based on the provider
    if configurable.llm_provider == "groq":
        translate_follow_up_llm = ChatGroq(
            model=configurable.groq_llm,
            temperature=0,
            max_tokens=34000,
            service_tier="auto",
        )
    else:  # Default to Ollama
        translate_follow_up_llm = ChatOllama(
            base_url=configurable.ollama_base_url,
            model=configurable.local_llm,
            temperature=0,
            num_predict=34000,
        )

    result = await translate_follow_up_llm.ainvoke(
        [
            SystemMessage(content=formatted_prompt),
            HumanMessage(content=human_message_content),
        ]
    )

    # Get the content
    content = result.content

    # Strip thinking tokens if configured
    question_answered_en = content
    if configurable.strip_thinking_tokens:
        question_answered_en = await strip_thinking_tokens(content)

    return {"question_answered_en": question_answered_en}


async def translate_answer(state: SummaryState, config: RunnableConfig) -> SummaryState:

    # Final answer in English
    char_limit = 12000 * 3
    char_limit = round(char_limit)

    # Get the first search result in English
    try:
        recent_saved_first_en = state.saved_frist_result_en[0]
    except IndexError:
        recent_saved_first_en = state.question_en

    # Check if the saved first result is over char_limit
    test_content_len = recent_saved_first_en
    print(len(test_content_len))
    print(char_limit)
    if len(test_content_len) > char_limit:
        recent_saved_first_en = test_content_len[0:char_limit] + "... [truncated]"
        print(len(recent_saved_first_en))

    # Get the first search result in Danish
    try:
        first_search_result_da = state.search_results_da[0]["results"][0]["content"]
    except IndexError:
        first_search_result_da = state.question_da

    # Check if first search is over char_limit
    test_content_len = first_search_result_da
    print(len(test_content_len))
    print(char_limit)
    if len(test_content_len) > char_limit:
        first_search_result_d = test_content_len[0:char_limit] + "... [truncated]"
        print(len(first_search_result_d))

    # Format the prompt
    formatted_prompt = translate_texts_whith_ex_instructions.format(
        english_text_example=recent_saved_first_en,
        translated_to_danish_text_example=first_search_result_da,
    )

    # Prepare the human message content
    human_message_content = f"Translate the following text from English to Danish. \n <User Input> \n {state.question_answered_en} \n <User Input>\n\n"

    # Configure the LLM based on the provider
    configurable = Configuration.from_runnable_config(config)

    if configurable.llm_provider == "groq":
        translate_follow_up_llm = ChatGroq(
            model=configurable.groq_llm,
            temperature=0,
            max_tokens=26000,
            service_tier="auto",
        )
    else:  # Default to Ollama
        translate_follow_up_llm = ChatOllama(
            base_url=configurable.ollama_base_url,
            model=configurable.local_llm,
            temperature=0,
            num_predict=26000,
        )

    result = await translate_follow_up_llm.ainvoke(
        [
            SystemMessage(content=formatted_prompt),
            HumanMessage(content=human_message_content),
        ]
    )
    translated_summery = result.content
    if configurable.strip_thinking_tokens:
        translated_summery = await strip_thinking_tokens(result.content)

    return {"question_answered_da": translated_summery}


async def finalize_answer(state: SummaryState) -> SummaryState:
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including question_answered_da key containing the formatted final summary with sources
    """

    # Deduplicate sources before joining
    seen_sources = set()
    unique_sources = []

    for source in state.sources_gathered:
        # Split the source into lines and process each individually
        for line in source.split("\n"):
            # Only process non-empty lines
            if line.strip() and line not in seen_sources:
                seen_sources.add(line)
                unique_sources.append(line)

    # Join the deduplicated sources
    all_sources = "\n".join(unique_sources)
    state.question_answered_da = (
        f"## Answer\n{state.question_answered_da}\n\n ### Sources:\n{all_sources}"
    )
    return {"question_answered_da": state.question_answered_da}


async def route_research(
    state: SummaryState, config: RunnableConfig
) -> Literal["generate_final_answer", "translate_content_follow_up"]:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_web_research_loops setting

    Returns:
        String literal indicating the next node to visit ("translate_content_follow_up" or "generate_final_answer")
    """

    configurable = Configuration.from_runnable_config(config)
    if state.research_loop_count <= configurable.max_web_research_loops:
        return "translate_content_follow_up"
    else:
        return "generate_final_answer"


# Add nodes and edges
builder = StateGraph(
    SummaryState,
    input=SummaryStateInput,
    output=SummaryStateOutput,
    config_schema=Configuration,
)
builder.add_node("translate_question", translate_question)
builder.add_node("generate_research_topic", generate_research_topic)
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("translate_search_results", translate_search_results)
builder.add_node("summarize_sources", summarize_sources)
builder.add_node("reflect_on_summary", reflect_on_summary)
builder.add_node("translate_content_follow_up", translate_content_follow_up)
builder.add_node("translate_answer", translate_answer)
builder.add_node("generate_final_answer", generate_final_answer)
builder.add_node("finalize_answer", finalize_answer)

# Add edges
builder.add_edge(START, "translate_question")
builder.add_edge("translate_question", "generate_research_topic")
builder.add_edge("generate_research_topic", "generate_query")
builder.add_edge("generate_query", "web_research")
builder.add_edge("web_research", "translate_search_results")
builder.add_edge("translate_search_results", "summarize_sources")
builder.add_edge("summarize_sources", "reflect_on_summary")
builder.add_conditional_edges("reflect_on_summary", route_research)
builder.add_edge("translate_content_follow_up", "web_research")
builder.add_edge("generate_final_answer", "translate_answer")
builder.add_edge("translate_answer", "finalize_answer")
builder.add_edge("finalize_answer", END)

graph = builder.compile()
