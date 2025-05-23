from typing import Any, Dict, Tuple
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from ollama_deep_researcher.prompts import translate_texts_instructions


async def get_config_value(value: Any) -> str:
    """
    Convert configuration values to string format, handling both string and enum types.

    Args:
        value (Any): The configuration value to process. Can be a string or an Enum.

    Returns:
        str: The string representation of the value.

    Examples:
        >>> get_config_value("tavily")
        'tavily'
        >>> get_config_value(SearchAPI.TAVILY)
        'tavily'
    """
    return value if isinstance(value, str) else value.value


async def strip_thinking_tokens(text: str) -> str:
    """
    Remove <think> and </think> tags and their content from the text.

    Iteratively removes all occurrences of content enclosed in thinking tokens.

    Args:
        text (str): The text to process

    Returns:
        str: The text with thinking tokens and their content removed
    """
    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>") + len("</think>")
        text = text[:start] + text[end:]
    return text


async def translate_content_async(
    content: str, llm_translate_s: ChatGroq | ChatOllama
) -> str:
    """
    Translate content from Danish to English asynchronously.
    Parameters:
        content (str): The text content to translate.
        llm_translate_s (ChatGroq | ChatOllama): The language model instance used for translation.
    Returns:
        str: The translated text with any '<think>' tags removed.
    """

    # Create a new instance of the ChatGroq model for translation
    human_message_content = f"Translate the following texts from Danish to English. \n <User Input> \n {content} \n <User Input>\n\n"
    result = await llm_translate_s.ainvoke(
        [
            SystemMessage(content=translate_texts_instructions),
            HumanMessage(content=human_message_content),
        ]
    )

    # Extract the translated text from the response

    # Remove <think> and </think> tags and their content
    translated_text = await strip_thinking_tokens(result.content)

    return translated_text


async def deduplicate_translate_and_format_sources(
    search_response: Dict[str, Any], llm_translate_s: ChatGroq | ChatOllama
) -> Tuple[str, Any]:
    """
    Deduplicate, translate, and format search result sources.
    This asynchronous function accepts a search response containing multiple sources,
    deduplicates them based on URL, translates the main content (and supplementary content
    when applicable) using the provided translation service, and builds a formatted string
    with retrieved metadata and translated texts.
    Parameters:
        search_response (Dict[str, Any]): A dictionary with a "results" key containing source entries.
        llm_translate_s (ChatGroq | ChatOllama): The translation service to use for asynchronously translating content.
    Returns:
        Tuple[str, Any]: A tuple where the first element is a string with formatted source data
                         and the second element is the translated main content of the first source.
    """

    # Convert input to list of results
    if isinstance(search_response, dict):
        sources_list = search_response["results"]
    else:
        raise ValueError(
            "Input must be either a dict with 'results' or a list of search results"
        )

    # only keep unique sources based on URL
    unique_sources = {}
    for source in sources_list:
        if source["url"] not in unique_sources:
            unique_sources[source["url"]] = source

    # Format output
    formatted_text = "Sources:\n\n"
    for i, source in enumerate(unique_sources.values(), 1):

        # translate the main content
        translated_main_content = await translate_content_async(
            source["content"], llm_translate_s
        )

        # Format metadata and content, and add to the single text
        formatted_text += f"Main source: {source['title']}\n===\n"
        formatted_text += f"Popular title: {source.get('popularTitle')}\n===\n"
        formatted_text += f"Short name: {source.get('shortName')}\n===\n"
        formatted_text += f"Document type: {source.get('documentType')}\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Main source content:\n{translated_main_content}\n"

        # save the translated main content for the first source
        # to be used translate it back to Danish
        if i == 1:
            save_translate_ex = translated_main_content

        # check if there are any supplementary sources for the source
        if source["suplementary_content"]:
            formatted_text += (
                f"Suplementary sources to main source \"{source['title']}\":\n\n"
            )

            # format each supplementary source the same way as main source
            for suplementary_content in source["suplementary_content"]:

                # check if the document type is one of the following:
                # 'Senere ændringer til forskriften'
                # 'Alle bekendtgørelser m.v. og cirkulærer m.v. til denne lov'
                if (
                    suplementary_content["documentType"]
                    == "Senere ændringer til forskriften"
                    or suplementary_content["documentType"]
                    == "Alle bekendtgørelser m.v. og cirkulærer m.v. til denne lov"
                ):
                    formatted_text += (
                        f"Suplementary source: {suplementary_content['title']}\n===\n"
                    )
                    formatted_text += (
                        f"Suplementary to main source \"{source['title']}\"\n===\n"
                    )
                    formatted_text += f"Popular title: {suplementary_content.get('popularTitle')}\n===\n"
                    formatted_text += (
                        f"Short name: {suplementary_content.get('shortName')}\n===\n"
                    )
                    formatted_text += f"Suplementary source type: {suplementary_content.get('documentType')}\n===\n"
                    formatted_text += f"URL: {suplementary_content['url']}\n===\n"
                    formatted_text += f"release date: {suplementary_content.get('releaseDate')}\n===\n"
                    formatted_text += f"Suplementary source content:\n{await translate_content_async(suplementary_content['content'], llm_translate_s)}\n\n"

    return formatted_text.strip(), save_translate_ex


async def format_sources(search_results: Dict[str, Any]) -> str:
    """
    Format search results into a bullet-point list of sources with URLs.

    Creates a bulleted list of search results with title and URL for each source.
    Additionally, if a source has supplementary content, it includes these entries indented
    beneath the main source.

    Args:
        search_results (Dict[str, Any]): Search response containing a 'results' key with
                                         a list of search result objects.

    Returns:
        str: Formatted string with sources as bullet points. Main sources are formatted as
             "* title : url" and supplementary sources as "  * title : url".
    """
    lines = []

    # Iterate through each source in the search results
    # and format them into a bullet-point list
    for source in search_results.get("results", []):
        lines.append(f"* {source['title']} : {source['url']}")
        if source.get("suplementary_content"):
            for supplementary in source["suplementary_content"]:

                # check if the document type is one of the following:
                # 'Senere ændringer til forskriften'
                # 'Alle bekendtgørelser m.v. og cirkulærer m.v. til denne lov'
                if (
                    supplementary["documentType"] == "Senere ændringer til forskriften"
                    or supplementary["documentType"]
                    == "Alle bekendtgørelser m.v. og cirkulærer m.v. til denne lov"
                ):
                    lines.append(
                        f"  * {supplementary['title']} : {supplementary['url']}"
                    )
    return "\n".join(lines)
