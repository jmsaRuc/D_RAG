from typing import Any, Dict

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
        if source['title']:
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