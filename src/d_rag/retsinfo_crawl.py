import asyncio
import collections.abc
import logging
import time
from typing import Any, Dict, List, Mapping

import httpx
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    RateLimiter,
)
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.models import CrawlResultContainer

from d_rag.state import (
    SummaryState,
)

# Set up logging for the module
log = logging.getLogger("d_rag.graph")

##_________________________search and link/metada grapper_______________________________


async def retsinfo_query_search(
    query: str,
    max_results: int,
    state: SummaryState,
) -> List[Dict[str, Any]] | List[None]:
    """
    Perform a query search on the Retsinformation API.

    Args:
        query (str): The search query string.
        max_results (int): The maximum number of results to return.
        Defaults to 3.
        state (SummaryState): The current state of the summary.

    Returns:
        List[Dict[str, Any]] | List[None]: A list of document dictionaries
            returned by the API.
            Returns an empty list if an HTTP error occurs.
    """
    # Check if max_results is greater than 10
    # If so, set results_per_page to 10, otherwise set it to max_results
    # Parameters used for the API request
    # dt: category of the document
    # h: False whether to include historical documents
    # ps: max_results how many results to return
    # t: query the search term
    params: Mapping[str, Any]
    params = {"dt": 10, "h": False, "ps": max_results, "t": query}
    if state.research_loop_count > 0:
        params = {
            "dt": [
                10,
                1480,
                20,
                30,
                40,
                50,
                90,
                120,
                270,
                60,
                100,
                80,
                110,
                130,
                140,
                150,
                160,
                170,
                180,
                200,
                210,
                220,
                1510,
                1490,
                -10,
                230,
                240,
                250,
                260,
                980,
            ],
            "h": False,
            "ps": max_results + 2,
            "t": query,
        }
    # Log the parameters used for the search
    log.info(f"Performing search with params: {params}")
    try:
        # Create an asynchronous HTTP client
        async with httpx.AsyncClient() as client:
            # Send a GET request to the API with the specified parameters
            response = await client.get(
                "https://www.retsinformation.dk/api/documentsearch",
                params=params,
            )
            # Log the response status code
            log.info(f"Search response status code: {response.status_code}")
            # Raise an exception if the request was unsuccessful
            response.raise_for_status()
            # Return the list of documents from the JSON response
            return response.json()["documents"]
    except httpx.HTTPStatusError as e:
        # Handle HTTP errors and log the error details
        log.error(f"HTTP error: {e.response.status_code!s} - {e.response.text!s}")
        log.error(f"Full error details: {e!s}")
        # Return an empty list if an error occurs
        return []


##_________________________________________extracting, child urls_______________________


async def retsinfo_get_related_documents(
    document_id: str,
) -> Dict[str, Any] | Dict[str, None]:
    """
    Fetch related documents for a given document ID from the Retsinformation API.

    Args:
        document_id (str): The unique identifier of the document.

    Returns:
         Dict[str, Any] | Dict[str, None]:
            A dictionary containing related document references.
    """
    try:
        # Create an asynchronous HTTP client
        async with httpx.AsyncClient() as client:
            # Send a GET request to the API to fetch related documents
            response = await client.get(
                f"https://www.retsinformation.dk/api/document/{document_id}/references/1",
            )
            # Raise an exception if the request was unsuccessful
            response.raise_for_status()
            # Return the list of related documents from the JSON response
            return response.json()
    except httpx.HTTPStatusError as e:
        log.error(f"HTTP error: {e.response.status_code!s} - {e.response.text!s}")
        log.error(f"Full error details: {e!s}")
        # Return an empty dictionary if an error occurs
        return {}


async def get_related(
    search_result: Dict[str, Any] | None,
) -> Dict[str, Any] | Dict[str, None]:
    """
    Retrieve document details and related documents.

    Args:
        search_result (Dict[str, Any] | None):
        A dictionary containing document information.

    Returns:
        Dict[str, Any] | Dict[str, None]: A dictionary that includes the document's id,
            title, popularTitle,shortName, documentType, retsinfoLink,
            and its related documents.
    """
    # Check if search_result is None or does not contain the required keys
    if search_result is None:
        # If not, return an empty dictionary
        return {}
    # fetch related documents using the retsinfo_get_related_documents function
    reletad_documents = await retsinfo_get_related_documents(search_result["id"])
    # create a dictionary to store the results
    return {
        "id": search_result["id"],
        "title": search_result["title"],
        "popularTitle": search_result["popularTitle"],
        "shortName": search_result["shortName"],
        "documentType": search_result["documentType"],
        # construct the retsinfoLink URL
        "retsinfoLink": f"https://www.retsinformation.dk{search_result['retsinfoLink']}",
        "referenceGroups": reletad_documents["referenceGroups"],
        "euReferenceGroups": reletad_documents["euReferenceGroups"],
        "documentLinkGroups": reletad_documents["documentLinkGroups"],
    }


async def retsinfo_search(
    qury: str,
    max_results: int,
    state: SummaryState,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Perform a search query on the Retsinformation API and retrieve related documents.

    Args:
        qury (str): The search query string.
        max_results (int): The maximum number of results to return.
        state (SummaryState): The current state of the summary.

    Returns:
        Dict[str, List[Dict[str, Any]]]:
            A dictionary containing a list of main documents
            and their related documents. Returns an empty
            list if an error occurs.
    """
    try:
        results = []
        ## get the search main law results
        search_results = await retsinfo_query_search(qury, max_results, state)

        ## create a list of tasks to fetch related documents for each main document
        tasks = [asyncio.create_task(get_related(result)) for result in search_results]
        # Use asyncio.gather to run the tasks concurrently
        # and wait for all of them to complete
        for task in tasks:
            result = await task
            results.append(result)
        return {"results": results}
    except Exception as e:
        # Handle any exceptions that occur during the API request
        log.error(f"Error in retsinfo_search: {e!s}")
        log.error(f"Full error details: {type(e).__name__}")
        # Return an empty list if an error occurs
        return {"results": []}


##____________________________final crawl_______________________________________________


async def clean_content_crawlr(
    id_url_pair: Dict[int, str],
) -> List[CrawlResultContainer]:
    """
    Crawl the site from url.

    clean the content of the provided URL using an asynchronous web crawler
    and convert it to markdown.

    Args:
        id_url_pair (Dict[int, str]):
            A dictionary mapping unique IDs to their corresponding URLs.

    Returns:
        CrawlResultContaine: The cleaned markdown content
        extracted from the webpage.
    """
    urls = list(id_url_pair.values())

    # create the browser configuration
    browser_config = BrowserConfig(browser_type="chromium", headless=True)

    # Create crawler run configuration
    crawler_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        verbose=True,
        excluded_tags=["nav", "footer", "header"],
        remove_overlay_elements=True,
        wait_for="css:.document-content",
        css_selector=".document-content",
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(
                threshold=0.43,
                threshold_type="dynamic",
                min_word_threshold=0,
            ),
            options={"ignore_links": True, "citations": True},
        ),
        magic=True,
    )

    # Create a memory adaptive dispatcher with rate limiting
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=90.0,
        check_interval=1.0,
        max_session_permit=40,
        rate_limiter=RateLimiter(
            base_delay=(1, 3),
            max_delay=10.0,
            max_retries=3,
            rate_limit_codes=[429, 503],
        ),
    )

    # enitialize the crawler
    crawler = AsyncWebCrawler(config=browser_config)

    # Start the crawler

    return await crawler.arun_many(
        urls=urls,
        config=crawler_config,
        dispatcher=dispatcher,
    )


async def main_crawl(id_url_pair_v: Dict[int, str]) -> Dict[int, CrawlResultContainer]:
    """
    This function orchestrates the crawling process.

    For a set of URLs associated with unique IDs.
    It utilizes the `clean_content_crawlr` function to fetch
    and clean the content of the URLs,
    then maps the results back to their corresponding IDs.
    The function also handles errors and logs the success
    or failure of each crawl operation.

    Args:
        id_url_pair_v (Dict[int, str]):
            A dictionary mapping unique IDs to their corresponding URLs.

    Returns:
        Dict[int, CrawlResultContainer]:
            A dictionary mapping unique IDs to their crawl results,
            where each result contains the cleaned content or error details.
    """

    end_results = {}

    # Call the clean_content_crawlr function to crawl the URLs and get the results
    crawl_results = await clean_content_crawlr(id_url_pair_v)

    # Iterate through the crawl results and map them back to their corresponding IDs
    for id, url in id_url_pair_v.items():
        for result in crawl_results:
            if result.url == url:
                if result.success:
                    end_results[id] = result
                    break
                # If the result is not successful, log the error message
                log.warning(
                    f"Warning crawling {url}: {result.error_message!s}",
                )
                log.warning(
                    f"Id: {id}, URL: {url}, Error: {result.error_message!s}",
                )
                log.warning(
                    f"URL Macthing Id: {id_url_pair_v[id]}",  # noqa: PLR1733
                )
                end_results[id] = result
                break

    # Check if the ID-URL pairs match the crawl results
    try:
        for id, result in end_results.items():
            if id_url_pair_v[id] == result.url:
                if result.success:
                    log.info(f"Successfully crawled {result.url} for ID {id}")
            elif id_url_pair_v[id] != result.url:
                log.error(
                    f"ID-URL pair mismatch: {id} - {id_url_pair_v[id]} != {result.url}",
                )
                raise ValueError(
                    f"ID-URL pair mismatch: {id} - {id_url_pair_v[id]} != {result.url}",
                    f"Id: {id}",
                    f"URL: {id_url_pair_v[id]}",
                    f"URL Macthing Id: {id_url_pair_v[id]}",
                )
            else:
                raise KeyError
    except ValueError as e:
        log.error(f"ValueError: {e!s}")
        log.error(f"Id: {id}")
        log.error(f"URL: {id_url_pair_v[id]}")
        log.error(f"URL Macthing Id: {id_url_pair_v[id]}")

    return end_results


##_____________formatting the data _______________________________________________


async def extract_ids(
    data: Dict[str, Any],
    state_header: str | None = None,
) -> Dict[int, str]:
    """
    Recursively extracts IDs and their associated URLs from a nested dictionary or list.

    Args:
        data (Dict[str, Any]): The input data, which can be a dictionary
        or list containing nested structures.
        state_header (str | None = None): The header state to check against
    Returns:
        Dict[str, str]: A dictionary mapping IDs to their corresponding URLs.
    """

    results = {}
    # Check if the data is a dictionary or a list
    if isinstance(data, dict):

        if "id" in data and "retsinfoLink" in data:
            # Construct the URL using the retsinfoLink
            results[data["id"]] = data["retsinfoLink"]

        elif "header" in data:
            state_header = data["header"]

        elif "id" in data and "eliPath" in data:  # noqa: SIM102

            # Check if the document is not historical
            if data["isHistorical"] is False:  # noqa: SIM102

                # check if header is the corect type
                if state_header in (
                    "Senere ændringer til forskriften",
                    "Alle bekendtgørelser m.v. og cirkulærer m.v. til denne lov",
                ):

                    # Construct the URL using the eliPath
                    results[data["id"]] = (
                        f"https://www.retsinformation.dk{data["eliPath"]}"
                    )

        for v in data.values():
            # Recursively extract IDs from nested dictionaries
            results.update(await extract_ids(v, state_header))

    # Check if the data is a list
    elif isinstance(data, list):

        for item in data:
            # Recursively extract IDs from nested lists
            results.update(await extract_ids(item, state_header))

    return results


async def better_update(
    d: Dict[str, Any],
    u: Dict[str, Any] | Mapping[Any, Any],
) -> Dict[str, Any]:
    """
    Asynchronously updates a dictionary with values from another dictionary.

    Args:
        d (Dict[str, Any]): The original dictionary to be updated.
        u (Dict[str, Any] | Mapping[Any, Any]): The dictionary containing updates.
    Returns:
        (Dict[str, Any]): The updated dictionary.
    """
    for k, v in u.items():
        # If the value is a dictionary, recursively update it
        if isinstance(v, collections.abc.Mapping):
            # If the key exists in the original dictionary, update it
            d[k] = await better_update(d.get(k, {}), v)
        else:
            # Otherwise, create a new dictionary for the key
            d[k] = v
    return d


async def sort_after_crawl(  # noqa: C901, PLR0912
    search_result: List[Dict[str, Any]] | Dict[str, Any],
    crawl_results: Dict[int, CrawlResultContainer],
    state_id: str | None = None,
    state_header: str | None = None,
) -> Dict[str, Any]:
    """
    Asynchronously processes and organizes crawl results based on a search result.

    Args:
        search_result (List[Dict[str, Any]] | Dict[str, Any]):
            The search result data to process.
        crawl_results (Dict[int, CrawlResultContainer]):
            The crawl results containing content data.
        state_id (str | None = None):
            The current state ID being processed. Defaults to None.
        state_header (str | None = None):
            The current state header being processed. Defaults to None.
    Returns:
        (Dict[str, Any]): A dictionary containing the organized and processed results.
    """
    results: Dict[str, Any] = {}
    # Check if the search result is a dictionary or a list
    if isinstance(search_result, dict):
        # Check if the search result contains an ID and a retsinfoLink

        if "id" in search_result and "retsinfoLink" in search_result:
            # declare the state_id variable
            state_id = search_result["id"]

            # extract relevant information and store it in the results dictionary
            results[search_result["id"]] = {}
            results[search_result["id"]]["title"] = search_result["title"]
            results[search_result["id"]]["popularTitle"] = search_result.get(
                "popularTitle",
            )
            results[search_result["id"]]["shortName"] = search_result["shortName"]
            results[search_result["id"]]["documentType"] = search_result["documentType"]
            results[search_result["id"]]["url"] = search_result["retsinfoLink"]

            # Check if the crawl result for the ID exists
            if crawl_results.get(int(search_result["id"])):

                # check if the crawl result is successful
                if crawl_results[int(search_result["id"])].success:
                    results[search_result["id"]]["content"] = crawl_results[
                        int(search_result["id"])
                    ].markdown.fit_markdown

                # If the crawl result is not successful, set content to a placeholder
                else:
                    results[search_result["id"]][
                        "content"
                    ] = "...[NO_CONTENT_AVALIABLE]"
                    log.warning(
                        f"Warning no content from crawl of {search_result["title"]}",
                    )
        # Check if the search result contains a header and a showExtendedReferenceLinks
        # If so, set the state_header variable
        elif "header" in search_result:
            state_header = search_result["header"]

        # Check if the search result contains an ID and an eliPath
        # If so, extract relevant information and store it in the results dictionary
        elif (
            "id" in search_result
            and "eliPath" in search_result
            and state_id is not None
        ):

            # Check if the document is not historical
            if search_result["isHistorical"] is False:  # noqa: SIM102

                # Check if the state_header is one of the specified types
                if state_header in (
                    "Senere ændringer til forskriften",
                    "Alle bekendtgørelser m.v. og cirkulærer m.v. til denne lov",
                ):
                    # extract relevant information
                    # and store it in the results dictionary
                    results[state_id] = {}
                    results[state_id]["suplementary_content"] = {}
                    results[state_id]["suplementary_content"][search_result["id"]] = {}
                    results[state_id]["suplementary_content"][search_result["id"]][
                        "title"
                    ] = search_result["title"]
                    results[state_id]["suplementary_content"][search_result["id"]][
                        "popularTitle"
                    ] = search_result.get("popularTitle")
                    results[state_id]["suplementary_content"][search_result["id"]][
                        "shortName"
                    ] = search_result.get("shortName")
                    results[state_id]["suplementary_content"][search_result["id"]][
                        "documentType"
                    ] = state_header
                    results[state_id]["suplementary_content"][search_result["id"]][
                        "url"
                    ] = f"https://www.retsinformation.dk{search_result["eliPath"]}"
                    results[state_id]["suplementary_content"][search_result["id"]][
                        "releaseDate"
                    ] = search_result["offentliggoerelsesDato"]

                    # Check if the crawl result for the ID exists
                    if crawl_results.get(int(search_result["id"])):

                        # check if the crawl result is successful
                        if crawl_results[int(search_result["id"])].success:
                            results[state_id]["suplementary_content"][
                                search_result["id"]
                            ]["content"] = crawl_results[
                                int(search_result["id"])
                            ].markdown.fit_markdown

                        # If the crawl result is not successful,
                        # set content to a placeholder
                        else:
                            results[state_id]["suplementary_content"][
                                search_result["id"]
                            ]["content"] = "...[NO_CONTENT_AVALIABLE]"
                            results[state_id]["suplementary_content"][
                                search_result["id"]
                            ]["ex_content"] = "...[NO_CONTENT_AVALIABLE]"

        # Recursively process nested dictionaries or lists
        for v in search_result.values():

            # Update the results dictionary with the processed data
            results = await better_update(
                results,
                await sort_after_crawl(v, crawl_results, state_id, state_header),
            )

    # Check if the search result is a list
    # If so, recursively process each item in the list
    elif isinstance(search_result, list):
        for item in search_result:
            results = await better_update(
                results,
                await sort_after_crawl(item, crawl_results, state_id, state_header),
            )

    return results


async def format_sorted_results(
    sorted_results_v: Dict[str, Any],
) -> tuple[list[Any], int]:
    """
    Format the sorted results into a more readable structure.

    Args:
        sorted_results (Dict[str, Any]): The sorted results from the crawl.

    Returns:
        list[Dict[str, Any]]: A dictionary containing the formatted results.
    """
    formated_results_v = []
    count: int = 0

    # format the results the sorted_results_v into a list of dictionaries
    formated_results_v = list(sorted_results_v.values())
    count = len(formated_results_v)

    # remove the "suplementary_content" key from each dictionary
    # and convert it to a list of dictionaries
    for value in formated_results_v:
        if value.get("suplementary_content") is not None:
            value["suplementary_content"] = list(value["suplementary_content"].values())
            count += len(value["suplementary_content"])
        else:
            value["suplementary_content"] = []

    return formated_results_v, count


##________________________________________________main function_________________________


async def retsinfo_search_and_crawl(
    query: str,
    max_results: int,
    state: SummaryState,
) -> Dict[str, Any]:
    """
    Asynchronously performs a search and crawl operation on Retsinfo data.

    This function executes a series of asynchronous tasks to search for data
    based on a query, extract relevant IDs and URLs, crawl the associated data,
    sort the results, and format them for output.

    Args:
        query (str): The search query string to be used for retrieving data.
        max_results (int): The maximum number of results to retrieve from the search.
        state (SummaryState): The current state of the summary,
        used to track research depth.
    Returns:
        Dict[str, Any]: A dictionary containing the following keys:
            - "query" (str): The original search query.
            - "results" (Any): The formatted results after crawling and sorting.
            - "response_time" (float):
                The total time taken to perform the operation, in seconds.
    Raises:
        Any exceptions raised by the underlying asynchronous functions.
    """

    # Start the timer
    t0 = time.time()
    # Perform the search operation
    search_result = await retsinfo_search(query, max_results, state)
    # Extract IDs and URLs from the search result
    id_url_pair = await extract_ids(search_result)
    # Perform the crawl operation
    crawl_results = await main_crawl(id_url_pair)
    # Sort the crawl results based on the search result
    sorted_results = await sort_after_crawl(search_result, crawl_results)
    # Format the sorted results for output
    formatted_results, count = await format_sorted_results(sorted_results)
    # stop the timer
    t1 = time.time()
    return {
        "query": query,
        "results": formatted_results,
        "response_time": t1 - t0,
        "amount": count,
    }
