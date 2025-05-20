from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
    RateLimiter,
)
from crawl4ai.models import CrawlResultContainer
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher
import time

from typing import Dict, Any, List
import asyncio
import httpx
import gc
import collections.abc
from langsmith import traceable
##_________________________________________search and link/metada grapper_______________________________________________


async def retsinfo_query_search(query: str, max_results: int) -> List[Dict[str, Any]]:
    """
    Perform a query search on the Retsinformation API.

    Args:
        query (str): The search query string.
        max_results (int, optional): The maximum number of results to return. Defaults to 3.

    Returns:
        List[Dict[str, Any]]: A list of document dictionaries returned by the API.
                              Returns an empty list if an HTTP error occurs.
    """
    gc.collect()
    # Check if max_results is greater than 10
    # If so, set results_per_page to 10, otherwise set it to max_results
    # Parameters used for the API request
    # dt: category of the document
    # h: False whether to include historical documents
    # ps: max_results how many results to return
    # t: query the search term
    params = {"dt": 10, "h": False, "ps": max_results, "t": query}
    try:
        # Create an asynchronous HTTP client
        async with httpx.AsyncClient() as client:
            # Send a GET request to the API with the specified parameters
            response = await client.get(
                "https://www.retsinformation.dk/api/documentsearch", params=params
            )
            # Raise an exception if the request was unsuccessful
            response.raise_for_status()
            # Return the list of documents from the JSON response
            return response.json()["documents"]
    except httpx.HTTPStatusError as e:
        print(f"HTTP error: {str(e.response.status_code)} - {str(e.response.text)}")
        print(f"Full error details: {str(e)}")
        # Return an empty list if an error occurs
        return []


##_________________________________________extracting, child urls_______________________________________________


async def retsinfo_get_related_documents(document_id: str) -> Dict[str, Any]:
    """
    Fetch related documents for a given document ID from the Retsinformation API.

    Args:
        document_id (str): The unique identifier of the document.

    Returns:
        Dict[str, Any]: A dictionary containing related document references.
    """
    gc.collect()
    try:
        # Create an asynchronous HTTP client
        async with httpx.AsyncClient() as client:
            # Send a GET request to the API to fetch related documents
            response = await client.get(
                f"https://www.retsinformation.dk/api/document/{document_id}/references/1"
            )
            # Raise an exception if the request was unsuccessful
            response.raise_for_status()
            # Return the list of related documents from the JSON response
            return response.json()
    except httpx.HTTPStatusError as e:
        print(f"HTTP error: {str(e.response.status_code)} - {str(e.response.text)}")
        print(f"Full error details: {str(e)}")
        # Return an empty dictionary if an error occurs
        return {}


async def get_related(search_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve document details and related documents.

    Args:
        search_result (Dict[str, Any]): A dictionary containing document information.

    Returns:
        Dict[str, Any]: A dictionary that includes the document's id, title, popularTitle,
                        shortName, documentType, retsinfoLink, and its related documents.
    """
    gc.collect()

    # fetch related documents using the retsinfo_get_related_documents function
    reletad_documents = await retsinfo_get_related_documents(search_result["id"])
    # create a dictionary to store the results
    result = {
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
    return result


async def retsinfo_search(
    qury: str, max_results: int
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Perform a search query on the Retsinformation API and retrieve related documents.

    Args:
        qury (str): The search query string.
        max_results (int): The maximum number of results to return.

    Returns:
        Dict[str, List[Dict[str, Any]]]: A dictionary containing a list of main documents
                                          and their related documents. Returns an empty
                                          list if an error occurs.
    """
    gc.collect()
    try:
        results = []
        ## get the search main law results
        search_results = await retsinfo_query_search(qury, max_results)

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
        print(f"Error in retsinfo_search: {str(e)}")
        print(f"Full error details: {type(e).__name__}")
        # Return an empty list if an error occurs
        return {"results": []}


##________________________________________________final crawl_______________________________________________


async def clean_content_crawlr(
    id_url_pair: Dict[int, str],
) -> Dict[str, CrawlResultContainer]:
    gc.collect()
    """
    Crawl the site from url,
    then clean the content of the provided URL using an asynchronous web crawler and convert it to markdown.

    Parameters:2
       urls: Dict[int, str: The URL from which to retrieve and clean content.

    Returns:
        Dict[str, CrawlResultContainer]: The cleaned markdown content extracted from the webpage.
    """
    urls = list(id_url_pair.values())

    # Create browser configuration
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
                threshold=0.43, threshold_type="dynamic", min_word_threshold=0
            ),
            options={"ignore_links": True, "citations": True},
        ),
    )

    # Create a dispatcher
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=90.0,
        check_interval=1.0,
        max_session_permit=40,
        rate_limiter=RateLimiter(
            base_delay=(0.5, 2),
            max_delay=10.0,
            max_retries=3,
            rate_limit_codes=[429, 503],
        ),
    )

    # enitialize the crawler
    crawler = AsyncWebCrawler(config=browser_config)

    # Start the crawler
    results = await crawler.arun_many(
        urls=urls,
        config=crawler_config,
        dispatcher=dispatcher,
    )

    return results


async def main_crawl(id_url_pair_v: Dict[int, str]) -> Dict[int, CrawlResultContainer]:
    """
    This function orchestrates the crawling process for a set of URLs associated with unique IDs.
    It utilizes the `clean_content_crawlr` function to fetch and clean the content of the URLs,
    then maps the results back to their corresponding IDs. The function also handles errors
    and logs the success or failure of each crawl operation.

    Args:
        id_url_pair_v (Dict[int, str]): A dictionary mapping unique IDs to their corresponding URLs.

    Returns:
        Dict[int, CrawlResultContainer]: A dictionary mapping unique IDs to their crawl results,
        where each result contains the cleaned content or error details.
    """
    gc.collect()

    end_results = {}

    # Call the clean_content_crawlr function to crawl the URLs and get the results
    crawl_results = await clean_content_crawlr(id_url_pair_v)

    # Iterate through the crawl results and map them back to their corresponding IDs
    for result in crawl_results:
        if result.success:
            for id, url in id_url_pair_v.items():
                if result.url == url:
                    end_results[id] = result
                    break
        else:
            print("Failed:", result.url, "-")
            print(f"   Id: {id}")
            print(f"   URL: {result.url}")
            print(f"   URL Macthing Id: {id_url_pair_v[id]}")
            end_results[id] = result

    # Check if the ID-URL pairs match the crawl results
    for id, result in end_results.items():
        if id_url_pair_v[id] == result.url:
            print(result.url, "crawled OK!")
        elif id_url_pair_v[id] != result.url:
            print("Failed:", "-", "Wrong Id Url pair")
            print(f"   Id: {id}")
            print(f"   URL: {result.url}")
            print(f"   URL Macthing Id: {id_url_pair_v[id]}")
        else:
            print("Failed:", result.url, "-")

    return end_results


##________________________________________________formatting the data _______________________________________________


async def extract_ids(data: Dict[str, Any]) -> Dict[str, str]:
    """
    Recursively extracts IDs and their associated URLs from a nested dictionary or list.
    Args:
        data (Dict[str, Any]): The input data, which can be a dictionary or list containing nested structures.
    Returns:
        Dict[str, str]: A dictionary mapping IDs to their corresponding URLs.
    """

    results = {}
    # Check if the data is a dictionary or a list
    if isinstance(data, dict):
        if "id" in data and "retsinfoLink" in data:
            # Construct the URL using the retsinfoLink
            results[data["id"]] = data["retsinfoLink"]
        elif "id" in data and "eliPath" in data:
            # Construct the URL using the eliPath
            results[data["id"]] = f"https://www.retsinformation.dk{data["eliPath"]}"
        for v in data.values():
            # Recursively extract IDs from nested dictionaries
            results.update(await extract_ids(v))
    # Check if the data is a list
    elif isinstance(data, list):
        for item in data:
            # Recursively extract IDs from nested lists
            results.update(await extract_ids(item))
    return results


async def better_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    """
    Asynchronously updates a dictionary with values from another dictionary.
    Args:
        d (Dict[str, Any]): The original dictionary to be updated.
        u (Dict[str, Any]): The dictionary containing updates.
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


async def sort_after_crawl(
    search_result: Dict[str, List[Dict[str, Any]]],
    crawl_results: Dict[int, CrawlResultContainer],
    state_id: int = None,
    state_header: str = None,
) -> Dict[str, Any]:
    """
    Asynchronously processes and organizes crawl results based on a search result structure.
    Args:
        search_result (Dict[str, List[Dict[str, Any]]]): The search result data to process.
        crawl_results (Dict[int, CrawlResultContainer]): The crawl results containing content data.
        state_id (int, optional): The current state ID being processed. Defaults to None.
        state_header (str, optional): The current state header being processed. Defaults to None.
    Returns:
        (Dict[str, Any]): A dictionary containing the organized and processed results.
    """
    results = {}
    # Check if the search result is a dictionary or a list
    if isinstance(search_result, dict):
        # Check if the search result contains an ID and a retsinfoLink
        # If so, extract relevant information and store it in the results dictionary
        if "id" in search_result and "retsinfoLink" in search_result:
            results[search_result["id"]] = {}
            results[search_result["id"]]["title"] = search_result["title"]
            results[search_result["id"]]["popularTitle"] = search_result.get(
                "popularTitle"
            )
            results[search_result["id"]]["shortName"] = search_result["shortName"]
            results[search_result["id"]]["documentType"] = search_result["documentType"]
            results[search_result["id"]]["url"] = search_result["retsinfoLink"]
            results[search_result["id"]]["content"] = crawl_results[
                search_result["id"]
            ].markdown.fit_markdown
            state_id = search_result["id"]
        # Check if the search result contains a header and a showExtendedReferenceLinks
        # If so, set the state_header variable
        elif "header" in search_result:
            state_header = search_result["header"]
        # Check if the search result contains an ID and an eliPath
        # If so, extract relevant information and store it in the results dictionary
        elif "id" in search_result and "eliPath" in search_result:
            results[state_id] = {}
            results[state_id]["suplementary_content"] = {}
            results[state_id]["suplementary_content"][search_result["id"]] = {}
            results[state_id]["suplementary_content"][search_result["id"]]["title"] = (
                search_result["title"]
            )
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
            results[state_id]["suplementary_content"][search_result["id"]][
                "content"
            ] = crawl_results[search_result["id"]].markdown.fit_markdown
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
) -> list[Dict[str, Any]]:
    """
    Format the sorted results into a more readable structure.

    Args:
        sorted_results (Dict[str, Any]): The sorted results from the crawl.

    Returns:
        list[Dict[str, Any]]: A dictionary containing the formatted results.
    """
    gc.collect()
    formated_results_v = []
    # format the results the sorted_results_v into a list of dictionaries
    formated_results_v = list(sorted_results_v.values())
    sorted_results_v = None
    # remove the "suplementary_content" key from each dictionary
    # and convert it to a list of dictionaries
    for value in formated_results_v:
        if value.get("suplementary_content") is not None:
            value["suplementary_content"] = list(value["suplementary_content"].values())
        else:
            value["suplementary_content"] = []

    return formated_results_v


##________________________________________________main function_______________________________________________

@traceable
async def retsinfo_search_and_crawl(query: str, max_results: int) -> Dict[str, Any]:
    """
    Asynchronously performs a search and crawl operation on Retsinfo data.
    This function executes a series of asynchronous tasks to search for data
    based on a query, extract relevant IDs and URLs, crawl the associated data,
    sort the results, and format them for output.

    Args:
        query (str): The search query string to be used for retrieving data.
        max_results (int): The maximum number of results to retrieve from the search.
    Returns:
        Dict[str, Any]: A dictionary containing the following keys:
            - "query" (str): The original search query.
            - "results" (Any): The formatted results after crawling and sorting.
            - "response_time" (float): The total time taken to perform the operation, in seconds.
    Raises:
        Any exceptions raised by the underlying asynchronous functions.
    """
    gc.collect()
    # Start the timer
    t0 = time.time()
    # Perform the search operation
    search_result = await retsinfo_search(query, max_results)
    # Extract IDs and URLs from the search result
    id_url_pair = await extract_ids(search_result)
    # Perform the crawl operation
    crawl_results = await main_crawl(id_url_pair)
    # Sort the crawl results based on the search result
    sorted_results = await sort_after_crawl(search_result, crawl_results)
    # Format the sorted results for output
    formatted_results = await format_sorted_results(sorted_results)
    # stop the timer
    t1 = time.time()
    final_results = {
        "query": query,
        "results": formatted_results,
        "response_time": t1 - t0,
    }
    return final_results