from typing import Any, Dict, Tuple
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer
import asyncio
import gc

from ollama_deep_researcher.prompts import translate_texts_instructions
from ollama_deep_researcher.utils import strip_thinking_tokens
from ollama_deep_researcher.state import SummaryState

gc.enable()

model = SentenceTransformer("dilovancelik/all-distilroberta-v1_danish_law_fine_tune")


class SlidingWindowChunking:
    @classmethod
    async def create(cls, window_size=100, step=50):
        self = cls()
        self.window_size = window_size
        self.step = step
        return self

    async def chunk(self, text):
        words = text.split()
        chunks = []
        for i in range(0, len(words) - self.window_size + 1, self.step):
            chunks.append(" ".join(words[i : i + self.window_size]))
        return chunks


class CosineSimilarityExtractor:
    @classmethod
    async def create(cls, query):
        self = cls()
        self.query = query
        self.model = model
        return self

    async def find_relevant_chunks(self, chunks):
        vectors = self.model.encode([self.query] + chunks)
        similarities = self.model.similarity(vectors[0:1], vectors[1:]).flatten()
        return [(chunks[i], similarities[i]) for i in range(len(chunks))]


async def chunk_and_give_relevent(
    content: str, state: SummaryState, cutoff: float = 0.1
) -> str:

    chunker = await SlidingWindowChunking.create(window_size=100, step=50)
    chunks = await chunker.chunk(content)

    query = state.search_query_da
    extractor = await CosineSimilarityExtractor.create(query)
    relevant_chunks = await extractor.find_relevant_chunks(chunks)

    final = ""
    for r, s in relevant_chunks:
        if s > cutoff:
            final += f"{r}\n"
    return final


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


async def handel_suplementary_conten(
    suplementary_content: Dict[str, Any],
    source: Dict[str, Any],
    char_limit: int,
    llm_translate_s: ChatGroq | ChatOllama,
    state: SummaryState,
) -> str:
    # check if over char_limit
    test_content_len = suplementary_content.get("content", "")
    if len(test_content_len) > char_limit:

        print("INFO- To long suplementary")
        print(f"    Len befor chunk: {len(test_content_len)}")

        suplementary_content["content"] = await chunk_and_give_relevent(
            suplementary_content["content"], state
        )

        if suplementary_content["content"] > char_limit:
            suplementary_content["content"] = await chunk_and_give_relevent(
                suplementary_content["content"], state, cutoff=0.2
            )

        if len(suplementary_content["content"]) <= 0:
            suplementary_content["content"] = "...[NO_CONTENT_AVAILABLE]"

        print(f"    Len after chunk: {len(source["content"])}")

    translated_suplementary_content = await translate_content_async(
        suplementary_content["content"], llm_translate_s
    )

    formatted_text: str = ""

    # check if the document type is one of the following:
    # 'Senere ændringer til forskriften'
    # 'Alle bekendtgørelser m.v. og cirkulærer m.v. til denne lov'
    if (
        suplementary_content["documentType"] == "Senere ændringer til forskriften"
        or suplementary_content["documentType"]
        == "Alle bekendtgørelser m.v. og cirkulærer m.v. til denne lov"
    ):
        formatted_text += f"Suplementary source: {suplementary_content['title']}\n===\n"
        formatted_text += f"Suplementary to main source \"{source['title']}\"\n===\n"
        formatted_text += (
            f"Popular title: {suplementary_content.get('popularTitle')}\n===\n"
        )
        formatted_text += f"Short name: {suplementary_content.get('shortName')}\n===\n"
        formatted_text += f"Suplementary source type: {suplementary_content.get('documentType')}\n===\n"
        formatted_text += f"URL: {suplementary_content['url']}\n===\n"
        formatted_text += (
            f"release date: {suplementary_content.get('releaseDate')}\n===\n"
        )
        formatted_text += (
            f"Suplementary source content:\n{translated_suplementary_content}\n\n"
        )
    return formatted_text


async def handel_main_source(
    source: Dict[str, Any],
    char_limit: int,
    llm_translate_s: ChatGroq | ChatOllama,
    state: SummaryState,
) -> str:

    # check if over char_limit
    test_content_len = source.get("content", "")

    if source["content"]:
        if len(test_content_len) > char_limit:
            print(f"INFO- To long main")
            print(f"    Len befor chunk: {len(test_content_len)}")
            source["content"] = await chunk_and_give_relevent(source["content"], state)
            if len(source["content"]) > char_limit:
                source["content"] = await chunk_and_give_relevent(
                    source["content"], state, cutoff=0.2
                )
                if len(source["content"]) > char_limit:
                    source["content"] = await chunk_and_give_relevent(
                        source["content"], state, cutoff=0.4
                    )
            if len(source["content"]) <= 0:
                source["content"] = "...[NO_CONTENT_AVAILABLE]"

            print(f"    Len after chunk: {len(source["content"])}")

    translated_main_content = await translate_content_async(
        source["content"], llm_translate_s
    )

    formatted_text: str = ""

    # Format metadata and content, and add to the single text
    formatted_text += f"Main source: {source['title']}\n===\n"
    formatted_text += f"Popular title: {source.get('popularTitle')}\n===\n"
    formatted_text += f"Short name: {source.get('shortName')}\n===\n"
    formatted_text += f"Document type: {source.get('documentType')}\n===\n"
    formatted_text += f"URL: {source['url']}\n===\n"
    formatted_text += f"Main source content:\n{translated_main_content}\n\n"

    # check if there are any supplementary sources for the source
    if source["suplementary_content"]:
        formatted_text += (
            f"\nSuplementary sources to main source \"{source['title']}\":\n\n\n"
        )
        tasks = [
            asyncio.create_task(
                handel_suplementary_conten(
                    suplementary_content, source, char_limit, llm_translate_s, state
                )
            )
            for suplementary_content in source["suplementary_content"]
        ]
        for task in tasks:
            formatted_text_out = await task
            formatted_text += formatted_text_out
    return formatted_text


async def deduplicate_translate_and_format_sources(
    search_response: Dict[str, Any],
    llm_translate_s: ChatGroq | ChatOllama,
    max_tokens_per_source: int,
    state: SummaryState,
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
        if source.get("title") and source.get("url"):
            if source.get("url") not in unique_sources:
                unique_sources[source["url"]] = source

    # Using rough estimate of 4 characters per token
    char_limit = max_tokens_per_source * 3
    char_limit = round(char_limit)
    # Varibel for saving first
    save_translate_ex: str = ""

    # Format output
    formatted_text = "Sources:\n\n"

    # Varibel for saving first
    save_translate_ex: str = ""

    tasks = [
        asyncio.create_task(
            handel_main_source(
                source,
                char_limit,
                llm_translate_s,
                state,
            )
        )
        for source in unique_sources.values()
    ]

    for i, task in enumerate(tasks, 1):
        formatted_text_out = await task
        formatted_text += formatted_text_out

        # save the translated main content for the first source
        # to be used translate it back to Danish
        if i == 1:
            save_translate_ex = formatted_text_out

    return formatted_text.strip(), save_translate_ex.strip()
