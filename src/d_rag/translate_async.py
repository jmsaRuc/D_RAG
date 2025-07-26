import asyncio
import logging
from typing import Any, Dict, List, Self, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
from torch import Tensor

from d_rag.prompts import translate_texts_instructions
from d_rag.state import SummaryState
from d_rag.utils import strip_thinking_tokens

# Configure logging for the module
log = logging.getLogger("d_rag.graph")

# Load the SentenceTransformer model for encoding text
model = SentenceTransformer("dilovancelik/all-distilroberta-v1_danish_law_fine_tune")


class SlidingWindowChunking:
    """
    Asynchronously creates a SlidingWindowChunking instance.

    Parameters:
        window_size (int | None):
            The number of words to include in each chunk. Defaults to 100 if None.
        step (int | None): The step size to move the window. Defaults to 50 if None.
    Returns:
        SlidingWindowChunking: A new instance with the specified window_size and step.
    """

    window_size: int
    step: int

    @classmethod
    async def create(
        cls,
        window_size: int | None = 100,
        step: int | None = 50,
    ) -> Self:
        """
        Asynchronously creates an instance of the class.

        with the specified window size and step.
        Args:
            window_size (int | None):
            The window size to set; defaults to 100 if not provided.
            step (int | None): The step value to set; defaults to 50 if not provided.
        Returns:
            Self: A new instance of the class with configured window_size and step.
        """
        # Create a new instance of the class
        self = cls()

        # Set default values if None
        if window_size is None or step is None:
            window_size = 100
            step = 50
        self.window_size = window_size
        self.step = step
        return self

    async def chunk(self: Self, text: str) -> list[str]:
        """
        Asynchronously splits the input text into overlapping chunks of words.

        Args:
            text (str): The text to be split into chunks.
        Returns:
            list[str]: A list of text chunks formed by a sliding window of words.
        """
        words = text.split()
        chunks = []

        # Iterate over the words with a sliding window approach
        for i in range(0, len(words) - self.window_size + 1, self.step):
            chunks.append(" ".join(words[i : i + self.window_size]))
        return chunks


class CosineSimilarityExtractor:
    """
    Asynchronously creates an instance of CosineSimilarityExtractor.

    Args:
        query (str): The search query to use for similarity extraction.
    Returns:
        CosineSimilarityExtractor: An initialized instance with the provided query.
    """

    query: str
    model: SentenceTransformer

    @classmethod
    async def create(cls, query: str) -> Self:
        """
        Asynchronously creates an instance of the class with the given query.

        Parameters:
            query (str): The query string to initialize the instance.
        Returns:
            Self: An instance of the class with initialized query and model attributes.
        """
        # Create a new instance of the class
        self = cls()
        self.query = query
        self.model = model
        return self

    async def find_relevant_chunks(
        self: Self,
        chunks: list[str],
    ) -> List[tuple[str, Tensor]]:
        """
        Find chunks that are most relevant to the query.

        Parameters:
            chunks (list[str]): A list of text chunks to be compared with the query.
        Returns:
            List[Tuple[str, Tensor]]: A list of tuples, each containing a text chunk and
            its corresponding similarity score.
        """
        # Encode the query and chunks
        vectors = self.model.encode([self.query, *chunks], batch_size=32)

        # Calculate cosine similarities between the query and chunks
        similarities = self.model.similarity(vectors[0:1], vectors[1:]).flatten()

        # return the chunks with their corresponding similarity scores
        return [(chunks[i], similarities[i]) for i in range(len(chunks))]


async def chunk_and_give_relevent(
    content: str,
    state: SummaryState,
    cutoff: float = 0.1,
) -> str:
    """
    Process the input content by chunking it.

    then filtering and concatenating chunks that are relevant to the search query
    contained in the state, based on a cutoff score.
    Parameters:
        content (str): The text content to be processed.
        state (SummaryState): An object containing the search query in its attributes.
        cutoff (float, optional): The similarity cutoff threshold for filtering chunks.
        Defaults to 0.1.
    Returns:
        str: A concatenated string of relevant text chunks.
    """
    # Create a sliding window chunker and chunk the content
    chunker = await SlidingWindowChunking.create(window_size=100, step=50)
    chunks = await chunker.chunk(content)

    # Check if query is empty
    if not state.search_query_da:
        raise ValueError(
            "Chunk ERROR: State should not be empty.",
        )

    # If no chunks are created, return an empty string
    query = state.search_query_da
    extractor = await CosineSimilarityExtractor.create(query)
    relevant_chunks = await extractor.find_relevant_chunks(chunks)

    # Filter the relevant chunks based on the cutoff score
    final = "Relevant excerpts from source:\n\n"
    for r, s in relevant_chunks:
        if s > cutoff:
            final += f"{r}\n---\n"
    return final


async def translate_content_async(
    content: str,
    llm_translate_s: ChatGroq | ChatOpenAI | ChatOllama,
) -> str:
    """
    Translate content from Danish to English asynchronously.

    Parameters:
        content (str): The text content to translate.
        llm_translate_s (ChatGroq | ChatOllama):
            The language model instance used for translation.
    Returns:
        str: The translated text with any '<think>' tags removed.
    """

    # Create a new instance of the ChatGroq model for translation
    human_message_content = (
        f"Translate the following texts from Danish to English."
        f"\n <User Input> \n {content} \n <User Input>\n\n"
    )

    # Invoke the translation model with the system and human messages
    result = await llm_translate_s.ainvoke(
        [
            SystemMessage(content=translate_texts_instructions),
            HumanMessage(content=human_message_content),
        ],
    )

    # Extract the translated text from the respons
    content = str(result.content)

    # strip any thinking tokens from the content
    # and return the cleaned content
    return await strip_thinking_tokens(content)


async def handel_suplementary_conten(
    suplementary_content: Dict[str, Any],
    source: Dict[str, Any],
    char_limit: int,
    llm_translate_s: ChatGroq | ChatOpenAI | ChatOllama,
    state: SummaryState,
) -> str:
    """
    Process and translate supplementary content asynchronously.

    Parameters:
        suplementary_content (Dict[str, Any]):
        A dictionary containing the supplementary content details.
        source (Dict[str, Any]): A dictionary containing main source details.
        char_limit (int): Maximum allowed character length for the content.
        llm_translate_s (ChatGroq | ChatOllama):
        The language model service used to translate the content.
        state (SummaryState): State information passed to the chunking function.
    Returns:
        str: The formatted and translated text of the supplementary content.
    """

    # get the content length of the supplementary content
    test_content_len = suplementary_content.get("content", "")

    # check if the content length is greater than the character limit
    if len(test_content_len) > char_limit:
        # Log the information about the content length before chunking
        log.info("INFO- To long suplementary")
        log.info(f"Len befor chunk: {len(test_content_len)}")

        # Chunk the content and filter it for relevance
        suplementary_content["content"] = await chunk_and_give_relevent(
            suplementary_content["content"],
            state,
        )

        # If the content is still too long, chunk it again with a lower cutoff
        if len(suplementary_content["content"]) > char_limit:
            suplementary_content["content"] = await chunk_and_give_relevent(
                suplementary_content["content"],
                state,
                cutoff=0.2,
            )

        # If the content is still too long, chunk it again with an even lower cutoff
        if len(suplementary_content["content"]) <= 0:
            suplementary_content["content"] = "...[NO_CONTENT_AVAILABLE]"

        log.info(f"Len after chunk: {len(suplementary_content['content'])}")

    # Translate the supplementary content asynchronously
    translated_suplementary_content = await translate_content_async(
        suplementary_content["content"],
        llm_translate_s,
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
        # Format the supplementary content for these specific document types
        formatted_text += f"Suplementary source: {suplementary_content['title']}\n===\n"
        formatted_text += (
            f"Suplementary to main source \"{source['title']}\"\n===\n"  # noqa: Q003
        )
        formatted_text += (
            f"Popular title: {suplementary_content.get('popularTitle')}\n===\n"
        )
        formatted_text += f"Short name: {suplementary_content.get('shortName')}\n===\n"
        formatted_text += (
            f"Suplementary source type: "
            f"{suplementary_content.get('documentType')}\n===\n"
        )
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
    llm_translate_s: ChatGroq | ChatOpenAI | ChatOllama,
    state: SummaryState,
) -> str:
    """
    Asynchronously handles the translation and formatting of the main source content.

    Parameters:
        source (Dict[str, Any]):
            Dictionary containing source attributes like title, content, URL, etc.
        char_limit (int):
            Maximum allowed character length for content; triggers chunking if exceeded.
        llm_translate_s (ChatGroq | ChatOllama):
            Translation service used to translate the content.
            state (SummaryState):
                State object used during chunking to manage summary details.
    Returns:
        str: A formatted string that includes metadata and translated content,
        including any supplementary sources.
    """

    # Get the content length of the main source
    test_content_len = source.get("content", "")

    # Check if the content length is greater than the character limit
    if source["content"] and len(test_content_len) > char_limit:
        # Log the information about the content length before chunking
        log.info("INFO- To long main source")
        log.info(f"Source: {source['title']}")
        log.info(f"Len befor chunk: {len(test_content_len)}")

        # Chunk the content and filter it for relevance
        source["content"] = await chunk_and_give_relevent(source["content"], state)

        # If the content is still too long, chunk it again with a lower cutoff
        if len(source["content"]) > char_limit:
            source["content"] = await chunk_and_give_relevent(
                source["content"],
                state,
                cutoff=0.2,
            )

            # If the content is still too long, chunk it again with an even lower cutoff
            if len(source["content"]) > char_limit:
                source["content"] = await chunk_and_give_relevent(
                    source["content"],
                    state,
                    cutoff=0.4,
                )

        # if the content got smaller than 0, set it to a placeholder
        if len(source["content"]) <= 0:
            source["content"] = "...[NO_CONTENT_AVAILABLE]"

        log.info(f"Len after chunk: {len(source['content'])}")

    # Translate the main content asynchronously
    translated_main_content = await translate_content_async(
        source["content"],
        llm_translate_s,
    )

    # Initialize a variable to hold the formatted text
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
        formatted_text += f"\nSuplementary sources to main source \"{source['title']}\":\n\n\n"  # noqa: E501, Q003

        # Create and run tasks for each supplementary content
        tasks = [
            asyncio.create_task(
                handel_suplementary_conten(
                    suplementary_content,
                    source,
                    char_limit,
                    llm_translate_s,
                    state,
                ),
            )
            for suplementary_content in source["suplementary_content"]
        ]

        # enitialize the tasks and concatenate the results
        for task in tasks:
            formatted_text_out = await task
            formatted_text += formatted_text_out

    return formatted_text


async def deduplicate_translate_and_format_sources(
    search_response: Dict[str, Any],
    llm_translate_s: ChatGroq | ChatOpenAI | ChatOllama,
    max_tokens_per_source: int,
    state: SummaryState,
) -> Tuple[str, str]:
    """
    Deduplicate, translate, and format search result sources.

    This asynchronous function accepts a search response containing multiple sources,
    deduplicates them based on URL, translates the main content
    (and supplementary contentwhen applicable) using the provided translation service,
    and builds a formatted string
    with retrieved metadata and translated texts.
    Parameters:
        search_response (Dict[str, Any]):
            A dictionary with a "results" key containing source entries.
        llm_translate_s (ChatGroq | ChatOllama):
            The translation service to use for asynchronously translating content.
    Returns:
        Tuple[str, Any]:
            A tuple where the first element is a string with formatted source data
            and the second element is the translated main content of the first source.
    """
    # Convert input to list of results
    if isinstance(search_response, dict):
        sources_list = search_response["results"]
    else:
        raise ValueError(
            "Input must be either a dict with 'results' or a list of search results",
        )

    # only keep unique sources based on URL
    unique_sources = {}
    for source in sources_list:
        if source.get("title") and source.get("url"):  # noqa: SIM102
            if source.get("url") not in unique_sources:
                unique_sources[source["url"]] = source

    # Using rough estimate of 4 characters per token
    char_limit = max_tokens_per_source * 3
    char_limit = round(char_limit)

    # Varibel for saving first
    save_translate_ex: str = ""

    # Format output
    formatted_text = "Sources:\n\n"

    # Create and run tasks for each unique source
    tasks = [
        asyncio.create_task(
            handel_main_source(
                source,
                char_limit,
                llm_translate_s,
                state,
            ),
        )
        for source in unique_sources.values()
    ]

    # Initialize the tasks and concatenate the results
    for i, task in enumerate(tasks, 1):
        formatted_text_out = await task
        formatted_text += formatted_text_out

        # save the translated main content for the first source
        # to be used translate it back to Danish
        if i == 1:
            save_translate_ex = formatted_text_out

    return formatted_text.strip(), save_translate_ex.strip()
