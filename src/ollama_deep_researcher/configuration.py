import os
from enum import Enum
from pydantic import BaseModel, Field
from typing import Any, Optional, Literal

from langchain_core.runnables import RunnableConfig

class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"
    SEARXNG = "searxng"

class Configuration(BaseModel):
    """The configurable fields for the research assistant."""

    max_web_research_loops: int = Field(
        default=2,
        title="Research Depth",
        description="Number of research iterations to perform"
    )
    local_llm: str = Field(
        default="llama3.2",
        title="LLM Model Name",
        description="Name of the LLM model to use"
    )
    groq_llm: str = Field(
        default="deepseek-r1-distill-llama-70b",
        title="GROQ LLM Model Name",
        description="Name of the GROQ LLM model to use"
    )
    openai_llm: str = Field(
        default="o4-mini",
        title="OpenAI LLM Model Name",
        description="Name of the OpenAI LLM model to use"
    )
    llm_provider: Literal["ollama","groq", "openai"] = Field(
        default="groq",
        title="LLM Provider",
        description="Provider for the LLM (Ollama or LMStudio)"
    )
    groq_api_base: str = Field(
        default="https://api.groq.com/",
        title="GROQ API Base URL",
        description="Base URL for the GROQ API"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434/",
        title="Ollama Base URL",
        description="Base URL for Ollama API"
    )
    openai_api_base: str = Field(
        default="https://api.openai.com/v1/",
        title="OpenAI Base URL",
        description="Base URL for OpenAI API"
    )
    strip_thinking_tokens: bool = Field(
        default=True,
        title="Strip Thinking Tokens",
        description="Whether to strip <think> tokens from model responses"
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        
        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }
        
        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}
        
        return cls(**values)