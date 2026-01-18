"""
Configuration module for RLHF Data Agent.
Handles environment variables, API settings, and generation parameters.
"""
import os
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

load_dotenv()


class ModelProvider(Enum):
    """Supported model providers for response generation."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"  # Free local models


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    provider: ModelProvider
    model_name: str
    display_name: str
    requires_api_key: bool = True
    # Pricing per 1M tokens (USD) - updated Jan 2025
    input_price_per_million: float = 0.0
    output_price_per_million: float = 0.0


# Available models - add more as needed
# Pricing from official sources (Jan 2025):
# - Anthropic: https://www.anthropic.com/pricing
# - OpenAI: https://openai.com/pricing
AVAILABLE_MODELS = {
    # Anthropic models
    "claude-sonnet": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-sonnet-4-20250514",
        display_name="Claude Sonnet 4",
        input_price_per_million=3.0,
        output_price_per_million=15.0,
    ),
    "claude-haiku": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-haiku-4-20250514",
        display_name="Claude Haiku 4 (faster/cheaper)",
        input_price_per_million=0.80,
        output_price_per_million=4.0,
    ),
    # OpenAI models
    "gpt-4o": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4o",
        display_name="GPT-4o",
        input_price_per_million=2.50,
        output_price_per_million=10.0,
    ),
    "gpt-4o-mini": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4o-mini",
        display_name="GPT-4o Mini (faster/cheaper)",
        input_price_per_million=0.15,
        output_price_per_million=0.60,
    ),
    # Ollama local models (free, no API key)
    "ollama-llama3": ModelConfig(
        provider=ModelProvider.OLLAMA,
        model_name="llama3.2",
        display_name="Llama 3.2 (local/free)",
        requires_api_key=False,
        input_price_per_million=0.0,
        output_price_per_million=0.0,
    ),
    "ollama-codellama": ModelConfig(
        provider=ModelProvider.OLLAMA,
        model_name="codellama",
        display_name="CodeLlama (local/free)",
        requires_api_key=False,
        input_price_per_million=0.0,
        output_price_per_million=0.0,
    ),
    "ollama-deepseek": ModelConfig(
        provider=ModelProvider.OLLAMA,
        model_name="deepseek-coder-v2",
        display_name="DeepSeek Coder V2 (local/free)",
        requires_api_key=False,
        input_price_per_million=0.0,
        output_price_per_million=0.0,
    ),
}


class Config:
    """Central configuration for the RLHF data generation agent."""

    # API Configuration
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Default model settings
    DEFAULT_MODEL: str = "claude-sonnet"
    MAX_TOKENS: int = 1024

    # Generation parameters
    DEFAULT_BATCH_SIZE: int = 100
    MAX_BATCH_SIZE: int = 1000
    RATE_LIMIT_DELAY: float = 0.5  # seconds between API calls
    MAX_RETRIES: int = 3

    # Scoring weights for preference ranking
    # RLHF relevance: These weights determine how "chosen" vs "rejected" is decided
    EFFICIENCY_WEIGHT: float = 0.6
    CLARITY_WEIGHT: float = 0.4

    # Domains for ML/Data coding tasks
    DOMAINS: list = ["pandas", "numpy", "sklearn", "pytorch"]

    # Task types for diverse prompt generation
    TASK_TYPES: list = ["optimize", "debug", "explain", "generate", "refactor"]

    # Complexity levels
    COMPLEXITY_LEVELS: list = ["beginner", "intermediate", "advanced"]

    # Output settings
    OUTPUT_DIR: str = os.path.join(os.path.dirname(__file__), "output")

    @classmethod
    def get_model_config(cls, model_key: str) -> ModelConfig:
        """Get configuration for a specific model."""
        if model_key not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(AVAILABLE_MODELS.keys())}")
        return AVAILABLE_MODELS[model_key]

    @classmethod
    def get_api_key(cls, provider: ModelProvider) -> str:
        """Get API key for a provider."""
        if provider == ModelProvider.ANTHROPIC:
            return cls.ANTHROPIC_API_KEY
        elif provider == ModelProvider.OPENAI:
            return cls.OPENAI_API_KEY
        return ""  # Ollama doesn't need API key

    @classmethod
    def validate_for_model(cls, model_key: str) -> "tuple[bool, str]":
        """Validate configuration for a specific model. Returns (is_valid, error_message)."""
        model_config = cls.get_model_config(model_key)

        if not model_config.requires_api_key:
            return True, ""

        api_key = cls.get_api_key(model_config.provider)
        if not api_key:
            provider_name = model_config.provider.value.upper()
            return False, f"{provider_name}_API_KEY not set. Set it in .env or environment."

        return True, ""

    @classmethod
    def validate(cls) -> "tuple[bool, str]":
        """Validate configuration for default model. Returns (is_valid, error_message)."""
        return cls.validate_for_model(cls.DEFAULT_MODEL)

    @classmethod
    def get_output_path(cls, filename: str) -> str:
        """Get full path for output file."""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        return os.path.join(cls.OUTPUT_DIR, filename)
