"""
Response generator with multi-model support.
Supports Anthropic Claude, OpenAI GPT, and Ollama local models.

RLHF relevance: By generating responses with different optimization targets
(efficiency vs verbosity), we create natural variation that enables meaningful
preference comparisons. This is more authentic than artificially degrading responses.
"""
import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator

from config import Config, ModelProvider, AVAILABLE_MODELS
from data.schema import RLHFDataPoint, ScoreBreakdown
from generator.ranker import determine_preference


@dataclass
class TokenUsage:
    """Tracks token usage for cost estimation."""
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: "TokenUsage") -> None:
        """Add another TokenUsage to this one."""
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens

    def calculate_cost(self, input_price_per_million: float, output_price_per_million: float) -> float:
        """Calculate cost in USD based on pricing per million tokens."""
        input_cost = (self.input_tokens / 1_000_000) * input_price_per_million
        output_cost = (self.output_tokens / 1_000_000) * output_price_per_million
        return input_cost + output_cost


@dataclass
class GenerationResult:
    """Result of a generation call including text and token usage."""
    text: str
    usage: TokenUsage = field(default_factory=TokenUsage)


# System prompts that encourage different code styles
SYSTEM_PROMPT_EFFICIENT = """You are an expert programmer focused on writing efficient, concise code.
When answering coding questions:
- Prioritize performance and brevity
- Use vectorized operations and built-in functions
- Minimize lines of code while maintaining correctness
- Include minimal but essential comments
- Use descriptive variable names
Respond with code in markdown code blocks."""

SYSTEM_PROMPT_VERBOSE = """You are a programmer who writes detailed, explanatory code.
When answering coding questions:
- Include extensive comments explaining each step
- Write verbose variable names that fully describe their purpose
- Add docstrings with examples
- Break operations into multiple explicit steps
- Include error handling even if not strictly required
Respond with code in markdown code blocks."""


class BaseModelClient(ABC):
    """Abstract base class for model clients."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str) -> GenerationResult:
        """Generate a response synchronously. Returns text and token usage."""
        pass

    @abstractmethod
    async def generate_async(self, prompt: str, system_prompt: str) -> GenerationResult:
        """Generate a response asynchronously. Returns text and token usage."""
        pass


class AnthropicClient(BaseModelClient):
    """Client for Anthropic Claude models."""

    def __init__(self, api_key: str, model_name: str):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.async_client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str, system_prompt: str) -> GenerationResult:
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=Config.MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        usage = TokenUsage(
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        )
        return GenerationResult(text=message.content[0].text, usage=usage)

    async def generate_async(self, prompt: str, system_prompt: str) -> GenerationResult:
        message = await self.async_client.messages.create(
            model=self.model_name,
            max_tokens=Config.MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        usage = TokenUsage(
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        )
        return GenerationResult(text=message.content[0].text, usage=usage)


class OpenAIClient(BaseModelClient):
    """Client for OpenAI GPT models."""

    def __init__(self, api_key: str, model_name: str):
        from openai import OpenAI, AsyncOpenAI
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str, system_prompt: str) -> GenerationResult:
        response = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=Config.MAX_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        usage = TokenUsage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )
        return GenerationResult(text=response.choices[0].message.content, usage=usage)

    async def generate_async(self, prompt: str, system_prompt: str) -> GenerationResult:
        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            max_tokens=Config.MAX_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        usage = TokenUsage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )
        return GenerationResult(text=response.choices[0].message.content, usage=usage)


class OllamaClient(BaseModelClient):
    """
    Client for Ollama local models.
    Free to use - requires Ollama running locally (ollama serve).
    Install: https://ollama.ai
    """

    def __init__(self, model_name: str, base_url: str = None):
        import httpx
        self.model_name = model_name
        self.base_url = base_url or Config.OLLAMA_BASE_URL
        self.client = httpx.Client(timeout=120.0)
        self.async_client = httpx.AsyncClient(timeout=120.0)

    def generate(self, prompt: str, system_prompt: str) -> GenerationResult:
        response = self.client.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()
        # Ollama returns token counts in the response
        usage = TokenUsage(
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
        )
        return GenerationResult(text=data["response"], usage=usage)

    async def generate_async(self, prompt: str, system_prompt: str) -> GenerationResult:
        response = await self.async_client.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()
        usage = TokenUsage(
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
        )
        return GenerationResult(text=data["response"], usage=usage)


def create_client(model_key: str, api_key: "str | None" = None) -> BaseModelClient:
    """
    Factory function to create the appropriate model client.

    Args:
        model_key: Key from AVAILABLE_MODELS (e.g., "claude-sonnet", "gpt-4o", "ollama-llama3")
        api_key: Optional API key override

    Returns:
        Configured model client
    """
    model_config = Config.get_model_config(model_key)

    if model_config.provider == ModelProvider.ANTHROPIC:
        key = api_key or Config.ANTHROPIC_API_KEY
        if not key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY or pass api_key.")
        return AnthropicClient(api_key=key, model_name=model_config.model_name)

    elif model_config.provider == ModelProvider.OPENAI:
        key = api_key or Config.OPENAI_API_KEY
        if not key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key.")
        return OpenAIClient(api_key=key, model_name=model_config.model_name)

    elif model_config.provider == ModelProvider.OLLAMA:
        return OllamaClient(model_name=model_config.model_name)

    else:
        raise ValueError(f"Unsupported provider: {model_config.provider}")


class ResponseGenerator:
    """
    Generates paired responses using configurable model backends.

    For each prompt, generates two responses:
    1. Efficient: Optimized for performance and brevity
    2. Verbose: Optimized for clarity and documentation

    The ranker then determines which becomes chosen/rejected based on scores.
    Tracks token usage for cost estimation.
    """

    def __init__(
        self,
        model_key: str = None,
        api_key: "str | None" = None,
    ):
        """
        Initialize the response generator.

        Args:
            model_key: Key from AVAILABLE_MODELS (default: Config.DEFAULT_MODEL)
            api_key: Optional API key override
        """
        self.model_key = model_key or Config.DEFAULT_MODEL
        self.client = create_client(self.model_key, api_key)
        self.total_usage = TokenUsage()  # Track cumulative token usage

    def generate_pair(self, prompt: str) -> "tuple[str, str, TokenUsage]":
        """
        Generate a pair of responses for preference comparison.

        Returns:
            Tuple of (efficient_response, verbose_response, combined_usage)
        """
        result_efficient = self.client.generate(prompt, SYSTEM_PROMPT_EFFICIENT)
        time.sleep(Config.RATE_LIMIT_DELAY)  # Rate limiting
        result_verbose = self.client.generate(prompt, SYSTEM_PROMPT_VERBOSE)

        # Combine usage from both calls
        pair_usage = TokenUsage()
        pair_usage.add(result_efficient.usage)
        pair_usage.add(result_verbose.usage)

        return result_efficient.text, result_verbose.text, pair_usage

    async def generate_pair_async(self, prompt: str) -> "tuple[str, str, TokenUsage]":
        """Generate a pair of responses asynchronously."""
        results = await asyncio.gather(
            self.client.generate_async(prompt, SYSTEM_PROMPT_EFFICIENT),
            self.client.generate_async(prompt, SYSTEM_PROMPT_VERBOSE),
        )
        pair_usage = TokenUsage()
        pair_usage.add(results[0].usage)
        pair_usage.add(results[1].usage)
        return results[0].text, results[1].text, pair_usage

    def generate_data_point(
        self,
        prompt: str,
        domain: str,
        task_type: str,
        complexity: str,
    ) -> "tuple[RLHFDataPoint, TokenUsage]":
        """
        Generate a complete RLHF data point with ranked preferences.

        This is the main entry point for data generation:
        1. Generate two responses with different styles
        2. Score both responses
        3. Determine chosen/rejected based on scores
        4. Package into RLHFDataPoint with hash

        Returns:
            Tuple of (data_point, token_usage_for_this_call)
        """
        response_efficient, response_verbose, usage = self.generate_pair(prompt)
        self.total_usage.add(usage)

        chosen, rejected, chosen_score, rejected_score = determine_preference(
            response_efficient, response_verbose
        )

        data_point = RLHFDataPoint.create_with_hash(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            task_type=task_type,
            domain=domain,
            complexity=complexity,
            chosen_score=chosen_score,
            rejected_score=rejected_score,
        )
        return data_point, usage

    async def generate_data_point_async(
        self,
        prompt: str,
        domain: str,
        task_type: str,
        complexity: str,
    ) -> "tuple[RLHFDataPoint, TokenUsage]":
        """Generate a data point asynchronously."""
        response_efficient, response_verbose, usage = await self.generate_pair_async(prompt)
        self.total_usage.add(usage)

        chosen, rejected, chosen_score, rejected_score = determine_preference(
            response_efficient, response_verbose
        )

        data_point = RLHFDataPoint.create_with_hash(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            task_type=task_type,
            domain=domain,
            complexity=complexity,
            chosen_score=chosen_score,
            rejected_score=rejected_score,
        )
        return data_point, usage


async def generate_batch_async(
    prompts: "list[tuple[str, str, str, str]]",
    model_key: str = None,
    api_key: "str | None" = None,
    concurrency: int = 5,
    on_progress: "callable" = None,
) -> "AsyncIterator[RLHFDataPoint]":
    """
    Generate multiple data points with controlled concurrency.

    Args:
        prompts: List of (prompt, domain, task_type, complexity) tuples
        model_key: Model to use (default: Config.DEFAULT_MODEL)
        api_key: Optional API key override
        concurrency: Max concurrent API calls
        on_progress: Optional callback(completed, total) for progress updates

    Yields:
        RLHFDataPoint for each prompt
    """
    generator = ResponseGenerator(model_key=model_key, api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)
    completed = 0
    total = len(prompts)

    async def generate_with_semaphore(prompt_data):
        nonlocal completed
        async with semaphore:
            prompt, domain, task_type, complexity = prompt_data
            try:
                result = await generator.generate_data_point_async(
                    prompt, domain, task_type, complexity
                )
                completed += 1
                if on_progress:
                    on_progress(completed, total)
                return result
            except Exception as e:
                print(f"Error generating data point: {e}")
                completed += 1
                if on_progress:
                    on_progress(completed, total)
                return None

    tasks = [generate_with_semaphore(p) for p in prompts]
    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result is not None:
            yield result


class GenerationStoppedError(Exception):
    """Raised when generation is stopped by user."""
    def __init__(self, message: str, partial_results: "list" = None, total_usage: "TokenUsage" = None):
        super().__init__(message)
        self.partial_results = partial_results or []
        self.total_usage = total_usage or TokenUsage()


def generate_batch_sync(
    prompts: "list[tuple[str, str, str, str]]",
    model_key: str = None,
    api_key: "str | None" = None,
    on_progress: "callable" = None,
    should_stop: "callable" = None,
    on_usage_update: "callable" = None,
) -> "tuple[list[RLHFDataPoint], TokenUsage]":
    """
    Synchronous batch generation with progress tracking, cancellation, and cost tracking.

    Args:
        prompts: List of (prompt, domain, task_type, complexity) tuples
        model_key: Model to use (default: Config.DEFAULT_MODEL)
        api_key: Optional API key override
        on_progress: Optional callback(completed, total) for progress updates
        should_stop: Optional callback() -> bool to check if generation should stop
        on_usage_update: Optional callback(total_usage: TokenUsage) for cost tracking

    Returns:
        Tuple of (list of RLHFDataPoint objects, total TokenUsage)

    Raises:
        GenerationStoppedError: If stopped by user (includes partial results and usage)
    """
    generator = ResponseGenerator(model_key=model_key, api_key=api_key)
    results = []
    total = len(prompts)
    stopped = False
    total_usage = TokenUsage()

    for i, (prompt, domain, task_type, complexity) in enumerate(prompts):
        # Check for stop request before each API call
        if should_stop and should_stop():
            stopped = True
            break

        try:
            data_point, usage = generator.generate_data_point(
                prompt, domain, task_type, complexity
            )
            results.append(data_point)
            total_usage.add(usage)

            # Notify about usage update for real-time cost display
            if on_usage_update:
                on_usage_update(total_usage)

        except Exception as e:
            print(f"Error generating data point {i+1}/{total}: {e}")

        if on_progress:
            on_progress(i + 1, total)

        # Check for stop request after each API call
        if should_stop and should_stop():
            stopped = True
            break

        if i < total - 1:
            time.sleep(Config.RATE_LIMIT_DELAY)

    if stopped:
        raise GenerationStoppedError(
            f"Generation stopped after {len(results)} data points",
            partial_results=results,
            total_usage=total_usage,
        )

    return results, total_usage
