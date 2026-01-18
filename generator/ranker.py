"""
Heuristic-based code ranker for RLHF preference scoring.
Evaluates code on efficiency and clarity metrics to determine chosen/rejected pairs.

RLHF relevance: The scoring system creates the preference signal that teaches
models to prefer efficient, readable code. The metrics are designed to be:
- Deterministic: Same code always gets same score
- Explainable: Each score component has clear meaning
- Balanced: Efficiency and clarity are both valued
"""
import ast
import re
from dataclasses import dataclass

from data.schema import ScoreBreakdown
from config import Config


@dataclass
class CodeMetrics:
    """Raw metrics extracted from code analysis."""

    line_count: int
    has_docstring: bool
    comment_ratio: float
    avg_name_length: float
    has_type_hints: bool
    cyclomatic_complexity: int
    has_anti_patterns: bool
    uses_vectorization: bool
    function_count: int


# Anti-patterns that indicate inefficient code (domain-specific)
ANTI_PATTERNS = [
    r"\.iterrows\(\)",  # pandas: use vectorized operations instead
    r"\.apply\(lambda",  # pandas: often slower than vectorized
    r"for\s+\w+\s+in\s+range\(len\(",  # numpy: use broadcasting
    r"\.append\([^)]+\)\s*$",  # building lists in loops
    r"pd\.concat\(\[.*\]\)",  # repeated concat is slow
]

# Patterns that indicate efficient, vectorized code
EFFICIENT_PATTERNS = [
    r"\.vectorize\(",
    r"np\.\w+\(",  # numpy operations are typically vectorized
    r"\.apply\(np\.",  # applying numpy functions
    r"@torch\.no_grad",  # proper gradient management
    r"\.to\(device\)",  # proper device handling
    r"torch\.cuda\.amp",  # mixed precision training
]


def extract_code_from_response(response: str) -> str:
    """Extract code blocks from a response that may contain markdown."""
    # Try to find code in markdown blocks
    code_blocks = re.findall(r"```(?:python)?\n?(.*?)```", response, re.DOTALL)
    if code_blocks:
        return "\n\n".join(code_blocks)

    # If no code blocks, try to identify code-like content
    lines = response.split("\n")
    code_lines = []
    for line in lines:
        # Heuristic: lines that look like code
        if re.match(r"^[\s]*(import |from |def |class |if |for |while |return |#|\w+\s*=)", line):
            code_lines.append(line)
        elif code_lines and (line.startswith("    ") or line.startswith("\t") or line.strip() == ""):
            code_lines.append(line)

    return "\n".join(code_lines) if code_lines else response


def calculate_cyclomatic_complexity(code: str) -> int:
    """
    Calculate cyclomatic complexity of code.
    CC = E - N + 2P where E=edges, N=nodes, P=connected components

    For simplicity, we count decision points:
    - if/elif/else: +1 each
    - for/while loops: +1 each
    - try/except: +1 each
    - and/or in conditions: +1 each

    RLHF relevance: Lower complexity = more maintainable code = higher efficiency score
    """
    complexity = 1  # Base complexity

    # Count decision points
    decision_patterns = [
        (r"\bif\b", 1),
        (r"\belif\b", 1),
        (r"\bfor\b", 1),
        (r"\bwhile\b", 1),
        (r"\bexcept\b", 1),
        (r"\band\b", 1),
        (r"\bor\b", 1),
        (r"\btry\b", 1),
    ]

    for pattern, weight in decision_patterns:
        complexity += len(re.findall(pattern, code)) * weight

    return complexity


def extract_variable_names(code: str) -> list[str]:
    """Extract variable names from code for naming analysis."""
    # Simple regex-based extraction (not using AST for robustness)
    # Match assignment targets and function parameters
    patterns = [
        r"(\w+)\s*=",  # assignments
        r"def\s+\w+\(([^)]*)\)",  # function params
        r"for\s+(\w+)\s+in",  # loop variables
    ]

    names = []
    for pattern in patterns:
        matches = re.findall(pattern, code)
        for match in matches:
            if isinstance(match, str):
                # Split function params
                for name in match.split(","):
                    name = name.strip().split(":")[0].split("=")[0].strip()
                    if name and name not in ["self", "cls", "_"]:
                        names.append(name)

    return names


def analyze_code(code: str) -> CodeMetrics:
    """
    Analyze code and extract metrics for scoring.

    RLHF relevance: These metrics form the basis of preference ranking.
    """
    lines = code.strip().split("\n")
    non_empty_lines = [l for l in lines if l.strip()]

    # Count comments
    comment_lines = sum(1 for l in lines if l.strip().startswith("#"))
    comment_ratio = comment_lines / max(len(non_empty_lines), 1)

    # Check for docstring
    has_docstring = '"""' in code or "'''" in code

    # Check for type hints
    has_type_hints = bool(re.search(r":\s*(int|str|float|bool|list|dict|List|Dict|Optional|Any)", code))

    # Analyze variable names
    var_names = extract_variable_names(code)
    avg_name_length = sum(len(n) for n in var_names) / max(len(var_names), 1)

    # Check for anti-patterns
    has_anti_patterns = any(re.search(p, code) for p in ANTI_PATTERNS)

    # Check for efficient patterns
    uses_vectorization = any(re.search(p, code) for p in EFFICIENT_PATTERNS)

    # Count functions (indicates modular design)
    function_count = len(re.findall(r"\bdef\s+\w+", code))

    # Calculate complexity
    complexity = calculate_cyclomatic_complexity(code)

    return CodeMetrics(
        line_count=len(non_empty_lines),
        has_docstring=has_docstring,
        comment_ratio=comment_ratio,
        avg_name_length=avg_name_length,
        has_type_hints=has_type_hints,
        cyclomatic_complexity=complexity,
        has_anti_patterns=has_anti_patterns,
        uses_vectorization=uses_vectorization,
        function_count=function_count,
    )


def calculate_efficiency_score(metrics: CodeMetrics, baseline_lines: int = 20) -> float:
    """
    Calculate efficiency score (0-1) based on code metrics.

    Components:
    - Line count: Fewer lines (relative to baseline) = better
    - Cyclomatic complexity: Lower = better
    - Anti-patterns: Presence reduces score
    - Vectorization: Presence increases score

    RLHF relevance: Efficiency teaches models to write concise, performant code.
    """
    score = 0.5  # Start at neutral

    # Line count factor (0.0 to 0.25)
    # Fewer lines than baseline is good, more is bad
    line_ratio = metrics.line_count / baseline_lines
    if line_ratio <= 1.0:
        score += 0.15 * (1 - line_ratio)  # Bonus for concise code
    else:
        score -= min(0.15, 0.05 * (line_ratio - 1))  # Penalty for verbose code

    # Complexity factor (0.0 to 0.25)
    # Target complexity: 1-5 is good, >10 is bad
    if metrics.cyclomatic_complexity <= 5:
        score += 0.2
    elif metrics.cyclomatic_complexity <= 10:
        score += 0.1
    else:
        score -= min(0.1, 0.02 * (metrics.cyclomatic_complexity - 10))

    # Anti-patterns penalty
    if metrics.has_anti_patterns:
        score -= 0.15

    # Vectorization bonus
    if metrics.uses_vectorization:
        score += 0.15

    return max(0.0, min(1.0, score))


def calculate_clarity_score(metrics: CodeMetrics) -> float:
    """
    Calculate clarity score (0-1) based on code metrics.

    Components:
    - Docstrings: Presence is good
    - Comment ratio: 5-20% is optimal
    - Variable naming: Longer, descriptive names are better
    - Type hints: Presence is good
    - Modular design: Having functions is good

    RLHF relevance: Clarity teaches models to write readable, maintainable code.
    """
    score = 0.3  # Start below neutral (clarity needs to be earned)

    # Docstring bonus
    if metrics.has_docstring:
        score += 0.15

    # Comment ratio (optimal: 5-20%)
    if 0.05 <= metrics.comment_ratio <= 0.20:
        score += 0.15
    elif metrics.comment_ratio > 0:
        score += 0.08  # Some comments are better than none

    # Variable naming (avg length 4-15 is good)
    if 4 <= metrics.avg_name_length <= 15:
        score += 0.15
    elif metrics.avg_name_length >= 2:
        score += 0.08

    # Type hints bonus
    if metrics.has_type_hints:
        score += 0.1

    # Modular design (having functions)
    if metrics.function_count >= 1:
        score += 0.1
    if metrics.function_count >= 2:
        score += 0.05

    return max(0.0, min(1.0, score))


def score_response(response: str) -> ScoreBreakdown:
    """
    Score a code response on efficiency and clarity.

    Args:
        response: The full response (may include markdown, explanations)

    Returns:
        ScoreBreakdown with efficiency, clarity, and total scores

    RLHF relevance: This is the core scoring function that determines preference.
    Higher total scores indicate better code that should be "chosen" over lower-scored
    alternatives ("rejected").
    """
    code = extract_code_from_response(response)
    metrics = analyze_code(code)

    efficiency = calculate_efficiency_score(metrics)
    clarity = calculate_clarity_score(metrics)

    # Weighted total (configured in Config)
    total = (Config.EFFICIENCY_WEIGHT * efficiency + Config.CLARITY_WEIGHT * clarity)

    return ScoreBreakdown(
        efficiency=round(efficiency, 4),
        clarity=round(clarity, 4),
        total=round(total, 4),
    )


def determine_preference(response_a: str, response_b: str) -> tuple[str, str, ScoreBreakdown, ScoreBreakdown]:
    """
    Compare two responses and determine which is chosen/rejected.

    Args:
        response_a: First response
        response_b: Second response

    Returns:
        Tuple of (chosen, rejected, chosen_score, rejected_score)

    RLHF relevance: This creates the preference pairs that train the reward model.
    The chosen response should consistently be better across our metrics.
    """
    score_a = score_response(response_a)
    score_b = score_response(response_b)

    if score_a.total >= score_b.total:
        return response_a, response_b, score_a, score_b
    else:
        return response_b, response_a, score_b, score_a
