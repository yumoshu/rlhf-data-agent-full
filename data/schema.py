"""
Pydantic models for RLHF/DPO data schema.
Designed for compatibility with REPPO and standard preference learning formats.
"""
import hashlib
import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, computed_field


class ScoreBreakdown(BaseModel):
    """
    Heuristic scores for a code response.
    RLHF relevance: These scores determine preference ranking.
    """

    efficiency: float = Field(ge=0.0, le=1.0, description="Code efficiency score")
    clarity: float = Field(ge=0.0, le=1.0, description="Code clarity/readability score")
    total: float = Field(ge=0.0, le=1.0, description="Weighted total score")


class DataPointMetadata(BaseModel):
    """Metadata for each RLHF data point."""

    task_type: Literal["optimize", "debug", "explain", "generate", "refactor"]
    domain: Literal["pandas", "numpy", "sklearn", "pytorch"]
    complexity: Literal["beginner", "intermediate", "advanced"]
    chosen_score: ScoreBreakdown
    rejected_score: ScoreBreakdown
    sha256: str = Field(description="Hash for data integrity verification")
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class RLHFDataPoint(BaseModel):
    """
    Single RLHF/DPO data point for preference learning.

    Format follows standard DPO structure:
    - prompt: The user query/task
    - chosen: Preferred response (higher score)
    - rejected: Less preferred response (lower score)
    - metadata: Scoring details and verification hash

    RLHF relevance: This structure enables direct use in DPO/RLHF training pipelines.
    The chosen/rejected pair teaches the model to prefer efficient, clear code.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str = Field(description="The coding task or question")
    chosen: str = Field(description="Preferred response (better efficiency/clarity)")
    rejected: str = Field(description="Less preferred response")
    metadata: DataPointMetadata

    @classmethod
    def create_with_hash(
        cls,
        prompt: str,
        chosen: str,
        rejected: str,
        task_type: str,
        domain: str,
        complexity: str,
        chosen_score: ScoreBreakdown,
        rejected_score: ScoreBreakdown,
    ) -> "RLHFDataPoint":
        """
        Factory method that creates a data point with computed SHA-256 hash.
        Hash covers prompt + chosen + rejected for integrity verification.
        """
        # Compute hash for REPPO verification
        content = f"{prompt}|{chosen}|{rejected}"
        sha256_hash = hashlib.sha256(content.encode()).hexdigest()

        metadata = DataPointMetadata(
            task_type=task_type,
            domain=domain,
            complexity=complexity,
            chosen_score=chosen_score,
            rejected_score=rejected_score,
            sha256=sha256_hash,
        )

        return cls(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            metadata=metadata,
        )


class DatasetManifest(BaseModel):
    """
    Manifest file for the generated dataset.
    REPPO compatibility: Includes metadata for dataset verification and discovery.
    """

    name: str = "rlhf-coding-ml-data"
    version: str = "1.0.0"
    description: str = "RLHF/DPO dataset for ML/Data coding assistance"
    total_records: int
    domains: "list[str]"
    task_types: "list[str]"
    complexity_distribution: "dict[str, int]"
    avg_chosen_score: float
    avg_rejected_score: float
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    generator: str = "rlhf_data_agent"

    @computed_field
    @property
    def preference_gap(self) -> float:
        """Average score difference between chosen and rejected."""
        return round(self.avg_chosen_score - self.avg_rejected_score, 4)
