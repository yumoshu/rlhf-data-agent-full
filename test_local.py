"""
Local test script for RLHF Data Agent.
Tests prompt generation and heuristic scoring without API calls.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from generator.prompts import generate_prompts, ALL_TEMPLATES
from generator.ranker import score_response, determine_preference
from data.schema import RLHFDataPoint, ScoreBreakdown
from data.exporter import get_dataset_stats


def test_prompt_generation():
    """Test prompt template expansion."""
    print("=" * 60)
    print("Testing Prompt Generation")
    print("=" * 60)

    # Generate 10 prompts
    prompts = list(generate_prompts(count=10))
    print(f"\nGenerated {len(prompts)} prompts from {len(ALL_TEMPLATES)} templates")

    for i, (prompt, domain, task_type, complexity) in enumerate(prompts[:3]):
        print(f"\n--- Prompt {i+1} [{domain}/{task_type}/{complexity}] ---")
        print(prompt[:200] + "..." if len(prompt) > 200 else prompt)

    return len(prompts) == 10


def test_heuristic_scoring():
    """Test code scoring heuristics."""
    print("\n" + "=" * 60)
    print("Testing Heuristic Scoring")
    print("=" * 60)

    # Efficient code example
    efficient_code = '''```python
import pandas as pd

df_filtered = df[df["age"] > 30]
result = df_filtered.groupby("category")["value"].sum()
```'''

    # Verbose/inefficient code example
    verbose_code = '''```python
import pandas as pd

# This function filters the dataframe by age
# and then groups by category to sum values
def process_dataframe(input_df):
    """
    Process the input dataframe by filtering and aggregating.

    Args:
        input_df: Input pandas DataFrame

    Returns:
        Aggregated result
    """
    filtered_rows = []
    # Iterate through each row to check age
    for index, row in input_df.iterrows():
        if row["age"] > 30:
            filtered_rows.append(row)

    filtered_df = pd.DataFrame(filtered_rows)
    result = filtered_df.groupby("category")["value"].sum()
    return result

result = process_dataframe(df)
```'''

    score_efficient = score_response(efficient_code)
    score_verbose = score_response(verbose_code)

    print(f"\nEfficient code score: {score_efficient}")
    print(f"Verbose code score: {score_verbose}")

    # Test preference determination
    chosen, rejected, chosen_score, rejected_score = determine_preference(
        efficient_code, verbose_code
    )

    print(f"\nChosen score: {chosen_score.total:.3f}")
    print(f"Rejected score: {rejected_score.total:.3f}")
    print(f"Preference gap: {chosen_score.total - rejected_score.total:.3f}")

    # Efficient code should generally score higher on efficiency
    return score_efficient.efficiency > score_verbose.efficiency


def test_data_schema():
    """Test data schema creation."""
    print("\n" + "=" * 60)
    print("Testing Data Schema")
    print("=" * 60)

    # Create a mock data point
    dp = RLHFDataPoint.create_with_hash(
        prompt="How do I filter a pandas DataFrame?",
        chosen="Use df[df['col'] > value]",
        rejected="Use a for loop with iterrows",
        task_type="generate",
        domain="pandas",
        complexity="beginner",
        chosen_score=ScoreBreakdown(efficiency=0.8, clarity=0.7, total=0.76),
        rejected_score=ScoreBreakdown(efficiency=0.4, clarity=0.6, total=0.48),
    )

    print(f"\nData point ID: {dp.id}")
    print(f"SHA-256 hash: {dp.metadata.sha256[:32]}...")
    print(f"Generated at: {dp.metadata.generated_at}")

    # Test serialization
    json_data = dp.model_dump()
    print(f"\nSerialized to JSON with {len(json_data)} keys")

    return dp.metadata.sha256 is not None


def test_stats_calculation():
    """Test dataset statistics."""
    print("\n" + "=" * 60)
    print("Testing Statistics Calculation")
    print("=" * 60)

    # Create mock data points
    data_points = []
    for i in range(5):
        dp = RLHFDataPoint.create_with_hash(
            prompt=f"Test prompt {i}",
            chosen=f"Chosen response {i}",
            rejected=f"Rejected response {i}",
            task_type=["optimize", "debug", "generate"][i % 3],
            domain=["pandas", "numpy"][i % 2],
            complexity=["beginner", "intermediate", "advanced"][i % 3],
            chosen_score=ScoreBreakdown(efficiency=0.7 + i * 0.05, clarity=0.6 + i * 0.05, total=0.66 + i * 0.05),
            rejected_score=ScoreBreakdown(efficiency=0.4 + i * 0.02, clarity=0.3 + i * 0.02, total=0.36 + i * 0.02),
        )
        data_points.append(dp)

    stats = get_dataset_stats(data_points)
    print(f"\nDataset statistics:")
    print(f"  Total records: {stats['total_records']}")
    print(f"  Domains: {stats['domain_distribution']}")
    print(f"  Tasks: {stats['task_type_distribution']}")
    print(f"  Avg preference gap: {stats['preference_gap']['mean']:.3f}")

    return stats["total_records"] == 5


def main():
    """Run all tests."""
    print("\n🧪 RLHF Data Agent - Local Tests\n")

    results = []

    results.append(("Prompt Generation", test_prompt_generation()))
    results.append(("Heuristic Scoring", test_heuristic_scoring()))
    results.append(("Data Schema", test_data_schema()))
    results.append(("Statistics", test_stats_calculation()))

    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print("\n" + ("✅ All tests passed!" if all_passed else "❌ Some tests failed"))
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
