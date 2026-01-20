"""
RLHF Data Agent - Streamlit Application

A web interface for generating RLHF/DPO training data for ML/Data coding assistance.
Designed for publication on reppo.ai.

RLHF relevance: This demo enables:
1. Interactive configuration of data generation
2. Real-time preview of generated preference pairs
3. Export in multiple formats for training pipelines
"""
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import json
import os
import random
import tempfile
import time
import streamlit as st

from config import Config, AVAILABLE_MODELS, ModelProvider

# File-based stop flag for interrupting generation
STOP_FLAG_FILE = os.path.join(tempfile.gettempdir(), "rlhf_agent_stop_flag")


def set_stop_flag():
    """Create stop flag file."""
    with open(STOP_FLAG_FILE, "w") as f:
        f.write("stop")


def clear_stop_flag():
    """Remove stop flag file."""
    if os.path.exists(STOP_FLAG_FILE):
        os.remove(STOP_FLAG_FILE)


def check_stop_flag():
    """Check if stop flag file exists."""
    return os.path.exists(STOP_FLAG_FILE)


from data.exporter import (
    generate_csv_content,
    generate_json_content,
    generate_jsonl_content,
    generate_manifest_content,
    get_dataset_stats,
)
from data.schema import RLHFDataPoint
from generator.prompts import generate_prompts
from generator.ranker import score_response
from generator.responses import generate_batch_sync, ResponseGenerator, GenerationStoppedError, TokenUsage


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="RLHF Data Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# SESSION STATE
# =============================================================================
if "generated_data" not in st.session_state:
    st.session_state.generated_data = []
if "generation_in_progress" not in st.session_state:
    st.session_state.generation_in_progress = False
if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False
if "session_usage" not in st.session_state:
    st.session_state.session_usage = TokenUsage()
if "session_cost" not in st.session_state:
    st.session_state.session_cost = 0.0

# Clean up any stale stop flag on fresh page load
# This prevents getting stuck if the app was closed mid-generation
if not st.session_state.generation_in_progress:
    clear_stop_flag()


# =============================================================================
# SIDEBAR - CONFIGURATION
# =============================================================================
st.sidebar.title("⚙️ Configuration")

# Model Selection
st.sidebar.subheader("Model Selection")

# Group models by provider for display
model_options = {}
for key, config in AVAILABLE_MODELS.items():
    model_options[key] = f"{config.display_name}"

selected_model = st.sidebar.selectbox(
    "Select Model",
    options=list(model_options.keys()),
    format_func=lambda x: model_options[x],
    index=list(model_options.keys()).index(Config.DEFAULT_MODEL),
    help="Choose which LLM to use for generating responses",
)

# Get the selected model's config
selected_model_config = AVAILABLE_MODELS[selected_model]

# Show provider-specific API key input
if selected_model_config.provider == ModelProvider.ANTHROPIC:
    api_key = st.sidebar.text_input(
        "Anthropic API Key",
        type="password",
        help="Required for Claude models. Set ANTHROPIC_API_KEY env var to pre-fill.",
        value=Config.ANTHROPIC_API_KEY,
    )
elif selected_model_config.provider == ModelProvider.OPENAI:
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        help="Required for GPT models. Set OPENAI_API_KEY env var to pre-fill.",
        value=Config.OPENAI_API_KEY,
    )
else:  # Ollama
    api_key = None  # Ollama doesn't need API key
    st.sidebar.info("💡 Ollama models run locally - no API key needed!\n\nMake sure Ollama is running: `ollama serve`")
    ollama_url = st.sidebar.text_input(
        "Ollama URL",
        value=Config.OLLAMA_BASE_URL,
        help="URL where Ollama is running",
    )

st.sidebar.divider()

# Generation Settings
st.sidebar.subheader("Generation Settings")

num_samples = st.sidebar.slider(
    "Number of samples",
    min_value=10,
    max_value=1000,
    value=100,
    step=10,
    help="Number of RLHF data points to generate.",
)

# Dynamic cost estimate based on slider value
if selected_model_config.requires_api_key:
    # Estimate: ~1000 input + ~1600 output tokens per data point (2 API calls)
    estimated_input = num_samples * 1000
    estimated_output = num_samples * 1600
    estimated_cost = (estimated_input / 1_000_000) * selected_model_config.input_price_per_million + \
                     (estimated_output / 1_000_000) * selected_model_config.output_price_per_million
    st.sidebar.caption(f"Estimated cost: ${estimated_cost:.2f}")
else:
    st.sidebar.caption("Estimated cost: Free (local model)")

# Domain Selection
st.sidebar.subheader("Domains")
domains = []
if st.sidebar.checkbox("pandas", value=True):
    domains.append("pandas")
if st.sidebar.checkbox("numpy", value=True):
    domains.append("numpy")
if st.sidebar.checkbox("sklearn", value=True):
    domains.append("sklearn")
if st.sidebar.checkbox("pytorch", value=True):
    domains.append("pytorch")

# Task Types
st.sidebar.subheader("Task Types")
task_types = []
if st.sidebar.checkbox("optimize", value=True):
    task_types.append("optimize")
if st.sidebar.checkbox("debug", value=True):
    task_types.append("debug")
if st.sidebar.checkbox("explain", value=True):
    task_types.append("explain")
if st.sidebar.checkbox("generate", value=True):
    task_types.append("generate")
if st.sidebar.checkbox("refactor", value=True):
    task_types.append("refactor")

st.sidebar.divider()

# Scoring Weights
st.sidebar.subheader("Scoring Weights")
efficiency_weight = st.sidebar.slider(
    "Efficiency weight",
    min_value=0.0,
    max_value=1.0,
    value=Config.EFFICIENCY_WEIGHT,
    step=0.1,
    help="Weight for code efficiency in total score",
)
clarity_weight = 1.0 - efficiency_weight
st.sidebar.caption(f"Clarity weight: {clarity_weight:.1f}")

st.sidebar.divider()

# Session Cost Tracker
st.sidebar.subheader("💰 Session Cost")
if selected_model_config.requires_api_key:
    # Calculate cost based on current model's pricing
    session_cost = st.session_state.session_usage.calculate_cost(
        selected_model_config.input_price_per_million,
        selected_model_config.output_price_per_million,
    )
    st.sidebar.metric(
        "Estimated Cost",
        f"${session_cost:.4f}",
        help="Estimated cost for this session based on token usage",
    )
    st.sidebar.caption(
        f"Input: {st.session_state.session_usage.input_tokens:,} tokens\n\n"
        f"Output: {st.session_state.session_usage.output_tokens:,} tokens"
    )
    if st.sidebar.button("🔄 Reset Cost Tracker", use_container_width=True):
        st.session_state.session_usage = TokenUsage()
        st.rerun()
else:
    st.sidebar.success("Free (local model)")


# =============================================================================
# MAIN CONTENT
# =============================================================================
st.title("🤖 RLHF Data Agent")
st.markdown(
    """
Generate preference-ranked coding data for RLHF/DPO training.
Specialized for ML/Data tasks (pandas, numpy, sklearn, pytorch).

**RLHF Relevance**: Each data point contains a prompt with two responses ranked by
efficiency and clarity, enabling direct use in preference learning pipelines.
""")

# Tabs for different views
tab_generate, tab_preview, tab_export, tab_about = st.tabs(
    ["📊 Generate", "👁️ Preview", "📥 Export", "ℹ️ About"]
)


# =============================================================================
# TAB: GENERATE
# =============================================================================
with tab_generate:
    st.header("Generate Dataset")

    # Show selected model info
    st.info(f"**Selected Model:** {selected_model_config.display_name} ({selected_model_config.provider.value})")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Validation
        needs_api_key = selected_model_config.requires_api_key
        has_api_key = api_key if needs_api_key else True

        if needs_api_key and not api_key:
            provider_name = selected_model_config.provider.value.capitalize()
            st.warning(f"⚠️ Please enter your {provider_name} API key in the sidebar.")
        if not domains:
            st.warning("⚠️ Please select at least one domain.")
        if not task_types:
            st.warning("⚠️ Please select at least one task type.")

        # Generation button
        can_generate = has_api_key and domains and task_types

        # Show either Generate or Stop button based on state
        if not st.session_state.generation_in_progress:
            if st.button(
                "🚀 Generate Data",
                disabled=not can_generate,
                type="primary",
                use_container_width=True,
            ):
                st.session_state.generation_in_progress = True
                st.session_state.stop_requested = False
                clear_stop_flag()  # Ensure clean start
                st.rerun()
        else:
            # Stop button - uses file-based flag for reliable interruption
            if st.button(
                "🛑 Stop Generation",
                type="secondary",
                use_container_width=True,
                key="stop_btn",
            ):
                set_stop_flag()
                st.session_state.stop_requested = True
                st.warning("⏳ Stopping generation after current API call completes...")
                st.rerun()  # Rerun to update UI immediately

            if check_stop_flag():
                st.warning("⏳ Stop requested - waiting for current API call to complete...")

        # Run generation if in progress
        if st.session_state.generation_in_progress:
            # Generate prompts
            with st.spinner("Generating prompts..."):
                prompts = list(
                    generate_prompts(
                        count=num_samples,
                        domains=domains,
                        task_types=task_types,
                    )
                )

            st.info(f"Generated {len(prompts)} unique prompts. Starting API calls with {selected_model_config.display_name}...")
            st.caption("💡 Click 'Stop Generation' to stop early and keep the data generated so far.")

            # Progress tracking - initialize with visible state
            progress_bar = st.progress(0, text="Starting generation...")
            status_text = st.empty()
            status_text.text(f"Processing: 0/{len(prompts)} (0.0%)")
            time_text = st.empty()
            time_text.text("⏱️ Estimated time: Calculating...")
            cost_text = st.empty()
            cost_text.text("💰 Current run cost: $0.0000")

            # Time tracking for estimation
            generation_start_time = time.time()

            def format_time(seconds: float) -> str:
                """Format seconds into human-readable time."""
                if seconds < 60:
                    return f"{int(seconds)}s"
                elif seconds < 3600:
                    mins = int(seconds // 60)
                    secs = int(seconds % 60)
                    return f"{mins}m {secs}s"
                else:
                    hours = int(seconds // 3600)
                    mins = int((seconds % 3600) // 60)
                    return f"{hours}h {mins}m"

            def update_progress(completed, total):
                progress = completed / total
                progress_bar.progress(progress, text=f"Generating data point {completed}/{total}")
                status_text.text(f"Processing: {completed}/{total} ({progress*100:.1f}%)")

                # Calculate time estimate after at least 3 samples for accuracy
                if completed >= 3:
                    elapsed = time.time() - generation_start_time
                    avg_time_per_item = elapsed / completed
                    remaining_items = total - completed
                    estimated_remaining = avg_time_per_item * remaining_items
                    time_text.text(f"⏱️ Elapsed: {format_time(elapsed)} | Remaining: ~{format_time(estimated_remaining)}")

            def update_cost(total_usage_so_far: TokenUsage):
                # Calculate current run cost and display it
                run_cost = total_usage_so_far.calculate_cost(
                    selected_model_config.input_price_per_million,
                    selected_model_config.output_price_per_million,
                )
                cost_text.text(f"💰 Current run cost: ${run_cost:.4f}")

            def check_stop():
                # Use file-based flag for reliable stop detection during sync operations
                return check_stop_flag()

            # Generate data points
            try:
                data_points, total_usage = generate_batch_sync(
                    prompts=prompts,
                    model_key=selected_model,
                    api_key=api_key if needs_api_key else None,
                    on_progress=update_progress,
                    should_stop=check_stop,
                    on_usage_update=update_cost,
                )

                st.session_state.generated_data = data_points
                # Update session usage with final total
                st.session_state.session_usage.add(total_usage)
                final_cost = total_usage.calculate_cost(
                    selected_model_config.input_price_per_million,
                    selected_model_config.output_price_per_million,
                )
                st.success(f"✅ Successfully generated {len(data_points)} data points! (Cost: ${final_cost:.4f})")

            except GenerationStoppedError as e:
                # Save partial results so user can export them
                if e.partial_results:
                    st.session_state.generated_data = e.partial_results
                    # Update session usage with partial usage
                    st.session_state.session_usage.add(e.total_usage)
                    partial_cost = e.total_usage.calculate_cost(
                        selected_model_config.input_price_per_million,
                        selected_model_config.output_price_per_million,
                    )
                    st.warning(f"⚠️ Generation stopped. Saved {len(e.partial_results)} data points. (Cost: ${partial_cost:.4f})")
                    st.info("You can export the partial data or start a new generation.")
                else:
                    st.warning("⚠️ Generation stopped before any data was generated.")

            except Exception as e:
                st.error(f"❌ Error during generation: {str(e)}")

            finally:
                st.session_state.generation_in_progress = False
                st.session_state.stop_requested = False
                clear_stop_flag()  # Clean up stop flag
                st.rerun()

    with col2:
        st.subheader("Current Dataset")
        if st.session_state.generated_data:
            stats = get_dataset_stats(st.session_state.generated_data)
            st.metric("Total Records", stats["total_records"])
            st.metric(
                "Avg Preference Gap",
                f"{stats['preference_gap']['mean']:.3f}",
                help="Average score difference between chosen and rejected",
            )

            # Domain breakdown
            st.markdown("**Domain Distribution:**")
            for domain, count in stats.get("domain_distribution", {}).items():
                st.caption(f"• {domain}: {count}")
        else:
            st.info("No data generated yet. Click 'Generate Data' to start.")


# =============================================================================
# TAB: PREVIEW
# =============================================================================
with tab_preview:
    st.header("Preview Data Points")

    if not st.session_state.generated_data:
        st.info("No data to preview. Generate data first.")
    else:
        # Sample selector
        num_preview = st.slider(
            "Number of samples to preview",
            min_value=1,
            max_value=min(10, len(st.session_state.generated_data)),
            value=3,
        )

        if st.button("🔄 Shuffle Samples"):
            st.rerun()

        # Show random samples
        samples = random.sample(
            st.session_state.generated_data,
            min(num_preview, len(st.session_state.generated_data)),
        )

        for i, dp in enumerate(samples):
            with st.expander(
                f"Sample {i+1}: {dp.metadata.domain} / {dp.metadata.task_type} / {dp.metadata.complexity}",
                expanded=(i == 0),
            ):
                st.markdown("**Prompt:**")
                st.code(dp.prompt, language="text")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**✅ Chosen Response:**")
                    st.markdown(
                        f"Score: `{dp.metadata.chosen_score.total:.3f}` "
                        f"(eff: {dp.metadata.chosen_score.efficiency:.2f}, "
                        f"clarity: {dp.metadata.chosen_score.clarity:.2f})"
                    )
                    st.code(dp.chosen[:1500] + "..." if len(dp.chosen) > 1500 else dp.chosen)

                with col2:
                    st.markdown("**❌ Rejected Response:**")
                    st.markdown(
                        f"Score: `{dp.metadata.rejected_score.total:.3f}` "
                        f"(eff: {dp.metadata.rejected_score.efficiency:.2f}, "
                        f"clarity: {dp.metadata.rejected_score.clarity:.2f})"
                    )
                    st.code(dp.rejected[:1500] + "..." if len(dp.rejected) > 1500 else dp.rejected)

                st.caption(f"SHA-256: `{dp.metadata.sha256[:16]}...`")


# =============================================================================
# TAB: EXPORT
# =============================================================================
with tab_export:
    st.header("Export Dataset")

    if not st.session_state.generated_data:
        st.info("No data to export. Generate data first.")
    else:
        st.markdown(
            f"**Ready to export {len(st.session_state.generated_data)} data points.**"
        )
        st.caption("💡 Click any download button to save the file to your computer.")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("📄 JSON")
            st.caption("Full data with nested metadata. Best for REPPO.")

            try:
                json_content = generate_json_content(st.session_state.generated_data)
                st.download_button(
                    "⬇️ Download JSON",
                    data=json_content,
                    file_name="rlhf_dataset.json",
                    mime="application/json",
                    use_container_width=True,
                )

                # Manifest download
                manifest_content = generate_manifest_content(st.session_state.generated_data)
                st.download_button(
                    "📋 Download Manifest",
                    data=manifest_content,
                    file_name="rlhf_dataset_manifest.json",
                    mime="application/json",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Export failed: {e}")

        with col2:
            st.subheader("📊 CSV")
            st.caption("Flattened format for spreadsheet analysis.")

            try:
                csv_content = generate_csv_content(st.session_state.generated_data)
                st.download_button(
                    "⬇️ Download CSV",
                    data=csv_content,
                    file_name="rlhf_dataset.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Export failed: {e}")

        with col3:
            st.subheader("🤗 HuggingFace")
            st.caption("JSONL format for HF datasets & trl library.")

            try:
                jsonl_content = generate_jsonl_content(st.session_state.generated_data)
                st.download_button(
                    "⬇️ Download JSONL",
                    data=jsonl_content,
                    file_name="rlhf_dataset.jsonl",
                    mime="application/jsonl",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Export failed: {e}")

        # Statistics
        st.divider()
        st.subheader("📈 Dataset Statistics")

        stats = get_dataset_stats(st.session_state.generated_data)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Score Statistics:**")
            st.json(
                {
                    "chosen_scores": stats["chosen_score_stats"],
                    "rejected_scores": stats["rejected_score_stats"],
                    "preference_gap": stats["preference_gap"],
                }
            )

        with col2:
            st.markdown("**Domain Distribution:**")
            st.json(stats["domain_distribution"])

        with col3:
            st.markdown("**Task Distribution:**")
            st.json(stats["task_type_distribution"])


# =============================================================================
# TAB: ABOUT
# =============================================================================
with tab_about:
    st.header("About RLHF Data Agent")

    st.markdown(
        """
    ## What is this?

    This tool generates **preference-ranked coding data** for training AI models using
    **RLHF (Reinforcement Learning from Human Feedback)** or **DPO (Direct Preference Optimization)**.

    ## Supported Models

    | Provider | Models | API Key Required |
    |----------|--------|------------------|
    | **Anthropic** | Claude Sonnet 4, Claude Haiku 4 | Yes |
    | **OpenAI** | GPT-4o, GPT-4o Mini | Yes |
    | **Ollama** | Llama 3.2, CodeLlama, DeepSeek Coder | No (local) |

    ### Using Ollama (Free Local Models)

    1. Install Ollama: https://ollama.ai
    2. Pull a model: `ollama pull llama3.2` or `ollama pull codellama`
    3. Start the server: `ollama serve`
    4. Select an Ollama model in the sidebar - no API key needed!

    ## How it works

    1. **Prompt Generation**: Creates diverse coding tasks across pandas, numpy, sklearn, and pytorch
    2. **Response Generation**: Uses the selected LLM to generate two responses per prompt with different styles
    3. **Heuristic Ranking**: Scores responses on efficiency and clarity metrics
    4. **Preference Pairing**: Higher-scored response becomes "chosen", lower becomes "rejected"

    ## Scoring Metrics

    ### Efficiency (default: 60%)
    - Line count (fewer = better)
    - Cyclomatic complexity (lower = better)
    - Anti-pattern detection (iterrows, repeated concat, etc.)
    - Vectorization usage (numpy/pandas operations)

    ### Clarity (default: 40%)
    - Docstring presence
    - Comment ratio (optimal: 5-20%)
    - Variable naming quality
    - Type hints
    - Modular design (functions)

    ## Data Format

    Each data point follows the DPO schema:
    ```json
    {
        "prompt": "User coding question",
        "chosen": "Better response",
        "rejected": "Worse response",
        "metadata": {
            "task_type": "optimize|debug|explain|generate|refactor",
            "domain": "pandas|numpy|sklearn|pytorch",
            "chosen_score": {"efficiency": 0.85, "clarity": 0.90, "total": 0.875},
            "rejected_score": {"efficiency": 0.60, "clarity": 0.70, "total": 0.65},
            "sha256": "verification_hash"
        }
    }
    ```

    ## REPPO Compatibility

    This tool is designed for publication on [reppo.ai](https://reppo.ai):
    - ✅ SHA-256 hashes for data integrity
    - ✅ Manifest file with dataset statistics
    - ✅ JSON export format
    - ✅ Comprehensive metadata
    """
    )

    st.divider()

    st.markdown(
        """
    ### Technical Details

    - **LLM Backends**: Anthropic Claude, OpenAI GPT, Ollama (local)
    - **Scoring**: Deterministic heuristics (no LLM-based evaluation)
    - **Rate Limiting**: Built-in delays to respect API limits
    - **Scalability**: Designed for 1000+ data points
    """
    )


# =============================================================================
# FOOTER
# =============================================================================
st.divider()
st.caption(
    "RLHF Data Agent | Built for ML/Data coding assistance training | "
    "Powered by REPPO"
)
