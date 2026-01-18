"""
Prompt templates for ML/Data coding tasks.
Generates diverse prompts across pandas, numpy, sklearn, and pytorch domains.

RLHF relevance: Diverse prompts ensure the trained model generalizes across
different coding scenarios, task types, and complexity levels.
"""
import random
from dataclasses import dataclass
from typing import Iterator


@dataclass
class PromptTemplate:
    """A single prompt template with metadata."""

    domain: str
    task_type: str
    complexity: str
    template: str
    variables: dict


# =============================================================================
# PANDAS TEMPLATES
# =============================================================================
PANDAS_TEMPLATES = [
    # OPTIMIZE tasks
    PromptTemplate(
        domain="pandas",
        task_type="optimize",
        complexity="beginner",
        template="Optimize this pandas code that filters a DataFrame:\n```python\nresult = []\nfor idx, row in df.iterrows():\n    if row['{column}'] > {threshold}:\n        result.append(row)\nresult_df = pd.DataFrame(result)\n```",
        variables={"column": ["age", "price", "score", "count"], "threshold": [10, 50, 100, 500]},
    ),
    PromptTemplate(
        domain="pandas",
        task_type="optimize",
        complexity="intermediate",
        template="Optimize this pandas code that calculates rolling statistics:\n```python\nresults = []\nfor i in range(len(df)):\n    window = df['{column}'].iloc[max(0, i-{window_size}):i+1]\n    results.append(window.mean())\ndf['rolling_mean'] = results\n```",
        variables={"column": ["value", "price", "temperature", "sales"], "window_size": [7, 14, 30]},
    ),
    PromptTemplate(
        domain="pandas",
        task_type="optimize",
        complexity="advanced",
        template="Optimize this pandas code for memory efficiency when processing a large CSV with {rows} rows:\n```python\ndf = pd.read_csv('large_file.csv')\ndf['{column}'] = df['{column}'].apply(lambda x: str(x).strip().lower())\ngrouped = df.groupby('{group_col}').apply(lambda g: g.sort_values('{sort_col}').head(10))\n```",
        variables={
            "rows": ["1M", "10M", "100M"],
            "column": ["name", "category", "description"],
            "group_col": ["user_id", "product_id", "region"],
            "sort_col": ["timestamp", "value", "score"],
        },
    ),
    # DEBUG tasks
    PromptTemplate(
        domain="pandas",
        task_type="debug",
        complexity="beginner",
        template="Fix the bug in this pandas code that merges two DataFrames:\n```python\ndf1 = pd.DataFrame({{'id': [1, 2, 3], 'value': [10, 20, 30]}})\ndf2 = pd.DataFrame({{'ID': [1, 2, 4], 'name': ['a', 'b', 'c']}})\nresult = df1.merge(df2, on='id')\n```\nExpected: Merge should match id=1 and id=2",
        variables={},
    ),
    PromptTemplate(
        domain="pandas",
        task_type="debug",
        complexity="intermediate",
        template="Fix the SettingWithCopyWarning in this pandas code:\n```python\ndf = pd.read_csv('data.csv')\nfiltered = df[df['{column}'] > {threshold}]\nfiltered['{new_col}'] = filtered['{column}'] * 2\n```",
        variables={
            "column": ["price", "quantity", "score"],
            "threshold": [0, 100, 1000],
            "new_col": ["doubled", "adjusted", "scaled"],
        },
    ),
    PromptTemplate(
        domain="pandas",
        task_type="debug",
        complexity="advanced",
        template="Debug this pandas code that's producing incorrect aggregation results:\n```python\ndf['date'] = pd.to_datetime(df['date'])\nmonthly = df.groupby(df['date'].dt.month).agg({{\n    '{value_col}': 'sum',\n    '{count_col}': 'count'\n}})\n# Issue: December 2023 and December 2024 are being combined\n```",
        variables={
            "value_col": ["revenue", "sales", "amount"],
            "count_col": ["transactions", "orders", "events"],
        },
    ),
    # EXPLAIN tasks
    PromptTemplate(
        domain="pandas",
        task_type="explain",
        complexity="beginner",
        template="Explain what this pandas code does and when you would use it:\n```python\ndf.{method}()\n```",
        variables={"method": ["describe()", "info()", "head(10)", "dtypes", "shape", "columns.tolist()"]},
    ),
    PromptTemplate(
        domain="pandas",
        task_type="explain",
        complexity="intermediate",
        template="Explain the difference between these two pandas operations and when to use each:\n```python\n# Option 1\ndf.groupby('{column}').transform('{agg}')\n\n# Option 2\ndf.groupby('{column}').agg('{agg}')\n```",
        variables={"column": ["category", "user_id", "region"], "agg": ["mean", "sum", "count"]},
    ),
    # GENERATE tasks
    PromptTemplate(
        domain="pandas",
        task_type="generate",
        complexity="beginner",
        template="Write pandas code to load a CSV file and display basic statistics for the '{column}' column.",
        variables={"column": ["price", "age", "score", "temperature"]},
    ),
    PromptTemplate(
        domain="pandas",
        task_type="generate",
        complexity="intermediate",
        template="Write pandas code to pivot a DataFrame with '{index}' as rows, '{columns}' as columns, and '{values}' as values, filling missing values with 0.",
        variables={
            "index": ["date", "user_id", "product"],
            "columns": ["category", "region", "status"],
            "values": ["amount", "count", "revenue"],
        },
    ),
    PromptTemplate(
        domain="pandas",
        task_type="generate",
        complexity="advanced",
        template="Write pandas code to perform time series resampling: convert {freq_from} data to {freq_to}, handling missing values with {fill_method}.",
        variables={
            "freq_from": ["hourly", "daily", "minute"],
            "freq_to": ["daily", "weekly", "monthly"],
            "fill_method": ["forward fill", "interpolation", "mean of adjacent values"],
        },
    ),
    # REFACTOR tasks
    PromptTemplate(
        domain="pandas",
        task_type="refactor",
        complexity="intermediate",
        template="Refactor this pandas code to use method chaining:\n```python\ndf = pd.read_csv('data.csv')\ndf = df.dropna()\ndf = df[df['{column}'] > {threshold}]\ndf = df.sort_values('{sort_col}')\ndf = df.reset_index(drop=True)\n```",
        variables={
            "column": ["value", "score", "amount"],
            "threshold": [0, 10, 100],
            "sort_col": ["date", "id", "name"],
        },
    ),
]

# =============================================================================
# NUMPY TEMPLATES
# =============================================================================
NUMPY_TEMPLATES = [
    # OPTIMIZE tasks
    PromptTemplate(
        domain="numpy",
        task_type="optimize",
        complexity="beginner",
        template="Optimize this numpy code that calculates element-wise operations:\n```python\nresult = []\nfor i in range(len(arr)):\n    result.append(arr[i] ** 2 + {constant})\nresult = np.array(result)\n```",
        variables={"constant": [1, 10, 100]},
    ),
    PromptTemplate(
        domain="numpy",
        task_type="optimize",
        complexity="intermediate",
        template="Optimize this numpy code that finds indices where condition is met:\n```python\nindices = []\nfor i in range(arr.shape[0]):\n    for j in range(arr.shape[1]):\n        if arr[i, j] > {threshold}:\n            indices.append((i, j))\n```",
        variables={"threshold": [0, 0.5, 0.9]},
    ),
    PromptTemplate(
        domain="numpy",
        task_type="optimize",
        complexity="advanced",
        template="Optimize this numpy code for batch matrix operations on {batch_size} matrices:\n```python\nresults = []\nfor i in range(len(matrices)):\n    result = np.dot(matrices[i], weights)\n    result = np.maximum(result, 0)  # ReLU\n    results.append(result)\nresults = np.array(results)\n```",
        variables={"batch_size": [100, 1000, 10000]},
    ),
    # DEBUG tasks
    PromptTemplate(
        domain="numpy",
        task_type="debug",
        complexity="beginner",
        template="Fix the broadcasting error in this numpy code:\n```python\na = np.array([[1, 2, 3], [4, 5, 6]])\nb = np.array([1, 2])\nresult = a + b\n```",
        variables={},
    ),
    PromptTemplate(
        domain="numpy",
        task_type="debug",
        complexity="intermediate",
        template="Debug this numpy code that's supposed to normalize each row:\n```python\narr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])\nnormalized = arr / arr.sum(axis={axis})\n# Expected: Each row should sum to 1\n```",
        variables={"axis": [0, 1]},
    ),
    # GENERATE tasks
    PromptTemplate(
        domain="numpy",
        task_type="generate",
        complexity="beginner",
        template="Write numpy code to create a {size}x{size} identity matrix and compute its eigenvalues.",
        variables={"size": [3, 5, 10]},
    ),
    PromptTemplate(
        domain="numpy",
        task_type="generate",
        complexity="intermediate",
        template="Write numpy code to implement {operation} without using np.{operation}.",
        variables={"operation": ["softmax", "sigmoid", "relu", "tanh"]},
    ),
    PromptTemplate(
        domain="numpy",
        task_type="generate",
        complexity="advanced",
        template="Write numpy code to implement efficient batched {metric} distance calculation between two sets of vectors.",
        variables={"metric": ["euclidean", "cosine", "manhattan"]},
    ),
    # EXPLAIN tasks
    PromptTemplate(
        domain="numpy",
        task_type="explain",
        complexity="intermediate",
        template="Explain the memory layout difference between these two arrays and its performance implications:\n```python\narr_c = np.array([[1,2,3],[4,5,6]], order='C')\narr_f = np.array([[1,2,3],[4,5,6]], order='F')\n```",
        variables={},
    ),
    # REFACTOR tasks
    PromptTemplate(
        domain="numpy",
        task_type="refactor",
        complexity="intermediate",
        template="Refactor this numpy code to avoid creating intermediate arrays:\n```python\na = arr * 2\nb = a + 10\nc = np.sqrt(b)\nd = c / c.max()\nresult = d\n```",
        variables={},
    ),
]

# =============================================================================
# SKLEARN TEMPLATES
# =============================================================================
SKLEARN_TEMPLATES = [
    # OPTIMIZE tasks
    PromptTemplate(
        domain="sklearn",
        task_type="optimize",
        complexity="beginner",
        template="Optimize this sklearn code that scales features:\n```python\nfrom sklearn.preprocessing import StandardScaler\nscalers = {{}}\nfor col in df.columns:\n    scaler = StandardScaler()\n    df[col] = scaler.fit_transform(df[[col]])\n    scalers[col] = scaler\n```",
        variables={},
    ),
    PromptTemplate(
        domain="sklearn",
        task_type="optimize",
        complexity="intermediate",
        template="Optimize this sklearn cross-validation code for faster execution:\n```python\nfrom sklearn.model_selection import cross_val_score\nfrom sklearn.ensemble import RandomForestClassifier\n\nscores = []\nfor n_estimators in [{est_range}]:\n    model = RandomForestClassifier(n_estimators=n_estimators)\n    score = cross_val_score(model, X, y, cv=5).mean()\n    scores.append((n_estimators, score))\n```",
        variables={"est_range": ["10, 50, 100, 200", "50, 100, 150, 200, 250"]},
    ),
    PromptTemplate(
        domain="sklearn",
        task_type="optimize",
        complexity="advanced",
        template="Optimize this sklearn pipeline for a dataset with {n_features} features and {n_samples} samples:\n```python\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.decomposition import PCA\nfrom sklearn.ensemble import GradientBoostingClassifier\n\npipe = Pipeline([\n    ('scaler', StandardScaler()),\n    ('pca', PCA(n_components=0.95)),\n    ('clf', GradientBoostingClassifier())\n])\npipe.fit(X_train, y_train)\n```",
        variables={"n_features": [100, 500, 1000], "n_samples": ["10K", "100K", "1M"]},
    ),
    # DEBUG tasks
    PromptTemplate(
        domain="sklearn",
        task_type="debug",
        complexity="beginner",
        template="Fix the data leakage bug in this sklearn code:\n```python\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LogisticRegression\n\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)\nX_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)\nmodel = LogisticRegression().fit(X_train, y_train)\nprint(model.score(X_test, y_test))\n```",
        variables={},
    ),
    PromptTemplate(
        domain="sklearn",
        task_type="debug",
        complexity="intermediate",
        template="Debug this sklearn code that's giving inconsistent results:\n```python\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.model_selection import cross_val_score\n\nmodel = RandomForestClassifier(n_estimators=100)\nscores = cross_val_score(model, X, y, cv=5)\nprint(f'Scores: {{scores}}')\n# Running this multiple times gives different results\n```",
        variables={},
    ),
    # GENERATE tasks
    PromptTemplate(
        domain="sklearn",
        task_type="generate",
        complexity="beginner",
        template="Write sklearn code to train a {model_type} classifier and evaluate it with a classification report.",
        variables={"model_type": ["logistic regression", "decision tree", "random forest", "SVM"]},
    ),
    PromptTemplate(
        domain="sklearn",
        task_type="generate",
        complexity="intermediate",
        template="Write sklearn code to perform {search_type} hyperparameter search for a {model_type} model.",
        variables={
            "search_type": ["grid search", "random search"],
            "model_type": ["RandomForest", "GradientBoosting", "SVM"],
        },
    ),
    PromptTemplate(
        domain="sklearn",
        task_type="generate",
        complexity="advanced",
        template="Write sklearn code to create a custom transformer that {transformation} and integrate it into a Pipeline.",
        variables={
            "transformation": [
                "applies log transform to skewed features",
                "creates polynomial interaction features",
                "handles missing values based on feature type",
            ]
        },
    ),
    # EXPLAIN tasks
    PromptTemplate(
        domain="sklearn",
        task_type="explain",
        complexity="intermediate",
        template="Explain the difference between these sklearn cross-validation strategies and when to use each:\n```python\nfrom sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit\n```",
        variables={},
    ),
    # REFACTOR tasks
    PromptTemplate(
        domain="sklearn",
        task_type="refactor",
        complexity="advanced",
        template="Refactor this sklearn code to use ColumnTransformer for mixed data types:\n```python\n# Numeric columns\nnum_cols = ['age', 'income', 'score']\nscaler = StandardScaler()\nX_num = scaler.fit_transform(df[num_cols])\n\n# Categorical columns\ncat_cols = ['gender', 'region', 'category']\nencoder = OneHotEncoder()\nX_cat = encoder.fit_transform(df[cat_cols])\n\n# Combine\nX = np.hstack([X_num, X_cat.toarray()])\n```",
        variables={},
    ),
]

# =============================================================================
# PYTORCH TEMPLATES
# =============================================================================
PYTORCH_TEMPLATES = [
    # OPTIMIZE tasks
    PromptTemplate(
        domain="pytorch",
        task_type="optimize",
        complexity="beginner",
        template="Optimize this PyTorch code for GPU memory efficiency:\n```python\nresults = []\nfor batch in dataloader:\n    x = batch.to(device)\n    output = model(x)\n    results.append(output)\nall_results = torch.cat(results)\n```",
        variables={},
    ),
    PromptTemplate(
        domain="pytorch",
        task_type="optimize",
        complexity="intermediate",
        template="Optimize this PyTorch training loop:\n```python\nfor epoch in range({epochs}):\n    for batch_idx, (data, target) in enumerate(train_loader):\n        optimizer.zero_grad()\n        output = model(data)\n        loss = criterion(output, target)\n        loss.backward()\n        optimizer.step()\n        print(f'Batch {{batch_idx}}, Loss: {{loss.item()}}')\n```",
        variables={"epochs": [10, 50, 100]},
    ),
    PromptTemplate(
        domain="pytorch",
        task_type="optimize",
        complexity="advanced",
        template="Optimize this PyTorch code for multi-GPU training with {num_gpus} GPUs:\n```python\nmodel = MyModel()\nmodel = model.to('cuda')\n\nfor epoch in range(epochs):\n    for data, target in dataloader:\n        data, target = data.to('cuda'), target.to('cuda')\n        output = model(data)\n        loss = criterion(output, target)\n        loss.backward()\n        optimizer.step()\n```",
        variables={"num_gpus": [2, 4, 8]},
    ),
    # DEBUG tasks
    PromptTemplate(
        domain="pytorch",
        task_type="debug",
        complexity="beginner",
        template="Fix the gradient issue in this PyTorch code:\n```python\nx = torch.tensor([1.0, 2.0, 3.0])\ny = x ** 2\nloss = y.sum()\nloss.backward()\nprint(x.grad)  # Returns None\n```",
        variables={},
    ),
    PromptTemplate(
        domain="pytorch",
        task_type="debug",
        complexity="intermediate",
        template="Debug this PyTorch model that's not learning:\n```python\nclass Model(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc1 = nn.Linear({input_dim}, 128)\n        self.fc2 = nn.Linear(128, {output_dim})\n\n    def forward(self, x):\n        x = self.fc1(x)\n        x = torch.relu(x)\n        x = self.fc2(x)\n        return x\n\n# Training shows loss not decreasing\n```",
        variables={"input_dim": [10, 100, 784], "output_dim": [2, 10, 100]},
    ),
    # GENERATE tasks
    PromptTemplate(
        domain="pytorch",
        task_type="generate",
        complexity="beginner",
        template="Write PyTorch code to create a simple {layer_count}-layer neural network for {task_type}.",
        variables={
            "layer_count": [2, 3, 4],
            "task_type": ["binary classification", "multi-class classification", "regression"],
        },
    ),
    PromptTemplate(
        domain="pytorch",
        task_type="generate",
        complexity="intermediate",
        template="Write PyTorch code to implement a custom Dataset class for {data_type} data.",
        variables={"data_type": ["image", "text", "tabular", "time series"]},
    ),
    PromptTemplate(
        domain="pytorch",
        task_type="generate",
        complexity="advanced",
        template="Write PyTorch code to implement {architecture} from scratch.",
        variables={"architecture": ["self-attention mechanism", "residual block", "batch normalization layer"]},
    ),
    # EXPLAIN tasks
    PromptTemplate(
        domain="pytorch",
        task_type="explain",
        complexity="beginner",
        template="Explain the difference between `model.train()` and `model.eval()` in PyTorch and when to use each.",
        variables={},
    ),
    PromptTemplate(
        domain="pytorch",
        task_type="explain",
        complexity="intermediate",
        template="Explain what happens in this PyTorch autograd example:\n```python\nx = torch.tensor([2.0], requires_grad=True)\ny = x ** 3 + 2 * x ** 2 + x\ny.backward()\nprint(x.grad)\n```",
        variables={},
    ),
    # REFACTOR tasks
    PromptTemplate(
        domain="pytorch",
        task_type="refactor",
        complexity="intermediate",
        template="Refactor this PyTorch code to use nn.Sequential:\n```python\nclass Model(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc1 = nn.Linear(784, 256)\n        self.bn1 = nn.BatchNorm1d(256)\n        self.fc2 = nn.Linear(256, 128)\n        self.bn2 = nn.BatchNorm1d(128)\n        self.fc3 = nn.Linear(128, 10)\n\n    def forward(self, x):\n        x = torch.relu(self.bn1(self.fc1(x)))\n        x = torch.relu(self.bn2(self.fc2(x)))\n        x = self.fc3(x)\n        return x\n```",
        variables={},
    ),
]

# =============================================================================
# PROMPT GENERATOR
# =============================================================================

ALL_TEMPLATES = PANDAS_TEMPLATES + NUMPY_TEMPLATES + SKLEARN_TEMPLATES + PYTORCH_TEMPLATES


def expand_template(template: PromptTemplate) -> str:
    """
    Expand a template by filling in random variable values.
    Returns a concrete prompt string.
    """
    prompt = template.template
    for var_name, var_options in template.variables.items():
        if var_options:
            chosen_value = random.choice(var_options)
            prompt = prompt.replace(f"{{{var_name}}}", str(chosen_value))
    return prompt


def generate_prompts(
    count: int,
    domains: "list[str] | None" = None,
    task_types: "list[str] | None" = None,
) -> "Iterator[tuple[str, str, str, str]]":
    """
    Generate specified number of unique prompts.

    Args:
        count: Number of prompts to generate
        domains: Filter by domains (default: all)
        task_types: Filter by task types (default: all)

    Yields:
        Tuples of (prompt, domain, task_type, complexity)

    RLHF relevance: Diverse prompts across domains and complexities ensure
    the preference model learns generalizable code quality judgments.
    """
    # Filter templates
    templates = ALL_TEMPLATES
    if domains:
        templates = [t for t in templates if t.domain in domains]
    if task_types:
        templates = [t for t in templates if t.task_type in task_types]

    if not templates:
        raise ValueError("No templates match the specified filters")

    generated = set()
    attempts = 0
    max_attempts = count * 10  # Prevent infinite loops

    while len(generated) < count and attempts < max_attempts:
        template = random.choice(templates)
        prompt = expand_template(template)

        # Ensure uniqueness
        if prompt not in generated:
            generated.add(prompt)
            yield (prompt, template.domain, template.task_type, template.complexity)

        attempts += 1

    if len(generated) < count:
        print(f"Warning: Could only generate {len(generated)}/{count} unique prompts")
