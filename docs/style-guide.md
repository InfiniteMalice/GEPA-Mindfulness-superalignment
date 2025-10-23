# GEPA Style Guide - Extended Examples and Patterns

**Version:** 1.0.0
**Companion to:** [CODING_STANDARDS.md](../CODING_STANDARDS.md)

This document provides extensive examples, anti-patterns, and best
practices for the GEPA Mindfulness Superalignment project.

## Table of Contents

- [Line Length Strategies](#line-length-strategies)
- [Import Patterns](#import-patterns)
- [Function Design](#function-design)
- [Class Design](#class-design)
- [Error Handling](#error-handling)
- [Testing Patterns](#testing-patterns)
- [Documentation Examples](#documentation-examples)
- [Domain-Specific Patterns](#domain-specific-patterns)
- [Anti-Patterns to Avoid](#anti-patterns-to-avoid)

---

## Line Length Strategies

### Strategy 1: Function Renaming

```python
# ❌ ANTI-PATTERN: Name too long
def calculate_mindfulness_alignment_score_with_temporal_decay_weighting(
    metrics: np.ndarray,
    temporal_weights: Dict[str, float],
) -> float:
    pass

# ✅ GOOD: Shortened name
def calc_mindful_align_temporal(
    metrics: np.ndarray,
    temp_weights: Dict[str, float],
) -> float:
    pass

# ✅ ALTERNATIVE: More descriptive but still short
def calc_temporal_mindful_score(
    metrics: np.ndarray,
    weights: Dict[str, float],
) -> float:
    """Calculate temporally-weighted mindfulness score.

    Note: Docstring clarifies what "temporal" means.
    """
    pass
```

### Strategy 2: Breaking Function Calls

```python
# ❌ ANTI-PATTERN: Long line
result = calculate_score(data, weights, threshold, decay_factor, normalize=True)

# ✅ GOOD: Vertical alignment
result = calculate_score(
    data,
    weights,
    threshold,
    decay_factor,
    normalize=True,
)

# ✅ ALTERNATIVE: Group related args
result = calculate_score(
    data, weights,
    threshold, decay_factor,
    normalize=True,
)
```

### Strategy 3: Long String Literals

```python
# ❌ ANTI-PATTERN: Single long string
error_msg = "Alignment score calculation failed: weights must sum to 1.0 and all values must be non-negative"

# ✅ GOOD: Implicit concatenation
error_msg = (
    "Alignment score calculation failed: "
    "weights must sum to 1.0 and all values "
    "must be non-negative"
)

# ✅ ALTERNATIVE: Multi-line f-string
error_msg = (
    f"Alignment score calculation failed for {model_name}: "
    f"weights must sum to 1.0 (got {weight_sum:.2f})"
)
```

### Strategy 4: Complex Conditionals

```python
# ❌ ANTI-PATTERN: Long conditional
if score > threshold and weights_valid and data_available and not is_corrupted:
    process_data()

# ✅ GOOD: Parenthesized continuation
if (
    score > threshold
    and weights_valid
    and data_available
    and not is_corrupted
):
    process_data()

# ✅ BETTER: Extract to variable
conditions_met = (
    score > threshold
    and weights_valid
    and data_available
    and not is_corrupted
)
if conditions_met:
    process_data()

# ✅ BEST: Extract to function
def should_process(score, threshold, weights_valid, data_available, is_corrupted):
    """Check if data processing conditions are met."""
    return (
        score > threshold
        and weights_valid
        and data_available
        and not is_corrupted
    )

if should_process(score, threshold, weights_valid, data_available, is_corrupted):
    process_data()
```

### Strategy 5: Dictionary Literals

```python
# ❌ ANTI-PATTERN: Long dictionary line
config = {"model": "mindfulness", "weights": weights, "threshold": 0.5, "normalize": True}

# ✅ GOOD: Multi-line dict
config = {
    "model": "mindfulness",
    "weights": weights,
    "threshold": 0.5,
    "normalize": True,
}
```

---

## Import Patterns

### Pattern 1: Standard Organization

```python
# ✅ PERFECT EXAMPLE
"""Module for mindfulness score calculation."""

# Standard library
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-party
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# Local application
from .config import Config, ModelConfig
from .preprocessing import clean_data, normalize
from .utils.helpers import log_metrics, validate_input
```

### Pattern 2: Conditional Imports

```python
# ✅ GOOD: Conditional imports for optional dependencies
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .advanced_models import AdvancedModel

# Regular imports
import numpy as np

from .config import Config
```

### Pattern 3: Aliasing

```python
# ✅ GOOD: Standard aliases
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ✅ GOOD: Descriptive aliases for name conflicts
from .models import Transformer as MindfulTransformer
from transformers import Transformer as HFTransformer
```

### Pattern 4: What NOT to Import

```python
# ❌ ANTI-PATTERN: Wildcard imports
from utils import *

# ❌ ANTI-PATTERN: Unused imports
import json
import os
import sys  # Never used

def process():
    return json.loads("{}")

# ❌ ANTI-PATTERN: Wrong order
from .config import Config
import numpy as np
import os

# ✅ CORRECT VERSION
import os

import numpy as np

from .config import Config
```

---

## Function Design

### Pattern 1: Clear Type Hints

```python
# ✅ EXCELLENT: Complete type hints
def calc_mindful_score(
    data: np.ndarray,
    weights: Dict[str, float],
    threshold: Optional[float] = None,
) -> Tuple[float, Dict[str, float]]:
    """Calculate mindfulness score with feature breakdown.

    Args:
        data: Input array of shape (n_samples, n_features)
        weights: Feature name to weight mapping
        threshold: Optional score threshold for filtering

    Returns:
        Tuple of (overall_score, feature_scores)
    """
    pass
```

### Pattern 2: Default Arguments

```python
# ❌ ANTI-PATTERN: Mutable default
def process(items: List[str] = []) -> List[str]:
    items.append("new")
    return items

# ✅ GOOD: Use None
def process(items: Optional[List[str]] = None) -> List[str]:
    if items is None:
        items = []
    items.append("new")
    return items

# ✅ ALTERNATIVE: Use tuple
def process(items: Tuple[str, ...] = ()) -> List[str]:
    items_list = list(items)
    items_list.append("new")
    return items_list
```

### Pattern 3: Single Responsibility

```python
# ❌ ANTI-PATTERN: Function does too much
def process_and_score_and_save(data, weights, path):
    """Process, score, and save results."""
    cleaned = clean(data)
    score = calculate(cleaned, weights)
    save(score, path)
    return score

# ✅ GOOD: Separate concerns
def process_data(data: np.ndarray) -> np.ndarray:
    """Clean and preprocess data."""
    return clean(data)

def calc_score(data: np.ndarray, weights: Dict) -> float:
    """Calculate alignment score."""
    return calculate(data, weights)

def save_results(score: float, path: Path) -> None:
    """Save score to file."""
    save(score, path)

# Usage
cleaned = process_data(data)
score = calc_score(cleaned, weights)
save_results(score, path)
```

---

## Class Design

### Pattern 1: Well-Documented Class

```python
# ✅ EXCELLENT EXAMPLE
class MindfulScorer:
    """Calculate mindfulness alignment scores.

    This class provides methods to compute weighted alignment
    scores from mindfulness metrics with optional temporal decay.

    Attributes:
        weights: Feature weights dictionary
        decay_rate: Temporal decay rate (0.0 to 1.0)
        normalize: Whether to normalize scores

    Example:
        >>> scorer = MindfulScorer(
        ...     weights={"attention": 0.5, "awareness": 0.5},
        ...     decay_rate=0.1,
        ... )
        >>> score = scorer.calc_score(data)
    """

    def __init__(
        self,
        weights: Dict[str, float],
        decay_rate: float = 0.0,
        normalize: bool = True,
    ) -> None:
        """Initialize scorer with weights and config.

        Args:
            weights: Feature name to weight mapping
            decay_rate: Temporal decay rate
            normalize: Whether to normalize output

        Raises:
            ValueError: If weights don't sum to 1.0
            ValueError: If decay_rate not in [0, 1]
        """
        self._validate_weights(weights)
        self._validate_decay(decay_rate)

        self.weights = weights
        self.decay_rate = decay_rate
        self.normalize = normalize

    def _validate_weights(self, weights: Dict[str, float]) -> None:
        """Validate weights sum to 1.0."""
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0, got {total:.4f}"
            )

    def _validate_decay(self, rate: float) -> None:
        """Validate decay rate in valid range."""
        if not 0.0 <= rate <= 1.0:
            raise ValueError(
                f"Decay rate must be in [0, 1], got {rate}"
            )

    def calc_score(self, data: np.ndarray) -> float:
        """Calculate alignment score from data.

        Args:
            data: Input array of shape (n_samples, n_features)

        Returns:
            Alignment score between 0.0 and 1.0
        """
        # Implementation
        return 0.0
```

### Pattern 2: Property Usage

```python
# ✅ GOOD: Use properties for computed attributes
class MindfulModel:
    """Mindfulness scoring model."""

    def __init__(self, weights: Dict[str, float]) -> None:
        self._weights = weights
        self._score_cache: Optional[float] = None

    @property
    def weights(self) -> Dict[str, float]:
        """Get weights dictionary."""
        return self._weights.copy()

    @property
    def total_weight(self) -> float:
        """Get sum of all weights."""
        return sum(self._weights.values())

    @property
    def is_valid(self) -> bool:
        """Check if model configuration is valid."""
        return abs(self.total_weight - 1.0) < 1e-6
```

---

## Error Handling

### Pattern 1: Specific Exceptions

```python
# ✅ GOOD: Raise specific exceptions
def calc_score(data: np.ndarray, weights: Dict) -> float:
    """Calculate alignment score."""
    if data.size == 0:
        raise ValueError("Input data cannot be empty")

    if not weights:
        raise ValueError("Weights dictionary cannot be empty")

    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Weights must sum to 1.0, got {total:.4f}"
        )

    return _compute_score(data, weights)
```

### Pattern 2: Exception Context

```python
# ✅ GOOD: Provide context in exceptions
def load_model(path: Path) -> torch.nn.Module:
    """Load model from path."""
    try:
        model = torch.load(path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Model file not found at {path}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model from {path}: {e}"
        ) from e

    return model
```

### Pattern 3: Custom Exceptions

```python
# ✅ GOOD: Domain-specific exceptions
class MindfulnessError(Exception):
    """Base exception for mindfulness module."""
    pass


class InvalidWeightsError(MindfulnessError):
    """Raised when weights are invalid."""
    pass


class TemporalDecayError(MindfulnessError):
    """Raised when temporal decay calculation fails."""
    pass


# Usage
def validate_weights(weights: Dict[str, float]) -> None:
    """Validate weights dictionary."""
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise InvalidWeightsError(
            f"Weights must sum to 1.0, got {total:.4f}"
        )
```

---

## Testing Patterns

### Pattern 1: Comprehensive Test Class

```python
# ✅ EXCELLENT: Well-organized test class
"""Tests for mindfulness scoring module."""

import unittest
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.scoring import MindfulScorer, calc_score


class TestMindfulScorer(unittest.TestCase):
    """Tests for MindfulScorer class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.valid_weights = {
            "attention": 0.5,
            "awareness": 0.5,
        }
        self.test_data = np.random.rand(100, 2)
        self.scorer = MindfulScorer(self.valid_weights)

    def test_init_valid_weights(self) -> None:
        """Test initialization with valid weights."""
        scorer = MindfulScorer(self.valid_weights)
        self.assertEqual(scorer.weights, self.valid_weights)

    def test_init_invalid_weights_sum(self) -> None:
        """Test initialization fails with invalid weight sum."""
        invalid_weights = {"attention": 0.6, "awareness": 0.6}
        with self.assertRaises(ValueError) as ctx:
            MindfulScorer(invalid_weights)
        self.assertIn("sum to 1.0", str(ctx.exception))

    def test_calc_score_valid_range(self) -> None:
        """Test calculated score is in valid range [0, 1]."""
        score = self.scorer.calc_score(self.test_data)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_calc_score_empty_data(self) -> None:
        """Test score calculation fails with empty data."""
        empty_data = np.array([])
        with self.assertRaises(ValueError) as ctx:
            self.scorer.calc_score(empty_data)
        self.assertIn("empty", str(ctx.exception).lower())

    @patch("src.scoring.normalize")
    def test_calc_score_normalization(self, mock_norm) -> None:
        """Test score normalization is called when enabled."""
        mock_norm.return_value = self.test_data
        scorer = MindfulScorer(
            self.valid_weights,
            normalize=True,
        )
        scorer.calc_score(self.test_data)
        mock_norm.assert_called_once()


# Pytest-style tests
@pytest.fixture
def scorer():
    """Create scorer fixture."""
    weights = {"attention": 0.5, "awareness": 0.5}
    return MindfulScorer(weights)


def test_scorer_initialization(scorer):
    """Test scorer initializes correctly."""
    assert scorer.weights == {"attention": 0.5, "awareness": 0.5}
    assert scorer.normalize is True


@pytest.mark.parametrize(
    "weights,should_raise",
    [
        ({"a": 0.5, "b": 0.5}, False),
        ({"a": 0.6, "b": 0.6}, True),
        ({"a": 1.0}, False),
        ({"a": 0.5}, True),
    ],
)
def test_weight_validation(weights, should_raise):
    """Test weight validation with various inputs."""
    if should_raise:
        with pytest.raises(ValueError):
            MindfulScorer(weights)
    else:
        scorer = MindfulScorer(weights)
        assert scorer.weights == weights
```

---

## Documentation Examples

### Pattern 1: Module Docstring

```python
# ✅ EXCELLENT: Complete module documentation
"""Mindfulness alignment scoring module.

This module provides functionality for calculating alignment scores
based on various mindfulness metrics. Scores are computed using
weighted combinations of features with optional temporal decay.

The main entry point is the `MindfulScorer` class, which handles
score calculation with configurable weighting schemes.

Example:
    Basic usage with default settings::

        from scoring import MindfulScorer

        weights = {"attention": 0.6, "awareness": 0.4}
        scorer = MindfulScorer(weights)
        score = scorer.calc_score(data)

Classes:
    MindfulScorer: Main scoring class with temporal weighting
    TemporalWeighter: Helper for applying temporal decay

Functions:
    calc_score: Convenience function for quick scoring
    validate_weights: Validate weight dictionary

Constants:
    DEFAULT_DECAY_RATE: Default temporal decay rate (0.1)
    MIN_SCORE: Minimum valid score (0.0)
    MAX_SCORE: Maximum valid score (1.0)
"""

from typing import Dict

import numpy as np

# Constants
DEFAULT_DECAY_RATE = 0.1
MIN_SCORE = 0.0
MAX_SCORE = 1.0
```

### Pattern 2: Complex Function Documentation

```python
# ✅ EXCELLENT: Comprehensive function docs
def calc_temporal_weighted_score(
    data: np.ndarray,
    weights: Dict[str, float],
    timestamps: np.ndarray,
    decay_rate: float = 0.1,
    normalize: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """Calculate temporally-weighted mindfulness score.

    Computes a weighted alignment score from mindfulness metrics,
    applying exponential temporal decay to weight recent observations
    more heavily than older ones.

    The temporal weighting follows the formula:
        w(t) = exp(-decay_rate * (t_now - t))

    Args:
        data: Input array of shape (n_samples, n_features)
            containing feature values for each sample
        weights: Dictionary mapping feature names to weights.
            Must sum to 1.0
        timestamps: Array of timestamps of shape (n_samples,)
            in seconds since epoch
        decay_rate: Rate of exponential temporal decay.
            Higher values weight recent samples more heavily.
            Must be in range [0, 1]. Default: 0.1
        normalize: Whether to normalize the final score to
            range [0, 1]. Default: True

    Returns:
        Tuple containing:
            - overall_score: Weighted score between 0.0 and 1.0
            - feature_scores: Dictionary mapping feature names
              to their individual scores

    Raises:
        ValueError: If data is empty
        ValueError: If weights don't sum to 1.0
        ValueError: If decay_rate not in [0, 1]
        ValueError: If timestamps and data shapes don't match

    Example:
        >>> data = np.random.rand(100, 2)
        >>> weights = {"attention": 0.6, "awareness": 0.4}
        >>> timestamps = np.arange(100)
        >>> score, features = calc_temporal_weighted_score(
        ...     data, weights, timestamps, decay_rate=0.1
        ... )
        >>> print(f"Score: {score:.2f}")
        Score: 0.73
        >>> print(f"Features: {features}")
        Features: {'attention': 0.75, 'awareness': 0.70}

    Note:
        This function applies temporal weighting before computing
        the weighted sum of features. For non-temporal scoring,
        use `calc_score()` instead.

    See Also:
        calc_score: Basic scoring without temporal weighting
        MindfulScorer: Class-based interface with caching
    """
    # Implementation
    pass
```

---

## Domain-Specific Patterns

### Pattern 1: Mindfulness Metrics

```python
# ✅ GOOD: Clear mindfulness-specific code
from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class MindfulMetrics:
    """Container for mindfulness metrics.

    Attributes:
        attention: Attention score (0.0 to 1.0)
        awareness: Awareness score (0.0 to 1.0)
        presence: Present-moment focus (0.0 to 1.0)
        acceptance: Acceptance level (0.0 to 1.0)
    """

    attention: float
    awareness: float
    presence: float
    acceptance: float

    def validate(self) -> None:
        """Validate all metrics are in [0, 1]."""
        for name, value in self.__dict__.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be in [0, 1], got {value}"
                )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "attention": self.attention,
            "awareness": self.awareness,
            "presence": self.presence,
            "acceptance": self.acceptance,
        }
```

### Pattern 2: Alignment Scoring

```python
# ✅ GOOD: Clear alignment scoring patterns
class AlignmentScorer:
    """Calculate AI alignment scores for mindfulness models.

    Evaluates how well model outputs align with intended
    mindfulness objectives using multiple metrics.
    """

    def __init__(
        self,
        target_metrics: MindfulMetrics,
        tolerance: float = 0.1,
    ) -> None:
        """Initialize alignment scorer.

        Args:
            target_metrics: Target mindfulness metrics
            tolerance: Acceptable deviation from targets
        """
        self.target = target_metrics
        self.tolerance = tolerance

    def calc_alignment(
        self,
        predicted: MindfulMetrics,
    ) -> float:
        """Calculate alignment score.

        Computes how closely predicted metrics align with
        target metrics, with tolerance for small deviations.

        Args:
            predicted: Predicted mindfulness metrics

        Returns:
            Alignment score between 0.0 (poor) and 1.0 (perfect)
        """
        target_dict = self.target.to_dict()
        pred_dict = predicted.to_dict()

        deviations = [
            abs(target_dict[k] - pred_dict[k])
            for k in target_dict
        ]

        avg_dev = np.mean(deviations)
        alignment = max(0.0, 1.0 - avg_dev / self.tolerance)

        return alignment
```

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Magic Numbers

```python
# ❌ BAD: Magic numbers
def calc_score(data):
    if data.shape[1] != 5:
        raise ValueError("Wrong shape")
    score = data.mean() * 0.7 + data.std() * 0.3
    return min(max(score, 0.0), 1.0)

# ✅ GOOD: Named constants
N_FEATURES = 5
MEAN_WEIGHT = 0.7
STD_WEIGHT = 0.3
MIN_SCORE = 0.0
MAX_SCORE = 1.0


def calc_score(data: np.ndarray) -> float:
    """Calculate weighted score from data statistics."""
    if data.shape[1] != N_FEATURES:
        raise ValueError(
            f"Expected {N_FEATURES} features, "
            f"got {data.shape[1]}"
        )

    score = (
        data.mean() * MEAN_WEIGHT
        + data.std() * STD_WEIGHT
    )

    return np.clip(score, MIN_SCORE, MAX_SCORE)
```

### Anti-Pattern 2: Deep Nesting

```python
# ❌ BAD: Deep nesting
def process(data, config):
    if data is not None:
        if len(data) > 0:
            if config.is_valid():
                if config.normalize:
                    data = normalize(data)
                    if data.mean() > 0:
                        return calc_score(data)
    return None

# ✅ GOOD: Early returns
def process(
    data: Optional[np.ndarray],
    config: Config,
) -> Optional[float]:
    """Process data and return score if valid."""
    if data is None or len(data) == 0:
        return None

    if not config.is_valid():
        return None

    if config.normalize:
        data = normalize(data)

    if data.mean() <= 0:
        return None

    return calc_score(data)
```

### Anti-Pattern 3: God Objects

```python
# ❌ BAD: Class does everything
class MindfulnessSystem:
    """Does everything related to mindfulness."""

    def load_data(self, path):
        pass

    def preprocess(self, data):
        pass

    def train_model(self, data):
        pass

    def evaluate(self, model, data):
        pass

    def save_results(self, results, path):
        pass

    def visualize(self, results):
        pass

# ✅ GOOD: Separate responsibilities
class DataLoader:
    """Load and validate mindfulness data."""
    pass


class Preprocessor:
    """Preprocess mindfulness metrics."""
    pass


class ModelTrainer:
    """Train mindfulness models."""
    pass


class Evaluator:
    """Evaluate model performance."""
    pass


class ResultsManager:
    """Save and load results."""
    pass


class Visualizer:
    """Create visualizations."""
    pass
```

---

## Summary

**Key Takeaways:**

1. **Line length is absolute**: Always ≤100 characters
2. **Imports are organized**: stdlib → 3rd-party → local
3. **Type hints are required**: On all public functions
4. **Documentation is comprehensive**: Module, class, function
5. **Tests are thorough**: Cover edge cases and errors
6. **Code is clear**: Prefer explicit over clever

**For More Information:**

- [CODING_STANDARDS.md](../CODING_STANDARDS.md) - Core standards
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution workflow

---

**Questions or suggestions? Open an issue!**
