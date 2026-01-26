# Contributing to GEPA Mindfulness Superalignment

Thank you for your interest in contributing to the GEPA Mindfulness
Superalignment project! This guide will help you get started.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Pull Request Process](#pull-request-process)
- [Review Process](#review-process)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)

---

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the project
- Show empathy towards other contributors

---

## Getting Started

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Development Dependencies

```bash
# Code formatting and linting
pip install black isort flake8 mypy

# Testing
pip install pytest pytest-cov

# Pre-commit hooks (recommended)
pip install pre-commit
pre-commit install
```

### Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR-USERNAME/GEPA-Mindfulness-superalignment.git
cd GEPA-Mindfulness-superalignment

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL-OWNER/GEPA-Mindfulness-superalignment.git
```

---

## Development Workflow

### 1. Create a Feature Branch

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create a feature branch
git checkout -b feature/your-feature-name
```

### Branch Naming Conventions

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test additions or changes

Examples:
- `feature/add-temporal-weighting`
- `fix/alignment-score-calculation`
- `docs/update-api-reference`

### 2. Make Your Changes

Follow the [CODING_STANDARDS.md](./CODING_STANDARDS.md) strictly:

- Maximum 100 characters per line for Python files
- Proper import organization
- No unused imports
- Black formatting
- Type hints on all public functions
- Comprehensive docstrings

### 3. Format Your Code

```bash
# Format with Black
black --line-length 100 .

# Sort imports
isort --profile black --line-length 100 .

# Check type hints
mypy src/

# Run linter
flake8 --max-line-length=100 src/
```

### 4. Write Tests

All new code must include tests:

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Aim for >80% coverage on new code
```

### 5. Commit Your Changes

Write clear, descriptive commit messages:

```bash
# Good commit messages
git commit -m "Add temporal weighting to alignment score"
git commit -m "Fix: Resolve NaN values in mindfulness calc"
git commit -m "Docs: Update API reference for scoring module"

# Bad commit messages (avoid these)
git commit -m "Update code"
git commit -m "Fix bug"
git commit -m "WIP"
```

**Commit Message Format:**

```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `style:` Formatting changes
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance tasks

**Example:**

```
feat: Add temporal weighting to alignment scores

Implement time-based decay function for alignment scoring
to give more weight to recent mindfulness metrics.

- Add calc_temporal_weight() function
- Update calc_mindful_score() to use temporal weights
- Add tests for temporal weighting
- Update documentation

Closes #123
```

---

## Coding Standards

### Required Reading

**Before contributing, you MUST read:**
- [CODING_STANDARDS.md](./CODING_STANDARDS.md) - Complete style guide

### Key Requirements Summary

#### 1. Line Length: 100 Characters Maximum (Python Only)

```python
# ❌ WILL BE REJECTED
def calculate_mindfulness_alignment_score_with_temporal_weighting_function(data, weights):
    pass

# ✅ WILL BE ACCEPTED
def calc_mindful_align_score_temporal(data, weights):
    pass
```

#### 2. Import Organization

```python
# Standard library
import json
import os
from typing import Dict, List

# Third-party
import numpy as np
import torch

# Local
from .config import Config
from .utils import helpers
```

#### 3. Type Hints Required

```python
# ✅ Required
def process(data: np.ndarray) -> Dict[str, float]:
    pass

# ❌ Will be rejected
def process(data):
    pass
```

#### 4. Docstrings Required

```python
def calc_score(data: np.ndarray) -> float:
    """Calculate mindfulness score from data.

    Args:
        data: Input array of shape (n_samples, n_features)

    Returns:
        Score between 0.0 and 1.0
    """
    pass
```

### Pre-Commit Checklist

Before committing, verify:

- [ ] Python lines ≤ 100 characters
- [ ] Imports organized and sorted
- [ ] No unused imports
- [ ] Code formatted with Black
- [ ] Type hints present
- [ ] Docstrings present
- [ ] Tests written and passing
- [ ] No linting errors

---

## Pull Request Process

### 1. Update Your Branch

```bash
# Fetch latest changes
git fetch upstream

# Rebase on main
git rebase upstream/main

# Resolve conflicts if any
```

### 2. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 3. Create Pull Request

1. Go to GitHub repository
2. Click "New Pull Request"
3. Select your feature branch
4. Fill out the PR template (auto-populated)
5. Ensure all checkboxes are completed

### 4. PR Title Format

```
<type>: <description>
```

Examples:
- `feat: Add temporal weighting to alignment scores`
- `fix: Resolve NaN values in mindfulness calculation`
- `docs: Update API reference for scoring module`

### 5. PR Description

Use the provided template. Include:

- **Summary**: What does this PR do?
- **Motivation**: Why is this change needed?
- **Changes**: List of key changes
- **Testing**: How was this tested?
- **Screenshots**: If UI changes (N/A for this project)
- **Checklist**: Complete all items

---

## Review Process

### What Reviewers Check

1. **Coding Standards Compliance**
   - 100-char line limit for Python files
   - Import organization
   - Black formatting
   - Type hints
   - Docstrings

2. **Code Quality**
   - Logic correctness
   - Edge case handling
   - Error handling
   - Performance considerations

3. **Tests**
   - Adequate coverage
   - Test quality
   - Edge cases tested

4. **Documentation**
   - Clear docstrings
   - Updated README if needed
   - Comments for complex logic

### Responding to Feedback

- Address all comments
- Ask questions if unclear
- Push new commits to address feedback
- Mark conversations as resolved when done
- Be respectful and professional

### Approval and Merge

- **Required approvals**: 1 maintainer
- **Required checks**: All CI checks must pass
- **Merge method**: Squash and merge (typically)

---

## Testing Guidelines

### Test Structure

```python
"""Test module for scoring functions."""

import unittest

import numpy as np
import pytest

from src.scoring import calc_mindful_score


class TestMindfulScore(unittest.TestCase):
    """Tests for mindfulness score calculation."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.test_data = np.random.rand(10, 5)

    def test_score_valid_range(self) -> None:
        """Test score is in valid range [0, 1]."""
        weights = {"f1": 0.5, "f2": 0.5}
        score = calc_mindful_score(self.test_data, weights)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_score_with_zero_weights(self) -> None:
        """Test score with zero weights raises error."""
        weights = {"f1": 0.0, "f2": 0.0}
        with self.assertRaises(ValueError):
            calc_mindful_score(self.test_data, weights)
```

### Test Coverage

- Aim for **>80% coverage** on new code
- Test edge cases
- Test error conditions
- Test with various input types

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_scoring.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run with verbose output
pytest -v
```

---

## Documentation

### Code Documentation

**Required:**
- Module docstrings
- Class docstrings
- Function docstrings (Google style)
- Complex logic comments

**Example:**

```python
"""Module for mindfulness alignment scoring.

This module provides functions to calculate alignment scores
based on various mindfulness metrics with temporal weighting.
"""

from typing import Dict

import numpy as np


def calc_mindful_score(
    data: np.ndarray,
    weights: Dict[str, float],
) -> float:
    """Calculate weighted mindfulness alignment score.

    Computes a score by applying weights to extracted features
    from mindfulness metrics data.

    Args:
        data: Input array of shape (n_samples, n_features)
        weights: Feature weights mapping names to floats

    Returns:
        Alignment score between 0.0 and 1.0

    Raises:
        ValueError: If weights don't sum to 1.0
        ValueError: If data shape is invalid

    Example:
        >>> data = np.random.rand(100, 5)
        >>> weights = {"attention": 0.6, "awareness": 0.4}
        >>> score = calc_mindful_score(data, weights)
        >>> print(f"Score: {score:.2f}")
        Score: 0.73
    """
    if abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError("Weights must sum to 1.0")
    # Implementation...
    return 0.0
```

### Updating Documentation

If your PR changes public APIs:

1. Update relevant docstrings
2. Update README.md if needed
3. Update API documentation
4. Add examples if appropriate

---

## Additional Resources

### Related Documents

- [CODING_STANDARDS.md](./CODING_STANDARDS.md) - Complete coding standards
- [README.md](./README.md) - Project overview
- [docs/style-guide.md](./docs/style-guide.md) - Extended style guide

### External Resources

- [Black Documentation](https://black.readthedocs.io/)
- [PEP 8 Style Guide](https://peps.python.org/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/)

---

## Getting Help

### Questions?

- Open an issue with the `question` label
- Check existing issues for answers
- Review documentation thoroughly first

### Found a Bug?

- Open an issue with the `bug` label
- Include reproduction steps
- Provide error messages and stack traces
- Specify your environment (OS, Python version, etc.)

### Feature Requests?

- Open an issue with the `enhancement` label
- Describe the feature and use case
- Explain why it's valuable
- Consider implementing it yourself!

---

## Recognition

Contributors are recognized in:
- CONTRIBUTORS.md (coming soon)
- Release notes
- Project README

---

Thank you for contributing to GEPA Mindfulness Superalignment!

Your contributions help advance the field of AI alignment and
mindfulness research.
