# GEPA Mindfulness Superalignment - Coding Standards

**Version:** 1.0.0
**Last Updated:** 2025-10-23
**Status:** Active

## Table of Contents

- [Overview](#overview)
- [Core Principles](#core-principles)
- [Line Length Requirements](#line-length-requirements)
- [Import Organization](#import-organization)
- [Code Formatting](#code-formatting)
- [Naming Conventions](#naming-conventions)
- [Type Hints and Documentation](#type-hints-and-documentation)
- [Testing Standards](#testing-standards)
- [Pre-Commit Checklist](#pre-commit-checklist)
- [Tools and Automation](#tools-and-automation)

---

## Overview

This document defines the mandatory coding standards for the GEPA
Mindfulness Superalignment project. All code contributions must
adhere to these standards without exception.

**Key Requirements:**
- Maximum 100 characters per line (absolute)
- Strict import organization
- No unused imports
- Black formatting compliance
- Comprehensive type hints

---

## Core Principles

### 1. Clarity Over Cleverness

Write code that is immediately understandable. Prefer explicit
over implicit. Document complex logic.

### 2. Consistency

Follow these standards uniformly across all files. Consistency
enables faster code review and maintenance.

### 3. Maintainability

Optimize for long-term maintenance. Future developers should
understand your code without deep context.

---

## Line Length Requirements

### Rule: Maximum 100 Characters Per Line

**This is an absolute requirement with NO exceptions.**

### Strategies for Compliance

#### 1. Rename Long Identifiers

```python
# ❌ BAD: Exceeds 100 characters
def calculate_mindfulness_alignment_score_temporal_weighted(
    data, weights
):
    pass

# ✅ GOOD: Shortened names
def calc_mindful_align_score_temporal(data, weights):
    pass
```

#### 2. Use Implicit Line Continuation

```python
# ❌ BAD: Single long line
result = processor.process(arg1, arg2, arg3, arg4, arg5, arg6)

# ✅ GOOD: Implicit continuation in parentheses
result = processor.process(
    arg1, arg2, arg3, arg4, arg5, arg6
)
```

#### 3. Break Complex Expressions

```python
# ❌ BAD: Chained method calls
result = obj.method1().method2().method3().method4()

# ✅ GOOD: Split into steps
temp = obj.method1().method2()
result = temp.method3().method4()
```

#### 4. Split Long Strings

```python
# ❌ BAD: Long string literal
msg = "This is a very long error message that exceeds limit"

# ✅ GOOD: Implicit string concatenation
msg = (
    "This is a very long error message "
    "that exceeds limit"
)
```

### Common Name Abbreviations

| Full Name | Abbreviated |
|-----------|-------------|
| calculate | calc |
| initialize | init |
| configuration | config |
| parameter | param |
| temporary | temp |
| manager | mgr |
| controller | ctrl |
| repository | repo |
| process | proc |
| execute | exec |
| generate | gen |
| validate | val |

---

## Import Organization

### Structure

All imports must follow this exact three-section structure:

```python
# Section 1: Standard library imports (alphabetically sorted)
import json
import os
from typing import Dict, List, Optional

# Section 2: Third-party imports (alphabetically sorted)
import numpy as np
import torch
from transformers import AutoModel

# Section 3: Local application imports (alphabetically sorted)
from .config import settings
from .utils import helper_func
```

### Rules

1. **Separate sections** with one blank line
2. **Alphabetize** within each section
3. **Group** `import X` before `from X import Y`
4. **NO wildcard imports** (`from module import *`)
5. **Remove all unused imports**

### Import Ordering Within Sections

```python
# ✅ GOOD: Properly ordered
import json
import os
from pathlib import Path
from typing import Dict, List

# ❌ BAD: Wrong order
from typing import Dict, List
import os
from pathlib import Path
import json
```

### Unused Imports

```python
# ❌ BAD: Unused imports
import json
import os
import sys  # Not used anywhere

def process_data(data):
    return json.loads(data)

# ✅ GOOD: Only used imports
import json

def process_data(data):
    return json.loads(data)
```

---

## Code Formatting

### Black Compliance

All code must be formatted with Black using these settings:

```bash
black --line-length 100 <file>
```

### Black Formatting Rules

#### 1. Double Quotes

```python
# ✅ GOOD: Double quotes
name = "example"

# ❌ BAD: Single quotes
name = 'example'
```

#### 2. Indentation

- Use **4 spaces** for indentation
- NO tabs
- NO mixing spaces and tabs

#### 3. Trailing Commas

```python
# ✅ GOOD: Trailing comma in multi-line
items = [
    "item1",
    "item2",
    "item3",
]

# ✅ ACCEPTABLE: Single line without trailing comma
items = ["item1", "item2", "item3"]
```

#### 4. Space After Comma

```python
# ✅ GOOD
func(a, b, c)

# ❌ BAD
func(a,b,c)
```

#### 5. No Space Before Colon

```python
# ✅ GOOD
data = {"key": "value"}

# ❌ BAD
data = {"key" : "value"}
```

### Conflict Resolution: Black vs Line Length

When Black's output exceeds 100 characters:

**Priority Order:**
1. Rename to shorter names
2. Use Black-compliant line continuation
3. Refactor into smaller functions
4. Add intermediate variables

```python
# Black produces >100 chars
very_long_function_name_that_exceeds(arg1, arg2, arg3)

# Strategy 1: Rename
vlong_func_name(arg1, arg2, arg3)

# Strategy 2: Break lines
very_long_function_name_that_exceeds(
    arg1,
    arg2,
    arg3,
)

# Strategy 3: Intermediate variable
args = (arg1, arg2, arg3)
result = very_long_function_name(*args)
```

---

## Naming Conventions

### General Rules

- Use `snake_case` for functions and variables
- Use `PascalCase` for classes
- Use `UPPER_SNAKE_CASE` for constants
- Use descriptive names, but respect 100-char limit

### Examples

```python
# ✅ GOOD
def calc_alignment_score(data: np.ndarray) -> float:
    pass

class MindfulnessModel:
    pass

MAX_ITERATIONS = 1000

# ❌ BAD
def CalcAlignmentScore(data):  # Wrong case
    pass

class mindfulness_model:  # Wrong case
    pass

max_iterations = 1000  # Should be constant
```

### Domain-Specific Naming

For mindfulness and alignment code:

```python
# Clarity is essential
alignment_score  # ✅ Clear
align_sc  # ❌ Too abbreviated

mindful_state  # ✅ Clear
mnd_st  # ❌ Too abbreviated

temporal_weight  # ✅ Clear
tmp_wgt  # ❌ Confusing
```

---

## Type Hints and Documentation

### Type Hints

**Required for:**
- All public function parameters
- All public function return values
- Class attributes
- Complex data structures

```python
from typing import Dict, List, Optional

# ✅ GOOD: Complete type hints
def process_data(
    items: List[str],
    config: Dict[str, float],
    threshold: Optional[float] = None,
) -> Dict[str, List[float]]:
    """Process data items with configuration."""
    pass

# ❌ BAD: Missing type hints
def process_data(items, config, threshold=None):
    pass
```

### Docstrings

**Required for:**
- All public modules
- All public classes
- All public functions
- All public methods

**Format:** Google-style docstrings

```python
def calc_mindful_score(
    data: np.ndarray,
    weights: Dict[str, float],
) -> float:
    """Calculate mindfulness alignment score.

    This function computes a weighted score based on multiple
    mindfulness metrics extracted from the input data.

    Args:
        data: Input array of shape (n_samples, n_features)
        weights: Dictionary mapping feature names to weights

    Returns:
        Weighted alignment score between 0.0 and 1.0

    Raises:
        ValueError: If weights don't sum to 1.0

    Example:
        >>> data = np.random.rand(100, 5)
        >>> weights = {"attention": 0.5, "awareness": 0.5}
        >>> score = calc_mindful_score(data, weights)
    """
    pass
```

---

## Testing Standards

### Test File Organization

```python
"""Test module for mindfulness scoring."""

import unittest
from typing import List

import numpy as np
import pytest

from src.scoring import calc_mindful_score


class TestMindfulScore(unittest.TestCase):
    """Tests for mindfulness score calculation."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.data = np.random.rand(10, 5)

    def test_score_range(self) -> None:
        """Test score is in valid range [0, 1]."""
        weights = {"f1": 0.5, "f2": 0.5}
        score = calc_mindful_score(self.data, weights)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
```

### Test Standards

- Test files follow same import rules
- Test functions use `test_` prefix
- Test names are descriptive but ≤100 chars
- Each test has a docstring
- Use fixtures for common setup

---

## Pre-Commit Checklist

Before committing code, verify:

- [ ] All lines ≤ 100 characters
- [ ] Imports organized in 3 sections (stdlib, 3rd-party, local)
- [ ] Imports alphabetically sorted within sections
- [ ] No unused imports
- [ ] Code formatted with Black (100 char line length)
- [ ] Type hints on all public functions
- [ ] Docstrings on all public functions/classes
- [ ] No trailing whitespace
- [ ] No debug print statements or imports (pdb, etc.)
- [ ] Tests pass
- [ ] No linting errors

---

## Tools and Automation

### Required Tools

```bash
# Install formatting tools
pip install black isort flake8 mypy

# Install testing tools
pip install pytest pytest-cov
```

### Format Code

```bash
# Format with Black
black --line-length 100 .

# Sort imports (compatible with Black)
isort --profile black --line-length 100 .

# Check type hints
mypy src/

# Run linter
flake8 --max-line-length=100 src/
```

### Pre-Commit Hook (Recommended)

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.4.2
    hooks:
      - id: black
        args: [--line-length=100]

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: [--profile=black, --line-length=100]

  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=100]
```

Install:
```bash
pip install pre-commit
pre-commit install
```

---

## Error Prevention

### Common Mistakes

#### 1. Line Continuation

```python
# ❌ BAD: Explicit backslash
result = very_long_function_name(arg1, arg2) \
    + another_function(arg3, arg4)

# ✅ GOOD: Implicit continuation
result = (
    very_long_function_name(arg1, arg2)
    + another_function(arg3, arg4)
)
```

#### 2. Mutable Default Arguments

```python
# ❌ BAD: Mutable default
def process(items: List[str] = []) -> List[str]:
    items.append("new")
    return items

# ✅ GOOD: Use None
def process(items: Optional[List[str]] = None) -> List[str]:
    if items is None:
        items = []
    items.append("new")
    return items
```

#### 3. Mixed Tabs and Spaces

```python
# ❌ BAD: Mixed indentation
def func():
    if True:
        return 1  # 4 spaces

# ✅ GOOD: Consistent spaces
def func():
    if True:
        return 1  # 4 spaces everywhere
```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────┐
│ GEPA MINDFULNESS CODING STANDARDS - QUICK REF          │
├─────────────────────────────────────────────────────────┤
│ Line Length:        100 chars max (ABSOLUTE)           │
│ Indentation:        4 spaces (NO tabs)                 │
│ Quotes:             Double quotes ""                    │
│ Import Sections:    stdlib → 3rd-party → local         │
│ Import Order:       Alphabetical within sections       │
│ Formatting:         Black (--line-length 100)          │
│ Type Hints:         Required on public functions       │
│ Docstrings:         Google-style, required             │
│ Test Prefix:        test_                              │
└─────────────────────────────────────────────────────────┘
```

---

## References

- [Black Documentation](https://black.readthedocs.io/)
- [PEP 8 Style Guide](https://peps.python.org/pep-0008/)
- [PEP 257 Docstring Conventions](https://peps.python.org/pep-0257/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Type Hints (PEP 484)](https://peps.python.org/pep-0484/)

---

**For questions or clarifications, please open an issue in the
project repository.**
