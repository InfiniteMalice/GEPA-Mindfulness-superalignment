# Beads instructions

This file mirrors the repository's AGENTS.md instructions so beads can ingest them directly.
Installation of beads could not complete here because access to the install script was blocked
with a 403 response. When network access allows, run the command below from the repository root
before using beads:

curl -fsSL https://raw.githubusercontent.com/steveyegge/beads/main/scripts/install.sh | bash

The rest of this file contains the instructions copied from AGENTS.md for beads to track.
# Codex Prompt: GEPA-Mindfulness-superalignment

You are coding for the GEPA-Mindfulness-superalignment repository. Follow these rules strictly.

## CRITICAL RULES (ALL CODE MUST FOLLOW)

### Rule 1: 100 Character Line Limit
- Maximum line length: 100 characters
- NO exceptions
- Count includes indentation
- Enforce before Black formatting

### Rule 2: Import Organization
Structure ALL imports exactly like this:

```
# Standard library
import os
import sys
from typing import Any, Dict

# Third-party  
import numpy as np
import torch

# Local
from .module import function
```

- Alphabetize each section
- Separate sections with blank lines
- No `from X import *`
- No unused imports

### Rule 3: No Unused Imports
- Every import must be used
- Remove debug imports before commit
- Check with linter

### Rule 4: Black Formatting
- Use Black's style guide
- 4 spaces indentation
- Double quotes for strings
- Trailing commas in multi-line structures
- Run: `black --line-length 100 .`

## CONFLICT RESOLUTION

When Black wants >100 chars:

**Option A: Rename**
```
# Before
def calculate_mindfulness_superalignment_coefficient(x, y):
    pass

# After  
def calc_mindful_super_coef(x, y):
    pass
```

**Option B: Break Lines**
```
# Before
result = model.forward(input_data, attention_mask, token_type_ids)

# After
result = model.forward(
    input_data,
    attention_mask,
    token_type_ids,
)
```

**Option C: Intermediate Variables**
```
# Before
output = process(transform(normalize(validate(data))))

# After
valid = validate(data)
norm = normalize(valid)
trans = transform(norm)
output = process(trans)
```

## SHORTHAND REFERENCE

Common abbreviations to stay under 100 chars:
- calculate → calc
- initialize → init
- configuration → config
- parameter → param
- manager → mgr
- temporary → temp
- alignment → align
- mindfulness → mindful
- superalignment → super_align

## CODE TEMPLATE

```
"""One-line module description."""

import json
from typing import Dict, List

import numpy as np

from .utils import helper


def short_func_name(
    arg1: str,
    arg2: int = 0,
) -> Dict[str, Any]:
    """Brief description.
    
    Args:
        arg1: Description
        arg2: Description
        
    Returns:
        Description
    """
    result = {"key": arg1, "value": arg2}
    return result
```

## PRE-COMMIT CHECKLIST

- [ ] All lines ≤ 100 characters (verify manually)
- [ ] Imports: sorted, organized, no unused
- [ ] Black formatted: `black --line-length 100 .`
- [ ] No syntax errors
- [ ] Type hints present

## PRIORITY ORDER

1. 100 char limit (ABSOLUTE)
2. Proper imports (REQUIRED)
3. No unused imports (REQUIRED)
4. Black formatting (REQUIRED)

If conflicts arise, rename or refactor until all 4 rules pass.

Generate code now following these rules.
