# Codex Prompt: GEPA-Mindfulness-superalignment

You are coding for the GEPA-Mindfulness-superalignment repository. Follow these rules strictly.

## BEADS USAGE

- Repository instructions are canonical here; beads reads them via `beads/README.md`.
- See `beads/README.md` for beads setup and ingestion notes.
- Beads is already enabled for this repository; run `bd onboard` from the repo root to fetch
  integration instructions. If `bd` is unavailable, ensure the beads CLI is on your PATH and
  ask the maintainer of the manual install for the expected location.

### Beads workflow goals

- Keep AGENTS.md as the single source of repository standards so beads can track them.
- Use beads to verify changes stay aligned with these rules before merging.

### Beads day-to-day workflow

1. Run `bd onboard` once to pull integration instructions for your environment.
2. Before committing, run `bd status` to confirm the repo is linked and rules are syncing.
3. After updates to AGENTS.md, rerun `bd onboard` if prompted to refresh tracked guidance.
4. If beads output indicates missing configuration, consult `beads/README.md` for
   troubleshooting or contact the maintainer.

## Issue Tracking with bd (beads)

**IMPORTANT**: This project uses **bd (beads)** for ALL issue tracking. Do NOT use markdown
TODOs, task lists, or other tracking methods.

### Why bd?

- Dependency-aware: Track blockers and relationships between issues
- Git-friendly: Auto-syncs to JSONL for version control
- Agent-optimized: JSON output, ready work detection, discovered-from links
- Prevents duplicate tracking systems and confusion

### Quick Start

**Check for ready work:**
```bash
bd ready --json
```

**Create new issues:**
```bash
bd create "Issue title" -t bug|feature|task -p 0-4 --json
bd create "Issue title" -p 1 --deps discovered-from:bd-123 --json
bd create "Subtask" --parent <epic-id> --json  # Hierarchical subtask (gets ID like epic-id.1)
```

**Claim and update:**
```bash
bd update bd-42 --status in_progress --json
bd update bd-42 --priority 1 --json
```

**Complete work:**
```bash
bd close bd-42 --reason "Completed" --json
```

### Issue Types

- `bug` - Something broken
- `feature` - New functionality
- `task` - Work item (tests, docs, refactoring)
- `epic` - Large feature with subtasks
- `chore` - Maintenance (dependencies, tooling)

### Priorities

- `0` - Critical (security, data loss, broken builds)
- `1` - High (major features, important bugs)
- `2` - Medium (default, nice-to-have)
- `3` - Low (polish, optimization)
- `4` - Backlog (future ideas)

### Workflow for AI Agents

1. **Check ready work**: `bd ready` shows unblocked issues.
2. **Claim your task**: `bd update <id> --status in_progress`.
3. **Work on it**: Implement, test, document.
4. **Discover new work?** Create linked issue:
   - `bd create "Found bug" -p 1 --deps discovered-from:<parent-id>`.
5. **Complete**: `bd close <id> --reason "Done"`.
6. **Commit together**: Always commit the `.beads/issues.jsonl` file with the code changes so
   issue state stays in sync with code state.

### Auto-Sync

bd automatically syncs with git:
- Exports to `.beads/issues.jsonl` after changes (5s debounce).
- Imports from JSONL when newer (e.g., after `git pull`).
- No manual export/import needed!

### GitHub Copilot Integration

If using GitHub Copilot, also create `.github/copilot-instructions.md` for automatic instruction
loading. Run `bd onboard` to get the content, or see step 2 of the onboard instructions.

### MCP Server (Recommended)

If using Claude or MCP-compatible clients, install the beads MCP server:

```bash
pip install beads-mcp
```

Add to MCP config (e.g., `~/.config/claude/config.json`):
```json
{
  "beads": {
    "command": "beads-mcp",
    "args": []
  }
}
```

Then use `mcp__beads__*` functions instead of CLI commands.

### Managing AI-Generated Planning Documents

AI assistants often create planning and design documents during development:
- PLAN.md, IMPLEMENTATION.md, ARCHITECTURE.md
- DESIGN.md, CODEBASE_SUMMARY.md, INTEGRATION_PLAN.md
- TESTING_GUIDE.md, TECHNICAL_DESIGN.md, and similar files

#### Best Practice: Use a dedicated directory for these ephemeral files

#### Recommended approach
- Create a `history/` directory in the project root.
- Store ALL AI-generated planning/design docs in `history/`.
- Keep the repository root clean and focused on permanent project files.
- Only access `history/` when explicitly asked to review past planning.

#### Example .gitignore entry (optional)
```gitignore
# AI planning documents (ephemeral)
history/
```

#### Benefits
- ✅ Clean repository root
- ✅ Clear separation between ephemeral and permanent documentation
- ✅ Easy to exclude from version control if desired
- ✅ Preserves planning history for archeological research
- ✅ Reduces noise when browsing the project

### CLI Help

Run `bd <command> --help` to see all available flags for any command. For example:
`bd create --help` shows `--parent`, `--deps`, `--assignee`, and more.

### Important Rules

- ✅ Use bd for ALL task tracking
- ✅ Always use `--json` flag for programmatic use
- ✅ Link discovered work with `discovered-from` dependencies
- ✅ Check `bd ready` before asking "what should I work on?"
- ✅ Store AI planning docs in `history/` directory
- ✅ Run `bd <cmd> --help` to discover available flags
- ❌ Do NOT create markdown TODO lists
- ❌ Do NOT use external issue trackers
- ❌ Do NOT duplicate tracking systems
- ❌ Do NOT clutter repo root with planning documents

For more details, see README.md and QUICKSTART.md.

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
