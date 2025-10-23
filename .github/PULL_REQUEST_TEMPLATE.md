## Pull Request

### Description

<!-- Provide a clear and concise description of your changes -->

#### Summary
<!-- What does this PR do? -->


#### Motivation
<!-- Why is this change needed? What problem does it solve? -->


#### Related Issues
<!-- Link any related issues using #issue_number -->

Closes #

---

### Type of Change

<!-- Mark the relevant option with an 'x' -->

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Refactoring (code restructuring without changing behavior)
- [ ] Documentation update
- [ ] Test additions or improvements
- [ ] Performance improvement
- [ ] Code style/formatting

---

### Changes Made

<!-- List the key changes in this PR -->

-
-
-

---

### Testing

#### Test Coverage

<!-- Describe how you tested your changes -->

- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests passing locally
- [ ] Coverage maintained or improved

#### Test Commands Run

```bash
# Add the commands you ran to test
pytest
pytest --cov=src --cov-report=term-missing
```

#### Test Results

<!-- Paste relevant test output or describe results -->

```
# Paste test output here
```

---

## Coding Standards Checklist

**IMPORTANT: All items must be checked before PR can be merged**

### Line Length & Formatting

- [ ] All lines are ≤ 100 characters (ABSOLUTE REQUIREMENT)
- [ ] Code formatted with `black --line-length 100 .`
- [ ] No line exceeds limit (verified manually if needed)
- [ ] Long names shortened to fit within limit
- [ ] Complex expressions broken into multiple lines

### Import Organization

- [ ] Imports organized in 3 sections: stdlib, 3rd-party, local
- [ ] Each section alphabetically sorted
- [ ] Sections separated by blank lines
- [ ] `import X` statements before `from X import Y`
- [ ] No wildcard imports (`from module import *`)
- [ ] All imports are used (no unused imports)

### Code Style

- [ ] Double quotes used for strings
- [ ] 4 spaces for indentation (no tabs)
- [ ] No trailing whitespace
- [ ] Trailing commas in multi-line structures
- [ ] Space after comma in lists/tuples/dicts
- [ ] No space before colon in dicts
- [ ] Imports sorted with `isort --profile black --line-length 100 .`

### Type Hints

- [ ] Type hints on all public function parameters
- [ ] Type hints on all public function return values
- [ ] Type hints on class attributes
- [ ] Complex types properly annotated
- [ ] `from typing import ...` imports added

### Documentation

- [ ] Docstrings on all public modules
- [ ] Docstrings on all public classes
- [ ] Docstrings on all public functions
- [ ] Docstrings on all public methods
- [ ] Google-style docstrings used
- [ ] Args, Returns, Raises sections included
- [ ] Examples provided where appropriate
- [ ] Complex logic has inline comments

### Code Quality

- [ ] No debug print statements
- [ ] No commented-out code (unless explained)
- [ ] No `pdb`, `ipdb`, or other debug imports
- [ ] No hardcoded file paths (use config)
- [ ] No mutable default arguments
- [ ] Error handling implemented
- [ ] Edge cases considered

### Testing

- [ ] Tests written for new functionality
- [ ] Tests updated for changed functionality
- [ ] All tests passing (`pytest`)
- [ ] Coverage ≥80% on new code
- [ ] Edge cases tested
- [ ] Error conditions tested
- [ ] Test docstrings present

### Linting & Type Checking

- [ ] `flake8 --max-line-length=100 src/` passes
- [ ] `mypy src/` passes (or issues documented)
- [ ] No linting errors
- [ ] No type checking errors (or explained)

### Git Hygiene

- [ ] Commit messages are clear and descriptive
- [ ] Commits are logically organized
- [ ] No merge commits (rebased on main)
- [ ] Branch is up to date with main
- [ ] No unrelated changes included

---

### Verification Commands

<!-- Confirm you ran these commands successfully -->

```bash
# Formatting
black --line-length 100 . && echo "✓ Black passed"
isort --profile black --line-length 100 . && echo "✓ isort passed"

# Linting
flake8 --max-line-length=100 src/ && echo "✓ Flake8 passed"

# Type checking
mypy src/ && echo "✓ mypy passed"

# Tests
pytest && echo "✓ Tests passed"
pytest --cov=src --cov-report=term-missing && echo "✓ Coverage checked"
```

**I confirm I ran all commands above:**
- [ ] Yes, all commands passed
- [ ] Some commands had issues (explained below)

**Issues (if any):**
<!-- Explain any linting/type checking issues here -->


---

### Screenshots

<!-- If applicable, add screenshots to demonstrate changes -->

N/A for this project (code-only changes)

---

### Additional Notes

<!-- Any additional information reviewers should know -->


---

### Documentation Updates

<!-- Check all that apply -->

- [ ] README.md updated (if needed)
- [ ] CHANGELOG.md updated (if exists)
- [ ] API documentation updated (if needed)
- [ ] Docstrings reflect changes
- [ ] No documentation changes needed

---

### Breaking Changes

<!-- If this PR includes breaking changes, describe them -->

- [ ] No breaking changes
- [ ] Breaking changes documented below

**Breaking changes:**
<!-- List breaking changes and migration path -->


---

### Reviewer Notes

<!-- Any specific areas you'd like reviewers to focus on? -->


---

## For Reviewers

### Review Checklist

- [ ] Code follows [CODING_STANDARDS.md](../CODING_STANDARDS.md)
- [ ] All lines ≤ 100 characters
- [ ] Imports properly organized
- [ ] Type hints present and correct
- [ ] Docstrings present and complete
- [ ] Tests adequate and passing
- [ ] Logic is correct and efficient
- [ ] Edge cases handled
- [ ] Error handling appropriate
- [ ] No security concerns
- [ ] Documentation updated

### Review Comments

<!-- Reviewers: Add your comments here -->


---

## Merge Checklist

**Before merging:**

- [ ] All checklist items completed
- [ ] At least 1 maintainer approval
- [ ] All CI checks passing
- [ ] No merge conflicts
- [ ] Branch up to date with main
- [ ] Breaking changes documented (if any)

---

## Post-Merge

- [ ] Delete feature branch
- [ ] Update related issues
- [ ] Monitor CI/CD pipelines
- [ ] Update CHANGELOG (if applicable)

---

**Thank you for contributing to GEPA Mindfulness Superalignment!**

<!--
Read before submitting:
- CODING_STANDARDS.md: /CODING_STANDARDS.md
- CONTRIBUTING.md: /CONTRIBUTING.md
-->
