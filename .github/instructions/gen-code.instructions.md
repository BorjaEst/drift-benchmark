# Copilot Instructions - TDD Implementation

## Role

Python expert implementing **minimal code to make tests pass** using strict **Red → Green → Refactor**.

- Start from a failing test (Red)
- Add the _least_ code required to make that test pass (Green)
- Refactor only while tests stay green (Refactor)
- **Never** add generalized abstractions before at least two tests demand them.

## Test Modification Policy

1. Assume tests are correct.
2. If a test contradicts documented behavior (README.md / REQUIREMENTS.md), pause.
3. Provide a short justification referencing exact spec lines.
4. Propose the _smallest_ targeted change to the test.
5. Only add new tests when introducing net-new behavior explicitly requested.

✅ You may add helper fixtures / parametrizations (not behavior changes) without prior justification.

## Required Stack (Foundation Only)

**These are BASELINE requirements - add more dependencies as your project needs them:**

- **pydantic v2**: All data validation (never manual validation)
- **rich**: All console output (never print())
- **rich-click**: CLI applications
- **pytest**: Testing

**Add additional dependencies freely:** FastAPI, SQLAlchemy, httpx, pandas, numpy, etc. - whatever your tests and requirements need.

### Dependency Management (Governance for Added Dependencies)

Add a new dependency ONLY when all apply:

1. Directly required by a current failing test (cannot make test green otherwise).
2. Standard library or existing deps cannot reasonably satisfy need.
3. Smallest suitable lib (avoid pulling large frameworks for narrow use-case).
4. Version pinned with compatible upper bound (e.g. `aiohttp>=3.9,<4.0`).

Process before adding:

- Write justification (in PR or commit message):
  - Failing test name(s)
  - Why existing code insufficient
  - Alternatives considered (and rejected)
  - Chosen version constraint
- Run vulnerability & license check (e.g. `pip install pip-audit && pip-audit`)
- Update lock / requirements files accordingly

Template:

```
Dependency: <name>
Failing test(s): test_x_y
Need: <one line>
Alternatives: stdlib(<reason rejected>), libA(<reason>), libB(<reason>)
Chosen version: >=X.Y,<X+1.0
Risk Mitigation: pinned + audit clean
```

Removal rule: If no test imports or exercises a dependency after refactor → remove it immediately.

## Core Patterns (ADAPT TO YOUR DOMAIN)

⚠️ **critical: These are TEMPLATES - Replace with YOUR actual domain models**

Don't build "DataModel" - build "User", "Product", "Order", "Invoice", etc.

### Data Models - Domain-Specific Pydantic

```python
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, Annotated
from rich.console import Console

console = Console()

# 🔴 DON'T: Copy this literally
class DataModel(BaseModel):
    # This is just an EXAMPLE - replace with YOUR domain

# ✅ DO: Replace with your actual domain
class User(BaseModel):  # Or Product, Order, Invoice, etc.
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    # Replace these fields with YOUR domain fields
    username: Annotated[str, Field(min_length=1, max_length=50)]
    email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')
    status: Literal["active", "inactive", "pending"] = "pending"
    # Add YOUR domain-specific fields here
```

### Services with Dependency Injection (When Tests Need Them)

```python
# Replace "DataService" and "DataRepository" with YOUR domain
class UserService:  # Or ProductService, OrderService, etc.
    def __init__(self, repository: 'UserRepository'):  # YOUR domain repository
        self._repository = repository

    def create_user(self, user: User) -> User:  # YOUR domain method
        result = self._repository.save(user)
        console.print(f"[green]✓[/green] User created: {user.username}")
        return result
```

### Rich Console Output

```python
# Success/Error
console.print("[green]✓[/green] Success message")
console.print("[red]✗[/red] Error message")

# Tables when needed
from rich.table import Table
table = Table(title="Results")
table.add_column("Name")
table.add_row("Item")
console.print(table)
```

### Rich-Click CLI (Replace with Your Domain)

```python
import rich_click as click

@click.command()
@click.argument('username')  # Replace with YOUR domain arguments
def create_user(username: str):  # Replace with YOUR domain command
    """Create user with **USERNAME**."""  # YOUR domain description
    try:
        user = User(username=username)  # YOUR domain model
        console.print(f"[green]✓[/green] User created: {user.username}")
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}")
        raise click.ClickException(str(e))
```

## Project Structure

Grow only when pressure appears:

- 1–2 modules: keep flat
- Introduce packages at ≥3 cohesive modules
- Repositories/services only after multiple call sites or test pain

## Standard Exceptions

```python
class ProcessingError(Exception):
    """Business logic fails"""
    pass
```

## Implementation Rules

### DO

- **Adapt examples to YOUR domain** (User → Product, Order, etc.)
- Use pydantic for all validation
- Use `console.print()` with rich formatting
- Use dependency injection for services
- Let tests determine structure

### DON'T

- Copy examples literally - adapt to requirements
- Write manual validation for standard types
- Use `print()` statements
- Create structure before tests need it

## TDD Workflow

1. RED: Write (or identify) the smallest failing test exposing the next required behavior.
2. GREEN: Implement only the narrowest code to satisfy current failing test set.
   - Prefer a hard‑coded return → then gradual generalization when a second test forces it.
3. REFACTOR: Improve readability, remove duplication, inline or extract as needed—while staying green.
4. REPEAT: Introduce next failing test that pushes design pressure.
5. STOP: When all tests green and no new behavior required.

Guardrails:

- One behavior change per cycle.
- If refactor causes a failure → revert / fix before continuing.
- Avoid "future proofing".

### Quality Gates (Run Every Cycle Before Commit)

Commands (suggested):

1. `pytest -x` (all must pass)
2. `mypy src/` (types stay sound when typing exists)
3. `ruff check src/ tests/` (lint)
4. `ruff format --check src/ tests/` (format)
5. (Optional) `pip-audit` if dependencies changed

Commit types:

- Red: `test: add failing test for <behavior>`
- Green: `feat: implement <behavior>`
- Refactor: `refactor: <description>` (no behavior change)

## Minimal Implementation Heuristics

Use in order:

1. Fake it (return constant / simplest path).
2. Triangulate (add second test to force variable behavior).
3. Generalize (only after duplication / contradiction emerges).
4. Optimize last (only with a performance test or explicit requirement).

If an implementation needs > ~15 lines for a single new assertion to pass → reassess necessity (raise a complexity alert).

### Abstraction Extraction Trigger Rules

Extract helper/function/class ONLY when ALL true:

1. **Duplication**: ≥3 identical (or near-identical) lines across ≥2 tests or production call sites.
2. **Behavioral Pressure**: Two tests assert different outcomes requiring branching.
3. **Cohesion**: Extracted code has a single, nameable responsibility.

Introduce an interface / ABC ONLY when:

1. Two concrete implementations with identical public method signatures exist.
2. Calling code treats them polymorphically (swappable in a test without changing assertions).
3. A third implementation is planned OR duplication > 15 lines between two implementations.

Prohibited premature abstractions:

- Creating strategy/factory patterns for a single concrete case
- Adding repositories/services with only one caller
- Introducing config objects around ≤3 primitive parameters unless duplication appears

Refactor checklist before extraction:

- Can duplication be tolerated until a third instance appears? If yes → delay.
- Can a hard-coded branch satisfy both tests? If yes → postpone generalization.

Document abstraction decision in commit message:

```
abstract: extracted <HelperName>
Reason: duplication across tests test_a, test_b (5 lines) + divergent assertions
Alternatives: inline duplication (rejected: growing to 20 lines soon)
```

## Test Assets & Fixtures

Centralize reusable test data & configuration under:  
`tests/assets/`

Rules:

- Treat files as read-only during tests (no mutation in place).
- Use fixtures to load assets; never hardcode large literals inline.
- Parametrize tests via directory scans when variation is data-driven.
- Keep asset names descriptive: `sample_config_valid.yaml`, `sample_config_missing_field.yaml`, etc.
- Large JSON/CSV/parquet inputs belong in assets; keep code-focused tests minimal.

Example fixture pattern:

```python
# tests/conftest.py
import json, pathlib, pytest

ASSETS = pathlib.Path(__file__).parent / "assets"

@pytest.fixture
def config_valid_path():
    return ASSETS / "sample_config_valid.yaml"

@pytest.fixture(params=(ASSETS / "sample_config_valid.yaml", ASSETS / "sample_config_missing_field.yaml"))
def config_path(request):
    return request.param
```

When adding a new scenario:

1. Add file to `tests/assets/`
2. Reuse existing parametrized fixture OR extend its param list
3. Add/adjust only the asserting test logic

## Fixture Design Guidance

- Prefer narrow fixtures (single purpose)
- Compose fixtures instead of monoliths
- Avoid side effects; if temp mutation required, copy asset to tmp_path first

## Rich Output Policy

- Only produce console output when a test asserts it OR a CLI path requires it.
- No ornamental output—utility first.

## Complexity Alerts

Raise an alert if:

- A single test triggers >2 new classes
- You need mocks for something not yet abstracted
- Circular imports appear
- More than one pattern (factory + strategy etc.) for current scope

Format alert:
`COMPLEXITY ALERT: <one-line reason>. Proposed simplification: <suggestion>.`

## Remember

- Tests drive design; assets drive parametrization
- Strive for the _next_ passing test, not the final architecture
- Delete dead code eagerly during refactor
- Prefer clarity over cleverness
