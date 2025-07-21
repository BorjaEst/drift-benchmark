# Copilot Instructions - TDD Implementation

## Role

Python expert implementing **minimal code to make tests pass**. Follow TDD strictly:

- **Green first**: Write only enough code to pass failing test
- **YAGNI**: Don't implement features not required by tests
- **Question complexity**: Alert if implementation feels overly complex

## Test Modification Policy

**Avoid modifying tests**. If test seems incorrect:

1. **Stop** - Read `README.md` and `REQUIREMENTS.md`
2. **Justify** - Explain why modification needed, reference requirements
3. **Modify** - Update test to align with documentation

## Required Stack (Foundation Only)

**These are BASELINE requirements - add more dependencies as your project needs them:**

- **pydantic v2**: All data validation (never manual validation)
- **rich**: All console output (never print())
- **rich-click**: CLI applications
- **pytest**: Testing

**Add additional dependencies freely:** FastAPI, SQLAlchemy, httpx, pandas, numpy, etc. - whatever your tests and requirements need.

## Core Patterns (ADAPT TO YOUR DOMAIN)

⚠️ **CRITICAL: These are TEMPLATES - Replace with YOUR actual domain models**

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

## Project Structure (Evolve Based on Tests)

**⚠️ DON'T create structure upfront - let tests drive what you need:**

**Start simple, evolve when needed:**

```
project/
├── src/
│   ├── __main__.py    # When you have an entry point
│   ├── [module].py    # When you have a domain model
│   └── [service].py   # When you have a service
├── tests/
└── pyproject.toml
```

**Create subpackages only when you have 3+ related files.**

**As tests require more structure:**

```
project_example/
├── src/
│   ├── models_example/         # Only when you have 3+ models
│   │   ├── configuration.py
│   │   └── metadata.py
│   ├── services_example/       # Only when you have 3+ services
│   │   ├── user_service.py
│   │   └── order_service.py
│   └── repositories/           # Only when tests need data abstraction
├── tests/
└── pyproject.toml
```

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

1. **Run tests** → see failures
2. **Write minimal code** → make test pass
3. **Verify green** → all tests pass
4. **Refactor** → improve while staying green
5. **Repeat** → next failing test

## Complexity Alerts

Warn if implementation needs:

- More than 3 abstraction levels for simple features
- Multiple design patterns for one test
- Significantly more complexity than test suggests

## Testing Patterns

```python
@pytest.fixture
def sample_data():
    return {"name": "Test Item", "value": 42}

def test_valid_creation(sample_data):
    item = DataModel(**sample_data)  # Adapt to YOUR domain
    assert item.name == "Test Item"

def test_validation_error():
    with pytest.raises(ValidationError):
        DataModel(name="", value=2000)
```

## Remember

- **Tests are sacred** - only modify with strong justification
- **Examples are templates** - replace `DataModel` with YOUR domain models
- **Structure evolves** - don't pre-create folders
- **Stack is baseline** - add more dependencies as needed
- **Adapt everything** - patterns, not prescriptions
