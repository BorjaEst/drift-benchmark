---
mode: agent
---

# Generate Comprehensive Test Suite for User Requirements

Generate a comprehensive test suite that validates user-specified functional requirements through complete end-to-end scenarios.

## Context Understanding

**User-Specified Requirements**: The functional requirements in `REQUIREMENTS.md` come from users/stakeholders and define what the application must do from a user perspective.

**Test Layer**: Create tests that validate complete user workflows and business functionality, not internal component behavior.

## Execution Steps

1. **Requirements Analysis**:

   - Read `README.md` for application purpose and user types
   - Analyze `REQUIREMENTS.md` for user-specified functional behaviors
   - Identify complete user workflows and scenarios

2. **Test Implementation**:

   - Create test structure in `tests/` directory organized by feature areas
   - Generate conftest.py files within test subdirectories for feature-specific fixtures
   - Write tests that validate complete user journeys
   - Include realistic user data and end-to-end assertions

3. **Coverage Verification**:
   - Ensure all functional requirements (REQ-XXX) have scenario coverage
   - Validate tests represent real user workflows
   - Confirm tests fail appropriately showing missing functionality

## Required Deliverables

1. **Test Structure** - `tests/` directory organized by user-facing features
2. **Application-Level Fixtures** - supporting complete scenario testing
3. **End-to-End Test Scenarios** - complete user workflow validation
4. **Coverage Report** - mapping user requirements to test scenarios
5. **Requirement Refinement Suggestions** - for better functional testability

## Test Success Criteria

- [ ] Tests validate complete user workflows from start to finish
- [ ] All user-specified functional requirements have scenario coverage
- [ ] Tests use realistic user data and scenarios
- [ ] Assertions focus on user-visible outcomes and business value
- [ ] Test organization reflects user feature boundaries
- [ ] Tests demonstrate expected user experience when implemented

Generate tests that serve as executable specifications for user-facing features, validating that the application meets user-specified functional requirements.
