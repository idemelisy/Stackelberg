# POSG Parser Unit Testing Framework

## Overview

This directory contains a comprehensive unit testing framework for the POSG (Partially Observable Stackelberg Game) parser. The framework is designed to ensure robust parsing of `.stackelberg` problem files with professional-grade error handling and validation.

## Features

- **Custom C++ Testing Framework**: Lightweight, colored output, exception handling
- **Comprehensive Test Coverage**: Tests all parser components and edge cases
- **Robust Error Handling**: Defensive programming prevents segfaults
- **Mock Dependencies**: Isolated testing with mock core classes
- **Professional Output**: Clear test results with pass/fail summaries

## Directory Structure

```
unit_tests/
├── README.md                 # This documentation
├── Makefile                  # Build configuration
├── test_framework.cpp        # Core testing framework
├── test_framework.hpp        # Test framework headers
├── parser_test.cpp           # Parser-specific tests
├── parser_test.hpp           # Parser test headers
├── test_runner.cpp           # Test execution engine
├── mock_core.hpp             # Mock implementations for dependencies
├── run_tests                 # Compiled test executable
└── *.o                       # Object files (generated)
```

## Setup

### Prerequisites

- C++17 compatible compiler (g++ recommended)
- Make build system
- POSIX-compliant system (Linux/macOS)

### Building the Tests

```bash
# Navigate to the unit tests directory
cd Stackelberg/src/unit_tests

# Clean previous builds (optional)
make clean

# Build the test framework and parser
make
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
./run_tests
```

### Complete Build and Test Cycle

```bash
# Clean, build, and run in one command
make clean && make && ./run_tests
```

### Expected Output

```
==========================================
           UNIT TEST RUNNER
==========================================

=== Running POSG Parser Tests ===
  Testing parser construction...
  Testing Tiger problem parsing...
  Testing Centipede problem parsing...
  Testing Conitzer problem parsing...
  Testing MABC problem parsing...
  Testing Patrolling problem parsing...
  Testing invalid file handling...
  Testing problem validation...
  Testing problem properties...
  POSG Parser Tests: PASSED (30/30)

==========================================
           FINAL SUMMARY
==========================================
Total test classes: 1
Passed: 1
Failed: 0

🎉 ALL TESTS PASSED! 🎉
```

## Test Structure

### Test Framework Components

#### TestBase Class
- Base class for all test classes
- Provides assertion methods (`assert_true`, `assert_equal`, etc.)
- Handles test result tracking
- Colored output for pass/fail status

#### ParserTest Class
- Tests POSG parser functionality
- Validates parsing of all problem example files
- Tests error handling for invalid files
- Verifies problem validation logic

### Test Categories

#### 1. Parser Construction Tests
- Verifies parser object creation
- Tests constructor parameter handling

#### 2. Problem File Parsing Tests
- **Tiger Problem**: Basic 2-state, 2-agent game
- **Centipede Problem**: 5-state sequential game
- **Conitzer Problem**: 4-state strategic game
- **MABC Problem**: 4-state coordination game
- **Patrolling Problem**: 6-state security game

#### 3. Error Handling Tests
- Non-existent file handling
- Empty file parsing
- Malformed file validation
- Exception throwing verification

#### 4. Problem Validation Tests
- Default problem validation
- Valid problem structure verification
- Invalid problem detection

#### 5. Problem Properties Tests
- Default value verification
- String representation testing
- Property access validation

## Adding New Tests

### Creating a New Test Method

1. **Add to ParserTest class** in `parser_test.hpp`:
```cpp
class ParserTest : public TestBase {
public:
    // ... existing methods ...
    bool test_new_functionality();
};
```

2. **Implement the test** in `parser_test.cpp`:
```cpp
bool ParserTest::test_new_functionality() {
    std::cout << "  Testing new functionality..." << std::endl;
    
    // Test logic here
    assert_true(condition, "Description of what should be true");
    assert_equal(expected, actual, "Description of equality check");
    
    return true;
}
```

3. **Register the test** in `test_runner.cpp`:
```cpp
bool run_parser_tests() {
    ParserTest parser_test;
    
    // ... existing tests ...
    parser_test.run_test("new_functionality", &ParserTest::test_new_functionality);
    
    return parser_test.all_tests_passed();
}
```

### Testing Best Practices

1. **Defensive Programming**: Always check for null/empty before accessing
2. **Exception Handling**: Test both success and failure cases
3. **Clear Assertions**: Use descriptive assertion messages
4. **Isolation**: Each test should be independent
5. **Coverage**: Test edge cases and error conditions

## Problem File Format

The parser expects `.stackelberg` files with the following format:

```
agents: 2
discount: 1.0
values: reward

states: 0 1 2
actions:
0 1
0 1
observations:
0 1
0 1

start: 1.0 0.0 0.0

# Transition dynamics
T: 0 0 : 0 : 1 : 1.0
T: 1 0 : 0 : 2 : 1.0
# ... more transitions

# Observation model
O: 0 0 : 0 : 0 0 : 1.0
# ... more observations

# Rewards
R: 0 0 : 0 : * : * : 1 0
# ... more rewards
```

## Troubleshooting

### Common Issues

#### Build Errors
```bash
# If you get include errors:
make clean
make
```

#### Test Failures
- Check that problem example files exist in `../../problem_examples/`
- Verify file permissions
- Ensure C++17 compatibility

#### Segmentation Faults
- The framework includes defensive programming to prevent segfaults
- If segfaults occur, check for uninitialized pointers or array access
- Use debug output to trace the issue

### Debug Output

The parser includes debug output to help diagnose issues:
```
[DEBUG] agents: 2
[DEBUG] discount: 1
[DEBUG] value_type: reward
[DEBUG] states: 0 1
[DEBUG] actions[0]: 0 1
[DEBUG] actions[1]: 0 1
[DEBUG] observations[0]: 0 1
[DEBUG] observations[1]: 0 1
[DEBUG] initial_belief: 1 0
```

### Validation Errors

The parser provides detailed validation error messages:
```
DEBUG: num_agents != 2: 0
DEBUG: discount_factor invalid: -1.0
DEBUG: states empty
DEBUG: actions.size() != num_agents: 1 != 2
DEBUG: observations.size() != num_agents: 1 != 2
DEBUG: initial_belief.size() != states.size(): 3 != 2
DEBUG: negative belief probability: -0.5
DEBUG: belief_sum != 1.0: 0.8
DEBUG: transition_model.is_valid() = false
DEBUG: observation_model.is_valid() = false
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: POSG Parser Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Install dependencies
      run: sudo apt-get update && sudo apt-get install -y g++ make
    - name: Run tests
      run: |
        cd Stackelberg/src/unit_tests
        make clean && make && ./run_tests
```

## Contributing

When adding new tests or modifying the parser:

1. **Follow the existing patterns** in the test framework
2. **Add comprehensive error handling** for new functionality
3. **Update this documentation** for any new features
4. **Test edge cases** and error conditions
5. **Ensure all tests pass** before submitting changes

## License

This testing framework is part of the LFPOSG-Stackelberg project. Please refer to the main project license for usage terms. 