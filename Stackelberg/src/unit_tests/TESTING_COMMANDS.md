# POSG Parser Testing - Quick Reference

## Essential Commands

### Build and Run
```bash
# Complete test cycle
make clean && make && ./run_tests

# Just run existing tests
./run_tests

# Just build
make

# Clean build artifacts
make clean
```

### Directory Navigation
```bash
# Navigate to test directory
cd Stackelberg/src/unit_tests

# Check current directory
pwd
```

## Test Output Examples

### Successful Run
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

🎉 ALL TESTS PASSED! 🎉
```

### Failed Test
```
  Testing Tiger problem parsing...
  ❌ FAILED: Tiger problem should be valid
  POSG Parser Tests: FAILED (29/30)
```

## Debug Commands

### Check File Existence
```bash
# Verify problem files exist
ls -la ../../problem_examples/*.stackelberg

# Check test executable
ls -la run_tests
```

### Verbose Build
```bash
# See detailed compilation
make V=1

# Check compiler version
g++ --version
```

## Common Issues & Solutions

### Build Errors
```bash
# Fix include errors
make clean && make

# Check compiler compatibility
g++ -std=c++17 --version
```

### Test Failures
```bash
# Check problem files
ls ../../problem_examples/

# Verify file permissions
chmod +x run_tests
```

### Segmentation Faults
```bash
# Run with debugger
gdb ./run_tests
run
bt  # if segfault occurs
```

## File Locations

- **Test Directory**: `Stackelberg/src/unit_tests/`
- **Problem Files**: `Stackelberg/problem_examples/`
- **Parser Source**: `Stackelberg/src/parser/`
- **Test Executable**: `Stackelberg/src/unit_tests/run_tests`

## Quick Test Development

### Add New Test
1. Add method to `parser_test.hpp`
2. Implement in `parser_test.cpp`
3. Register in `test_runner.cpp`
4. Run: `make && ./run_tests`

### Test Single File
```bash
# Create minimal test file
echo "agents: 2" > test_file.stackelberg
echo "discount: 1.0" >> test_file.stackelberg
# ... add more content
```

## Integration

### CI/CD Pipeline
```yaml
- name: Test POSG Parser
  run: |
    cd Stackelberg/src/unit_tests
    make clean && make && ./run_tests
```

### Pre-commit Hook
```bash
#!/bin/bash
cd Stackelberg/src/unit_tests
make clean && make && ./run_tests
exit $?
``` 

# Unit Testing: Build & Run Instructions

## Build the Unit Tests

From the `Stackelberg/src/unit_tests` directory, run:

```sh
make
```

This will compile all source files and produce the test runner executable `run_tests`.

## Run the Unit Tests

After building, run:

```sh
./run_tests
```

This will execute all parser and core unit tests and print a summary to the terminal.

## Troubleshooting

- If you see linker errors about multiple definitions, ensure only one implementation of each function exists in the codebase.
- If you see missing header or hash errors, ensure all headers are included and hash specializations are visible.
- To clean the build, run:

```sh
make clean
```

and then rebuild with `make`.

## Notes
- All test output is printed to the terminal.
- Failed assertions and exceptions will be highlighted in red.
- The test runner will print a final summary at the end. 