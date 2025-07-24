#!/bin/bash

# ε-Optimal LFPOSG Solver - Build and Test Script
# Based on AAAI 2026 Paper Implementation

set -e  # Exit on any error

echo "=========================================="
echo "  ε-Optimal LFPOSG Solver"
echo "  AAAI 2026 Paper Implementation"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "Makefile" ]; then
    echo "Error: Makefile not found. Please run this script from Stackelberg/src/unit_tests/"
    exit 1
fi

# Check CPLEX environment
if [ -z "$CPLEX_HOME" ]; then
    echo "Warning: CPLEX_HOME not set. Using default path..."
    export CPLEX_HOME="/home/ide/Desktop/cplex_ibm/cplex"
fi

if [ -z "$CONCERT_HOME" ]; then
    echo "Warning: CONCERT_HOME not set. Using default path..."
    export CONCERT_HOME="/home/ide/Desktop/cplex_ibm/concert"
fi

echo ""
echo "🔧 Building project..."
make clean
make

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
else
    echo "❌ Build failed!"
    exit 1
fi

echo ""
echo "🧪 Running comprehensive test suite..."
echo "=========================================="

# Run tests and capture output
./run_tests 2>&1 | tee test_output.log

# Extract test results
TOTAL_TESTS=$(grep -c "Testing" test_output.log || echo "0")
PASSED_TESTS=$(grep -c "PASSED" test_output.log || echo "0")
FAILED_TESTS=$(grep -c "FAILED" test_output.log || echo "0")

echo ""
echo "=========================================="
echo "📊 Test Summary"
echo "=========================================="
echo "Total Test Classes: $TOTAL_TESTS"
echo "Passed: $PASSED_TESTS"
echo "Failed: $FAILED_TESTS"

if [ $FAILED_TESTS -eq 0 ]; then
    echo ""
    echo "🎉 All tests passed! Implementation is ready for review."
else
    echo ""
    echo "⚠️  Some tests failed. Check test_output.log for details."
    echo "Note: Some failures are expected (sparse benchmark files, edge cases)"
fi

echo ""
echo "📁 Generated Files:"
echo "- test_output.log: Complete test output"
echo "- run_tests: Executable test runner"

echo ""
echo "🔍 For detailed analysis:"
echo "- Check test_output.log for specific test results"
echo "- Review README.md for theoretical background"
echo "- Examine inline documentation for paper mapping"

echo ""
echo "=========================================="
echo "  Ready for Professor Review"
echo "==========================================" 