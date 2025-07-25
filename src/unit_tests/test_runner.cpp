#include "test_framework.hpp"
#include "parser_test.hpp"
#include <iostream>

int main() {
    test_framework::TestRunner runner;
    
    // Add all test classes to the runner
    runner.add_test(std::make_unique<test_framework::ParserTest>());
    
    // Run all tests
    bool all_passed = runner.run_all();
    
    // Return appropriate exit code
    return all_passed ? 0 : 1;
} 