#include "test_framework.hpp"
#include "parser_test.hpp"
#include "core_test.hpp"
#include <iostream>
#include <memory>

// Only keep main() and necessary includes. Remove TestRunner method implementations.

int main() {
    test_framework::TestRunner runner;
    
    // Add parser tests
    runner.add_test(std::make_unique<test_framework::ParserTest>());
    
    // Add core tests
    runner.add_test(std::make_unique<test_framework::CoreTest>());
    
    bool all_passed = runner.run_all();
    return all_passed ? 0 : 1;
} 