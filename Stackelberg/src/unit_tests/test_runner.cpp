// Minimal, paper-aligned test runner
#include "test_framework.hpp"
#include "core_theory_tests.hpp"
#include "algorithm_theory_tests.hpp"
#include <iostream>
#include <memory>

int main(int argc, char* argv[]) {
    test_framework::TestRunner runner;
    runner.add_test(std::make_unique<test_framework::CoreTest>());
    runner.add_test(std::make_unique<test_framework::AlgorithmTest>());
    return runner.run_all();
} 