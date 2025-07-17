#pragma once

#include <string>
#include <iostream>
#include <vector>
#include <memory>

namespace test_framework {

    // ANSI color codes for terminal output
    namespace colors {
        const std::string RESET = "\033[0m";
        const std::string RED = "\033[31m";
        const std::string GREEN = "\033[32m";
        const std::string YELLOW = "\033[33m";
        const std::string BLUE = "\033[34m";
        const std::string MAGENTA = "\033[35m";
        const std::string CYAN = "\033[36m";
        const std::string WHITE = "\033[37m";
        const std::string BOLD = "\033[1m";
    }

    /**
     * @brief Base class for all test classes
     */
    class TestBase {
    protected:
        std::string test_name;
        int passed_tests;
        int total_tests;
        std::vector<std::string> failed_tests;

        // Helper methods for assertions
        bool assert_true(bool condition, const std::string& message = "");
        bool assert_false(bool condition, const std::string& message = "");
        bool assert_equal(const std::string& expected, const std::string& actual, const std::string& message = "");
        bool assert_equal(int expected, int actual, const std::string& message = "");
        bool assert_equal(double expected, double actual, double tolerance = 1e-6, const std::string& message = "");
        bool assert_not_null(const void* ptr, const std::string& message = "");
        bool assert_null(const void* ptr, const std::string& message = "");

    public:
        TestBase(const std::string& name);
        virtual ~TestBase() = default;

        /**
         * @brief Run all tests in this test class
         * @return true if all tests passed, false otherwise
         */
        virtual bool run();

        /**
         * @brief Get test name
         */
        const std::string& get_name() const { return test_name; }

        /**
         * @brief Get test results summary
         */
        void print_summary() const;
    };

    /**
     * @brief Test runner that manages and executes all test classes
     */
    class TestRunner {
    private:
        std::vector<std::unique_ptr<TestBase>> tests;
        int total_passed;
        int total_failed;

    public:
        TestRunner();
        
        /**
         * @brief Add a test to the runner
         */
        void add_test(std::unique_ptr<TestBase> test);
        
        /**
         * @brief Run all tests and report results
         * @return true if all tests passed, false otherwise
         */
        bool run_all();
        
        /**
         * @brief Print final summary
         */
        void print_final_summary() const;
    };

} // namespace test_framework 