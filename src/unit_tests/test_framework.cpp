#include "test_framework.hpp"
#include <sstream>
#include <iomanip>
#include <cmath>

namespace test_framework {

    TestBase::TestBase(const std::string& name) 
        : test_name(name), passed_tests(0), total_tests(0) {}

    bool TestBase::run() {
        std::cout << colors::CYAN << colors::BOLD << "\n=== Running " << test_name << " ===" << colors::RESET << std::endl;
        
        try {
            // This will be overridden by derived classes
            return true;
        } catch (const std::exception& e) {
            std::cout << colors::RED << "EXCEPTION in " << test_name << ": " << e.what() << colors::RESET << std::endl;
            return false;
        } catch (...) {
            std::cout << colors::RED << "UNKNOWN EXCEPTION in " << test_name << colors::RESET << std::endl;
            return false;
        }
    }

    void TestBase::print_summary() const {
        std::cout << colors::CYAN << "  " << test_name << ": ";
        if (failed_tests.empty()) {
            std::cout << colors::GREEN << "PASSED (" << passed_tests << "/" << total_tests << ")" << colors::RESET;
        } else {
            std::cout << colors::RED << "FAILED (" << (total_tests - failed_tests.size()) << "/" << total_tests << ")" << colors::RESET;
            std::cout << colors::YELLOW << "\n    Failed tests:" << colors::RESET;
            for (const auto& failed : failed_tests) {
                std::cout << colors::RED << "\n      - " << failed << colors::RESET;
            }
        }
        std::cout << std::endl;
    }

    // Assertion methods
    bool TestBase::assert_true(bool condition, const std::string& message) {
        total_tests++;
        if (condition) {
            passed_tests++;
            return true;
        } else {
            std::string test_name = message.empty() ? "assert_true" : message;
            failed_tests.push_back(test_name);
            std::cout << colors::RED << "    FAILED: " << test_name << colors::RESET << std::endl;
            return false;
        }
    }

    bool TestBase::assert_false(bool condition, const std::string& message) {
        total_tests++;
        if (!condition) {
            passed_tests++;
            return true;
        } else {
            std::string test_name = message.empty() ? "assert_false" : message;
            failed_tests.push_back(test_name);
            std::cout << colors::RED << "    FAILED: " << test_name << colors::RESET << std::endl;
            return false;
        }
    }

    bool TestBase::assert_equal(const std::string& expected, const std::string& actual, const std::string& message) {
        total_tests++;
        if (expected == actual) {
            passed_tests++;
            return true;
        } else {
            std::string test_name = message.empty() ? "assert_equal(string)" : message;
            failed_tests.push_back(test_name);
            std::cout << colors::RED << "    FAILED: " << test_name << colors::RESET << std::endl;
            std::cout << colors::YELLOW << "      Expected: '" << expected << "'" << colors::RESET << std::endl;
            std::cout << colors::YELLOW << "      Actual:   '" << actual << "'" << colors::RESET << std::endl;
            return false;
        }
    }

    bool TestBase::assert_equal(int expected, int actual, const std::string& message) {
        total_tests++;
        if (expected == actual) {
            passed_tests++;
            return true;
        } else {
            std::string test_name = message.empty() ? "assert_equal(int)" : message;
            failed_tests.push_back(test_name);
            std::cout << colors::RED << "    FAILED: " << test_name << colors::RESET << std::endl;
            std::cout << colors::YELLOW << "      Expected: " << expected << colors::RESET << std::endl;
            std::cout << colors::YELLOW << "      Actual:   " << actual << colors::RESET << std::endl;
            return false;
        }
    }

    bool TestBase::assert_equal(double expected, double actual, double tolerance, const std::string& message) {
        total_tests++;
        if (std::abs(expected - actual) <= tolerance) {
            passed_tests++;
            return true;
        } else {
            std::string test_name = message.empty() ? "assert_equal(double)" : message;
            failed_tests.push_back(test_name);
            std::cout << colors::RED << "    FAILED: " << test_name << colors::RESET << std::endl;
            std::cout << colors::YELLOW << "      Expected: " << std::fixed << std::setprecision(6) << expected << colors::RESET << std::endl;
            std::cout << colors::YELLOW << "      Actual:   " << std::fixed << std::setprecision(6) << actual << colors::RESET << std::endl;
            std::cout << colors::YELLOW << "      Tolerance: " << std::fixed << std::setprecision(6) << tolerance << colors::RESET << std::endl;
            return false;
        }
    }

    bool TestBase::assert_not_null(const void* ptr, const std::string& message) {
        total_tests++;
        if (ptr != nullptr) {
            passed_tests++;
            return true;
        } else {
            std::string test_name = message.empty() ? "assert_not_null" : message;
            failed_tests.push_back(test_name);
            std::cout << colors::RED << "    FAILED: " << test_name << " (pointer is null)" << colors::RESET << std::endl;
            return false;
        }
    }

    bool TestBase::assert_null(const void* ptr, const std::string& message) {
        total_tests++;
        if (ptr == nullptr) {
            passed_tests++;
            return true;
        } else {
            std::string test_name = message.empty() ? "assert_null" : message;
            failed_tests.push_back(test_name);
            std::cout << colors::RED << "    FAILED: " << test_name << " (pointer is not null)" << colors::RESET << std::endl;
            return false;
        }
    }

    // TestRunner implementation
    TestRunner::TestRunner() : total_passed(0), total_failed(0) {}

    void TestRunner::add_test(std::unique_ptr<TestBase> test) {
        tests.push_back(std::move(test));
    }

    bool TestRunner::run_all() {
        std::cout << colors::BLUE << colors::BOLD << "\n==========================================" << colors::RESET << std::endl;
        std::cout << colors::BLUE << colors::BOLD << "           UNIT TEST RUNNER" << colors::RESET << std::endl;
        std::cout << colors::BLUE << colors::BOLD << "==========================================" << colors::RESET << std::endl;

        bool all_passed = true;
        
        for (const auto& test : tests) {
            try {
                bool result = test->run();
                test->print_summary();
                
                if (result) {
                    total_passed++;
                } else {
                    total_failed++;
                    all_passed = false;
                }
            } catch (const std::exception& e) {
                std::cout << colors::RED << "EXCEPTION in test runner: " << e.what() << colors::RESET << std::endl;
                total_failed++;
                all_passed = false;
            } catch (...) {
                std::cout << colors::RED << "UNKNOWN EXCEPTION in test runner" << colors::RESET << std::endl;
                total_failed++;
                all_passed = false;
            }
        }

        print_final_summary();
        return all_passed;
    }

    void TestRunner::print_final_summary() const {
        std::cout << colors::BLUE << colors::BOLD << "\n==========================================" << colors::RESET << std::endl;
        std::cout << colors::BLUE << colors::BOLD << "           FINAL SUMMARY" << colors::RESET << std::endl;
        std::cout << colors::BLUE << colors::BOLD << "==========================================" << colors::RESET << std::endl;
        
        std::cout << "Total test classes: " << tests.size() << std::endl;
        std::cout << colors::GREEN << "Passed: " << total_passed << colors::RESET << std::endl;
        std::cout << colors::RED << "Failed: " << total_failed << colors::RESET << std::endl;
        
        if (total_failed == 0) {
            std::cout << colors::GREEN << colors::BOLD << "\n🎉 ALL TESTS PASSED! 🎉" << colors::RESET << std::endl;
        } else {
            std::cout << colors::RED << colors::BOLD << "\n❌ SOME TESTS FAILED! ❌" << colors::RESET << std::endl;
        }
        std::cout << std::endl;
    }

} // namespace test_framework 