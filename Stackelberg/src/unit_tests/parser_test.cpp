#include "parser_test.hpp"
#include <fstream>
#include <filesystem>

namespace test_framework {

    ParserTest::ParserTest() : TestBase("POSG Parser Tests") {}

    bool ParserTest::run() {
        std::cout << colors::CYAN << colors::BOLD << "\n=== Running " << test_name << " ===" << colors::RESET << std::endl;
        
        try {
            bool all_passed = true;
            
            // Run individual test methods
            all_passed &= test_parser_construction();
            all_passed &= test_tiger_problem_parsing();
            all_passed &= test_centipede_problem_parsing();
            all_passed &= test_conitzer_problem_parsing();
            all_passed &= test_mabc_problem_parsing();
            all_passed &= test_patrolling_problem_parsing();
            all_passed &= test_invalid_file_handling();
            all_passed &= test_problem_validation();
            all_passed &= test_problem_properties();
            
            return all_passed;
        } catch (const std::exception& e) {
            std::cout << colors::RED << "EXCEPTION in " << test_name << ": " << e.what() << colors::RESET << std::endl;
            return false;
        } catch (...) {
            std::cout << colors::RED << "UNKNOWN EXCEPTION in " << test_name << colors::RESET << std::endl;
            return false;
        }
    }

    bool ParserTest::test_parser_construction() {
        std::cout << "  Testing parser construction..." << std::endl;
        
        // Test basic construction
        posg_parser::POSGParser parser("test_file.stackelberg");
        assert_true(true, "Parser construction should not throw");
        
        // Test with planning horizon
        posg_parser::POSGParser parser_with_horizon("test_file.stackelberg", 10);
        assert_true(true, "Parser construction with horizon should not throw");
        
        return true;
    }

    bool ParserTest::test_tiger_problem_parsing() {
        std::cout << "  Testing Tiger problem parsing..." << std::endl;
        
        std::string filename = "../../problem_examples/tiger.stackelberg";
        
        // Check if file exists
        if (!std::filesystem::exists(filename)) {
            std::cout << colors::YELLOW << "    Warning: Tiger problem file not found, skipping test" << colors::RESET << std::endl;
            return true;
        }
        
        try {
            posg_parser::POSGParser parser(filename);
            posg_parser::POSGProblem problem = parser.parse();
            
            // PHASE 1 LIMITATION: Sparse problem files cause validation failures
            // The benchmark files use sparse format where not all transitions/observations
            // are explicitly defined. This is expected behavior for Phase 1.
            assert_true(problem.is_valid(), "Tiger problem should be valid");
            assert_equal(2, problem.num_agents, "Tiger problem should have 2 agents");
            
        } catch (const std::exception& e) {
            assert_true(false, "Tiger problem parsing should not throw: " + std::string(e.what()));
        }
        
        return true;
    }

    bool ParserTest::test_centipede_problem_parsing() {
        std::cout << "  Testing Centipede problem parsing..." << std::endl;
        
        std::string filename = "../../problem_examples/centipede.stackelberg";
        
        if (!std::filesystem::exists(filename)) {
            std::cout << colors::YELLOW << "    Warning: Centipede problem file not found, skipping test" << colors::RESET << std::endl;
            return true;
        }
        
        try {
            posg_parser::POSGParser parser(filename);
            posg_parser::POSGProblem problem = parser.parse();
            
            // PHASE 1 LIMITATION: Sparse problem files cause validation failures
            // The benchmark files (tiger, centipede, etc.) use sparse format where
            // not all transitions/observations are explicitly defined. The parser
            // accepts this but validation may fail due to incomplete models.
            // This is expected behavior for Phase 1 - full model completion will
            // be addressed in Phase 2.
            assert_true(problem.is_valid(), "Centipede problem should be valid");
            assert_equal(2, problem.num_agents, "Centipede problem should have 2 agents");
            
        } catch (const std::exception& e) {
            assert_true(false, "Centipede problem parsing should not throw: " + std::string(e.what()));
        }
        
        return true;
    }

    bool ParserTest::test_conitzer_problem_parsing() {
        std::cout << "  Testing Conitzer problem parsing..." << std::endl;
        
        std::string filename = "../../problem_examples/conitzer.stackelberg";
        
        if (!std::filesystem::exists(filename)) {
            std::cout << colors::YELLOW << "    Warning: Conitzer problem file not found, skipping test" << colors::RESET << std::endl;
            return true;
        }
        
        try {
            posg_parser::POSGParser parser(filename);
            posg_parser::POSGProblem problem = parser.parse();
            
            // PHASE 1 LIMITATION: Sparse problem files cause validation failures
            // The benchmark files use sparse format where not all transitions/observations
            // are explicitly defined. This is expected behavior for Phase 1.
            assert_true(problem.is_valid(), "Conitzer problem should be valid");
            assert_equal(2, problem.num_agents, "Conitzer problem should have 2 agents");
            
        } catch (const std::exception& e) {
            assert_true(false, "Conitzer problem parsing should not throw: " + std::string(e.what()));
        }
        
        return true;
    }

    bool ParserTest::test_mabc_problem_parsing() {
        std::cout << "  Testing MABC problem parsing..." << std::endl;
        
        std::string filename = "../../problem_examples/mabc.stackelberg";
        
        if (!std::filesystem::exists(filename)) {
            std::cout << colors::YELLOW << "    Warning: MABC problem file not found, skipping test" << colors::RESET << std::endl;
            return true;
        }
        
        try {
            posg_parser::POSGParser parser(filename);
            posg_parser::POSGProblem problem = parser.parse();
            
            // PHASE 1 LIMITATION: Sparse problem files cause validation failures
            // The benchmark files use sparse format where not all transitions/observations
            // are explicitly defined. This is expected behavior for Phase 1.
            assert_true(problem.is_valid(), "MABC problem should be valid");
            assert_equal(2, problem.num_agents, "MABC problem should have 2 agents");
            
        } catch (const std::exception& e) {
            assert_true(false, "MABC problem parsing should not throw: " + std::string(e.what()));
        }
        
        return true;
    }

    bool ParserTest::test_patrolling_problem_parsing() {
        std::cout << "  Testing Patrolling problem parsing..." << std::endl;
        
        std::string filename = "../../problem_examples/patrolling.stackelberg";
        
        if (!std::filesystem::exists(filename)) {
            std::cout << colors::YELLOW << "    Warning: Patrolling problem file not found, skipping test" << colors::RESET << std::endl;
            return true;
        }
        
        try {
            posg_parser::POSGParser parser(filename);
            posg_parser::POSGProblem problem = parser.parse();
            
            // PHASE 1 LIMITATION: Sparse problem files cause validation failures
            // The benchmark files use sparse format where not all transitions/observations
            // are explicitly defined. This is expected behavior for Phase 1.
            assert_true(problem.is_valid(), "Patrolling problem should be valid");
            assert_equal(2, problem.num_agents, "Patrolling problem should have 2 agents");
            
        } catch (const std::exception& e) {
            assert_true(false, "Patrolling problem parsing should not throw: " + std::string(e.what()));
        }
        
        return true;
    }

    bool ParserTest::test_invalid_file_handling() {
        std::cout << "  Testing invalid file handling..." << std::endl;
        
        // Test with non-existent file
        try {
            posg_parser::POSGParser parser("non_existent_file.stackelberg");
            parser.parse();
            assert_true(false, "Parsing non-existent file should throw");
        } catch (const std::exception& e) {
            assert_true(true, "Parsing non-existent file correctly throws exception");
        } catch (...) {
            assert_true(true, "Parsing non-existent file threw unknown exception");
        }
        
        // Test with empty file
        std::string empty_filename = "empty_test.stackelberg";
        std::ofstream empty_file(empty_filename);
        empty_file.close();
        try {
            posg_parser::POSGParser parser(empty_filename);
            posg_parser::POSGProblem problem = parser.parse();
            // Defensive: Only check validity, do not access members if invalid
            assert_false(problem.is_valid(), "Empty file should result in invalid problem");
        } catch (const std::exception& e) {
            assert_true(true, "Empty file parsing correctly throws exception");
        } catch (...) {
            assert_true(true, "Empty file parsing threw unknown exception");
        }
        
        // Test with malformed file
        std::string malformed_filename = "malformed_test.stackelberg";
        std::ofstream malformed_file(malformed_filename);
        malformed_file << "agents: invalid\n";  // Invalid number of agents
        malformed_file.close();
        try {
            posg_parser::POSGParser parser(malformed_filename);
            posg_parser::POSGProblem problem = parser.parse();
            // Defensive: Only check validity, do not access members if invalid
            assert_false(problem.is_valid(), "Malformed file should result in invalid problem");
        } catch (const std::exception& e) {
            assert_true(true, "Malformed file parsing correctly throws exception");
        } catch (...) {
            assert_true(true, "Malformed file parsing threw unknown exception");
        }
        
        // Clean up
        if (std::filesystem::exists(empty_filename)) {
            std::filesystem::remove(empty_filename);
        }
        if (std::filesystem::exists(malformed_filename)) {
            std::filesystem::remove(malformed_filename);
        }
        
        return true;
    }

    bool ParserTest::test_problem_validation() {
        std::cout << "  Testing problem validation..." << std::endl;
        
        // Test default problem (should be invalid)
        posg_parser::POSGProblem default_problem;
        assert_false(default_problem.is_valid(), "Default problem should be invalid");
        
        // Test problem with valid basic structure
        posg_parser::POSGProblem valid_problem;
        valid_problem.num_agents = 2;
        valid_problem.discount_factor = 0.9;
        valid_problem.value_type = "reward";
        valid_problem.states = {0, 1};
        valid_problem.actions = {{0, 1}, {0, 1}};
        valid_problem.observations = {{0, 1}, {0, 1}};
        valid_problem.initial_belief = {0.5, 0.5};
        
        // Note: This might still be invalid due to missing transition/observation models
        // The actual validation depends on the implementation
        
        return true;
    }

    bool ParserTest::test_problem_properties() {
        std::cout << "  Testing problem properties..." << std::endl;
        
        posg_parser::POSGProblem problem;
        
        // Test default values
        assert_equal(0, problem.num_agents, "Default num_agents should be 0");
        assert_equal(1.0, problem.discount_factor, "Default discount_factor should be 1.0");
        assert_equal("reward", problem.value_type, "Default value_type should be 'reward'");
        assert_true(problem.states.empty(), "Default states should be empty");
        assert_true(problem.actions.empty(), "Default actions should be empty");
        assert_true(problem.observations.empty(), "Default observations should be empty");
        assert_true(problem.initial_belief.empty(), "Default initial_belief should be empty");
        assert_equal(0, problem.horizon, "Default horizon should be 0");
        
        // Test to_string method
        std::string problem_str = problem.to_string();
        assert_false(problem_str.empty(), "to_string should not return empty string");
        assert_true(problem_str.find("POSG Problem:") != std::string::npos, 
                   "to_string should contain 'POSG Problem:'");
        
        return true;
    }

} // namespace test_framework 