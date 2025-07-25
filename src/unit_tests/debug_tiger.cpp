#include "../parser/include/posg_parser.hpp"
#include <iostream>

int main() {
    std::cout << "=== Debugging Tiger Parser ===" << std::endl;
    
    try {
        posg_parser::POSGParser parser("../../problem_examples/tiger.stackelberg");
        std::cout << "Parser created successfully" << std::endl;
        
        posg_parser::POSGProblem problem = parser.parse();
        std::cout << "Parser completed successfully" << std::endl;
        
        std::cout << "\n=== Final Problem Summary ===" << std::endl;
        std::cout << "Number of agents: " << problem.num_agents << std::endl;
        std::cout << "Discount factor: " << problem.discount_factor << std::endl;
        std::cout << "Value type: " << problem.value_type << std::endl;
        std::cout << "Number of states: " << problem.states.size() << std::endl;
        std::cout << "Number of actions per agent: ";
        for (size_t i = 0; i < problem.actions.size(); ++i) {
            std::cout << problem.actions[i].size() << " ";
        }
        std::cout << std::endl;
        std::cout << "Number of observations per agent: ";
        for (size_t i = 0; i < problem.observations.size(); ++i) {
            std::cout << problem.observations[i].size() << " ";
        }
        std::cout << std::endl;
        std::cout << "Initial belief size: " << problem.initial_belief.size() << std::endl;
        
        bool is_valid = problem.is_valid();
        std::cout << "Problem is valid: " << (is_valid ? "YES" : "NO") << std::endl;
        
        if (!is_valid) {
            std::cout << "\n=== Validation Errors ===" << std::endl;
            auto errors = parser.get_validation_errors();
            for (const auto& error : errors) {
                std::cout << "ERROR: " << error << std::endl;
            }
        }
        
    } catch (const std::exception& e) {
        std::cout << "EXCEPTION: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cout << "UNKNOWN EXCEPTION" << std::endl;
        return 1;
    }
    
    return 0;
} 