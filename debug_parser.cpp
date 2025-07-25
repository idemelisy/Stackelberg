#include "Stackelberg/src/parser/include/posg_parser.hpp"
#include <iostream>

int main() {
    try {
        std::cout << "Testing Tiger problem parsing..." << std::endl;
        posg_parser::POSGParser parser("../../problem_examples/tiger.stackelberg");
        posg_parser::POSGProblem problem = parser.parse();
        std::cout << "Tiger problem parsed successfully!" << std::endl;
        std::cout << "Problem has " << problem.states.size() << " states" << std::endl;
        std::cout << "Problem has " << problem.actions[0].size() << " leader actions" << std::endl;
        std::cout << "Problem has " << problem.actions[1].size() << " follower actions" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cout << "ERROR: " << e.what() << std::endl;
        return 1;
    }
} 