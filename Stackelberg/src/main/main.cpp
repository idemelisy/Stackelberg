/// @file main.cpp
/// @brief Command-line interface for the ε-optimal Leader–Follower POSG solver
///
/// Usage:
///   ./run_solver --problem centipede.stackelberg --epsilon 1e-4 --maxIter 500
///
/// The executable parses the given problem file, constructs the initial
/// occupancy state & credible set, runs PBVI with MILP improve phase, and
/// prints the resulting leader policy together with the ε-exploitability bound
/// (Theorem 5.3).  All heavy lifting is delegated to the CMDPSolver class.

#include <iostream>
#include <exception>
#include <string>
#include <unordered_map>
#include <cstdlib>

#include "../parser/include/posg_parser.hpp"
#include "../core/include/occupancy_state.hpp"
#include "../core/include/credible_set.hpp"
#include "../algorithms/include/cmdp_solver.hpp"
#include "../common/logging.hpp"

namespace {
    struct CLIArgs {
        std::string problem_file;
        size_t max_iterations = 500;
        double epsilon = 1e-4;
        double milp_time_limit = 10.0; // Default MILP time limit in seconds
    };

    CLIArgs parse_args(int argc, char* argv[]) {
        CLIArgs args;
        for (int i = 1; i < argc; ++i) {
            std::string token = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) {
                    throw std::runtime_error("Missing value for argument " + token);
                }
                return std::string(argv[++i]);
            };

            if (token == "--problem" || token == "-p") {
                args.problem_file = next();
            } else if (token == "--maxIter" || token == "-n") {
                args.max_iterations = static_cast<size_t>(std::stoul(next()));
            } else if (token == "--epsilon" || token == "-e") {
                args.epsilon = std::stod(next());
            } else if (token == "--milpTimeLimit" || token == "--milp-time-limit") {
                args.milp_time_limit = std::stod(next());
            } else if (token == "--help" || token == "-h") {
                std::cout << "Usage: " << argv[0] << " --problem file.stackelberg [--maxIter N] [--epsilon E] [--milpTimeLimit S]\n";
                std::cout << "  --problem/-p         Path to .stackelberg problem file\n";
                std::cout << "  --maxIter/-n         Max PBVI iterations (default: 500)\n";
                std::cout << "  --epsilon/-e         Target epsilon (default: 1e-4)\n";
                std::cout << "  --milpTimeLimit      MILP time limit in seconds (default: 10)\n";
                std::exit(0);
            } else {
                throw std::runtime_error("Unknown argument: " + token);
            }
        }
        if (args.problem_file.empty()) {
            throw std::runtime_error("--problem argument is required (provide a .stackelberg file)");
        }
        return args;
    }
}

int main(int argc, char* argv[]) {
    try {
        // ------------------------------------------------------------------
        // CLI Parsing
        // ------------------------------------------------------------------
        CLIArgs cli = parse_args(argc, argv);
        LOG_INFO("Problem File      : " << cli.problem_file);
        LOG_INFO("Max PBVI Iterations: " << cli.max_iterations);
        LOG_INFO("Target ε          : " << cli.epsilon);
        LOG_INFO("MILP Time Limit     : " << cli.milp_time_limit << " seconds");

        // ------------------------------------------------------------------
        // Parse POSG problem file
        // ------------------------------------------------------------------
        LOG_INFO("[1] Parsing problem file…");
        posg_parser::POSGParser parser(cli.problem_file, /*horizon=*/2, /*sparse=*/true);
        posg_parser::POSGProblem problem = parser.parse();

        // ------------------------------------------------------------------
        // Construct initial occupancy state & credible set
        // ------------------------------------------------------------------
        LOG_INFO("[2] Constructing initial occupancy / credible set…");
        posg_core::OccupancyState initial_occupancy(problem.initial_belief);
        std::vector<posg_core::OccupancyState> initial_occupancies = {initial_occupancy};

        // ------------------------------------------------------------------
        // Solver setup
        // ------------------------------------------------------------------
        LOG_INFO("[3] Initialising CMDPSolver & running PBVI+MILP…");
        posg_algorithms::CMDPSolver solver(problem, cli.milp_time_limit);
        posg_algorithms::ValueFunction value_function = solver.pbvi_with_milp(
            initial_occupancies, cli.max_iterations, cli.epsilon);

        // ------------------------------------------------------------------
        // Extract policy and exploitability bound
        // ------------------------------------------------------------------
        auto policy_fn = solver.extract_policy(value_function, cli.epsilon);
        double initial_value = value_function.get_value(initial_occupancy);

        LOG_INFO("[4] Results");
        std::cout << "Leader best action at initial belief: "
                  << policy_fn(initial_occupancy).to_string() << std::endl;
        std::cout << "Value at initial belief: " << initial_value << std::endl;
        // Note: ε-bound is printed inside pbvi_with_milp() each iteration.

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }
} 