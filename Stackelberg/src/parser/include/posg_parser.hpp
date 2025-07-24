#pragma once

#include "../../core/include/transition_model.hpp"
#include "../../core/include/observation_model.hpp"
// #include "../unit_tests/mock_core.hpp" // Removed: use real core headers instead
#include <string>
#include <vector>
#include <unordered_map>

namespace posg_parser {

    /**
     * @brief Represents a complete POSG problem definition
     */
    struct POSGProblem {
        // Problem parameters
        int num_agents;
        double discount_factor;
        std::string value_type;  // "reward" or "cost"
        
        // State and action spaces
        std::vector<int> states;
        std::vector<std::vector<int>> actions;  // actions[agent_id] = list of actions
        std::vector<std::vector<int>> observations;  // observations[agent_id] = list of observations
        
        // Initial belief state
        std::vector<double> initial_belief;
        
        // Models
        posg_core::TransitionModel transition_model;
        posg_core::ObservationModel observation_model;
        
        // Reward functions: reward[state][joint_action] = reward
        std::vector<std::vector<double>> rewards_leader;
        std::vector<std::vector<double>> rewards_follower;
        
        int horizon = 0;
        
        // Constructor
        POSGProblem();
        
        // Validation
        bool is_valid() const;
        std::string to_string() const;
    };

    /**
     * @brief Parser for POSG problem files
     */
    class POSGParser {
    private:
        std::string filename;
        POSGProblem problem;
        int planning_horizon_ = 0;
        bool allow_sparse_ = true;
        
        // Helper methods
        void parse_header(std::ifstream& file);
        void parse_states(std::ifstream& file);
        void parse_actions(std::ifstream& file);
        void parse_observations(std::ifstream& file);
        void parse_initial_belief(std::ifstream& file);
        void parse_transitions(std::ifstream& file);
        void parse_observations_model(std::ifstream& file);
        void parse_rewards(std::ifstream& file);
        
        std::vector<std::string> split_line(const std::string& line);
        std::vector<double> parse_probabilities(const std::vector<std::string>& tokens, int start_idx);

    public:
        POSGParser(const std::string& filename, int planning_horizon, bool allow_sparse = true);
        
        /**
         * @brief Parse the POSG problem file
         * @return The parsed problem
         */
        POSGProblem parse();
        
        /**
         * @brief Get the parsed problem
         */
        const POSGProblem& get_problem() const { return problem; }
        
        /**
         * @brief Validate the parsed problem
         */
        bool validate() const;
        
        /**
         * @brief Get error messages if validation fails
         */
        std::vector<std::string> get_validation_errors() const;
    };

} // namespace posg_parser 