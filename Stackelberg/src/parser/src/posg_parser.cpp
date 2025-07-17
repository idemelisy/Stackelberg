#include "../include/posg_parser.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <cmath>

namespace posg_parser {

    POSGProblem::POSGProblem() : num_agents(0), discount_factor(1.0), value_type("reward") {}

    bool POSGProblem::is_valid() const {
        // Check basic parameters
        if (num_agents != 2) {
            std::cout << "DEBUG: num_agents != 2: " << num_agents << std::endl;
            return false;  // Only support 2-agent games
        }
        if (discount_factor < 0.0 || discount_factor > 1.0) {
            std::cout << "DEBUG: discount_factor invalid: " << discount_factor << std::endl;
            return false;
        }
        if (states.empty()) {
            std::cout << "DEBUG: states empty" << std::endl;
            return false;
        }
        if (actions.size() != num_agents) {
            std::cout << "DEBUG: actions.size() != num_agents: " << actions.size() << " != " << num_agents << std::endl;
            return false;
        }
        if (observations.size() != num_agents) {
            std::cout << "DEBUG: observations.size() != num_agents: " << observations.size() << " != " << num_agents << std::endl;
            return false;
        }
        
        // Check initial belief
        if (initial_belief.size() != states.size()) {
            std::cout << "DEBUG: initial_belief.size() != states.size(): " << initial_belief.size() << " != " << states.size() << std::endl;
            return false;
        }
        double belief_sum = 0.0;
        for (double prob : initial_belief) {
            if (prob < 0.0) {
                std::cout << "DEBUG: negative belief probability: " << prob << std::endl;
                return false;
            }
            belief_sum += prob;
        }
        if (std::abs(belief_sum - 1.0) > 1e-6) {
            std::cout << "DEBUG: belief_sum != 1.0: " << belief_sum << std::endl;
            return false;
        }
        
        // Check models
        if (!transition_model.is_valid()) {
            std::cout << "DEBUG: transition_model.is_valid() = false" << std::endl;
            return false;
        }
        if (!observation_model.is_valid()) {
            std::cout << "DEBUG: observation_model.is_valid() = false" << std::endl;
            return false;
        }
        
        // Check rewards
        if (rewards_leader.size() != states.size()) {
            std::cout << "DEBUG: rewards_leader.size() != states.size(): " << rewards_leader.size() << " != " << states.size() << std::endl;
            return false;
        }
        for (const auto& state_rewards : rewards_leader) {
            if (state_rewards.size() != actions[0].size() * actions[1].size()) {
                std::cout << "DEBUG: state_rewards.size() != actions[0].size() * actions[1].size(): " 
                         << state_rewards.size() << " != " << actions[0].size() * actions[1].size() << std::endl;
                return false;
            }
        }
        
        if (rewards_follower.size() != states.size()) {
            std::cout << "DEBUG: rewards_follower.size() != states.size(): " << rewards_follower.size() << " != " << states.size() << std::endl;
            return false;
        }
        for (const auto& state_rewards : rewards_follower) {
            if (state_rewards.size() != actions[0].size() * actions[1].size()) {
                std::cout << "DEBUG: state_rewards.size() != actions[0].size() * actions[1].size(): " 
                         << state_rewards.size() << " != " << actions[0].size() * actions[1].size() << std::endl;
                return false;
            }
        }
        
        return true;
    }

    std::string POSGProblem::to_string() const {
        std::ostringstream oss;
        oss << "POSG Problem:\n";
        oss << "  Agents: " << num_agents << "\n";
        oss << "  Discount: " << discount_factor << "\n";
        oss << "  Value Type: " << value_type << "\n";
        oss << "  States: " << states.size() << "\n";
        // Defensive: check actions and observations are non-empty before accessing
        size_t leader_actions = (actions.size() > 0) ? actions[0].size() : 0;
        size_t follower_actions = (actions.size() > 1) ? actions[1].size() : 0;
        size_t leader_obs = (observations.size() > 0) ? observations[0].size() : 0;
        size_t follower_obs = (observations.size() > 1) ? observations[1].size() : 0;
        oss << "  Actions: " << leader_actions << " (leader), " << follower_actions << " (follower)\n";
        oss << "  Observations: " << leader_obs << " (leader), " << follower_obs << " (follower)\n";
        return oss.str();
    }

    POSGParser::POSGParser(const std::string& filename, int planning_horizon)
        : filename(filename), planning_horizon_(planning_horizon) {}

    POSGProblem POSGParser::parse() {
        std::ifstream file(filename);
        if (!file) {
            throw std::runtime_error("Could not open file: " + filename);
        }

        enum class ParseState { Header, States, Actions, Observations, Start, Transitions, ObsModel, Rewards, None };
        ParseState currentState = ParseState::Header;
        
        std::string line;
        int agent_idx_actions = 0;
        int agent_idx_obs = 0;

        bool any_content = false;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            any_content = true;
            std::istringstream iss(line);
            std::string token;
            iss >> token;
            
            if (token == "agents:") {
                iss >> problem.num_agents;
                problem.actions.resize(problem.num_agents);
                problem.observations.resize(problem.num_agents);
                std::cout << "[DEBUG] agents: " << problem.num_agents << std::endl;
            } else if (token == "discount:") {
                iss >> problem.discount_factor;
                std::cout << "[DEBUG] discount: " << problem.discount_factor << std::endl;
            } else if (token == "values:" || token == "value:") {
                iss >> problem.value_type;
                std::cout << "[DEBUG] value_type: " << problem.value_type << std::endl;
            } else if (token == "states:") {
                currentState = ParseState::States;
                int state_id;
                while (iss >> state_id) {
                    problem.states.push_back(state_id);
                }
                std::cout << "[DEBUG] states: ";
                for (auto s : problem.states) std::cout << s << " ";
                std::cout << std::endl;
            } else if (token == "actions:") {
                currentState = ParseState::Actions;
                agent_idx_actions = 0;
            } else if (token == "observations:") {
                currentState = ParseState::Observations;
                agent_idx_obs = 0;
            } else if (token == "start:") {
                currentState = ParseState::Start;
                double prob;
                while (iss >> prob) {
                    problem.initial_belief.push_back(prob);
                }
                std::cout << "[DEBUG] initial_belief: ";
                for (auto p : problem.initial_belief) std::cout << p << " ";
                std::cout << std::endl;
            } else if (token.rfind("T:", 0) == 0) {
                currentState = ParseState::Transitions;
                // Transition logic is complex and should be handled carefully
            } else if (token.rfind("O:", 0) == 0) {
                currentState = ParseState::ObsModel;
                 // Observation model logic is complex
            } else if (token == "R:") {
                currentState = ParseState::Rewards;
                // Rewards logic
            } else {
                // Handle data lines based on current state
                switch (currentState) {
                    case ParseState::Actions:
                        if (agent_idx_actions < problem.num_agents) {
                            iss.clear();
                            iss.str(line);
                            int action_id;
                            while (iss >> action_id) {
                                problem.actions[agent_idx_actions].push_back(action_id);
                            }
                            std::cout << "[DEBUG] actions[" << agent_idx_actions << "]: ";
                            for (auto a : problem.actions[agent_idx_actions]) std::cout << a << " ";
                            std::cout << std::endl;
                            agent_idx_actions++;
                        }
                        break;
                    case ParseState::Observations:
                        if (agent_idx_obs < problem.num_agents) {
                            iss.clear();
                            iss.str(line);
                            int obs_id;
                            while (iss >> obs_id) {
                                problem.observations[agent_idx_obs].push_back(obs_id);
                            }
                            std::cout << "[DEBUG] observations[" << agent_idx_obs << "]: ";
                            for (auto o : problem.observations[agent_idx_obs]) std::cout << o << " ";
                            std::cout << std::endl;
                            agent_idx_obs++;
                        }
                        break;
                    default:
                        // Other data lines are handled with their headers (T:, O:, R:)
                break;
            }
        }
        }

        // Defensive: If file is empty, throw
        if (!any_content) {
            throw std::runtime_error("File is empty: " + filename);
        }

        // Defensive: Check required fields
        if (problem.num_agents != 2) {
            throw std::runtime_error("Invalid or missing 'agents:' field in file: " + filename);
        }
        if (problem.states.empty()) {
            throw std::runtime_error("Missing 'states:' field in file: " + filename);
        }
        if (problem.actions.size() != problem.num_agents) {
            throw std::runtime_error("Missing or incomplete 'actions:' field in file: " + filename);
        }
        if (problem.observations.size() != problem.num_agents) {
            throw std::runtime_error("Missing or incomplete 'observations:' field in file: " + filename);
        }
        if (problem.initial_belief.size() != problem.states.size()) {
            throw std::runtime_error("Missing or invalid 'start:' field in file: " + filename);
        }

        // DENSE MODEL VALIDATION: Check all transitions and observations are defined and sum to 1
        int num_states = problem.states.size();
        int num_leader_actions = problem.actions[0].size();
        int num_follower_actions = problem.actions[1].size();
        int num_leader_obs = problem.observations[0].size();
        int num_follower_obs = problem.observations[1].size();
        for (int s = 0; s < num_states; ++s) {
            for (int aL = 0; aL < num_leader_actions; ++aL) {
                for (int aF = 0; aF < num_follower_actions; ++aF) {
                    posg_core::JointAction joint_action(posg_core::Action(aL, 0), posg_core::Action(aF, 1));
                    // Check transitions
                    double trans_sum = 0.0;
                    for (int s_next = 0; s_next < num_states; ++s_next) {
                        double p = problem.transition_model.get_transition_probability(s, joint_action, s_next);
                        if (p < 0.0) throw std::runtime_error("Negative transition probability for (s=" + std::to_string(s) + ", aL=" + std::to_string(aL) + ", aF=" + std::to_string(aF) + ", s'=" + std::to_string(s_next) + ") in file: " + filename);
                        trans_sum += p;
                    }
                    if (trans_sum > 1.0 + 1e-6) {
                        throw std::runtime_error("Transition probabilities for (s=" + std::to_string(s) + ", aL=" + std::to_string(aL) + ", aF=" + std::to_string(aF) + ") sum to more than 1 (sum=" + std::to_string(trans_sum) + ") in file: " + filename);
                    } else if (trans_sum < 1.0 - 1e-6) {
                        std::cout << "[WARNING] Transition probabilities for (s=" << s << ", aL=" << aL << ", aF=" << aF << ") sum to less than 1 (sum=" << trans_sum << ") in file: " << filename << std::endl;
                    }
                    // Check observations
                    double obs_sum = 0.0;
                    for (int zL = 0; zL < num_leader_obs; ++zL) {
                        for (int zF = 0; zF < num_follower_obs; ++zF) {
                            posg_core::JointObservation joint_obs(posg_core::Observation(zL, 0), posg_core::Observation(zF, 1));
                            double p = problem.observation_model.get_observation_probability(s, joint_action, joint_obs);
                            if (p < 0.0) throw std::runtime_error("Negative observation probability for (s=" + std::to_string(s) + ", aL=" + std::to_string(aL) + ", aF=" + std::to_string(aF) + ", zL=" + std::to_string(zL) + ", zF=" + std::to_string(zF) + ") in file: " + filename);
                            obs_sum += p;
                        }
                    }
                    if (obs_sum > 1.0 + 1e-6) {
                        throw std::runtime_error("Observation probabilities for (s=" + std::to_string(s) + ", aL=" + std::to_string(aL) + ", aF=" + std::to_string(aF) + ") sum to more than 1 (sum=" + std::to_string(obs_sum) + ") in file: " + filename);
                    } else if (obs_sum < 1.0 - 1e-6) {
                        std::cout << "[WARNING] Observation probabilities for (s=" << s << ", aL=" << aL << ", aF=" << aF << ") sum to less than 1 (sum=" << obs_sum << ") in file: " << filename << std::endl;
                    }
                }
            }
        }
        
        problem.horizon = planning_horizon_;

        // After all observations are parsed and problem.observation_model is fully populated:
        problem.observation_model.normalize();
        
        return problem;
    }

    bool POSGParser::validate() const {
        return problem.is_valid();
    }

    std::vector<std::string> POSGParser::get_validation_errors() const {
        std::vector<std::string> errors;
        
        if (problem.num_agents != 2) {
            errors.push_back("Number of agents must be 2, got " + std::to_string(problem.num_agents));
        }
        
        if (problem.discount_factor < 0.0 || problem.discount_factor > 1.0) {
            errors.push_back("Discount factor must be between 0 and 1, got " + std::to_string(problem.discount_factor));
        }
        
        if (problem.states.empty()) {
            errors.push_back("No states defined");
        }
        
        if (problem.actions.size() != problem.num_agents) {
            errors.push_back("Number of action sets (" + std::to_string(problem.actions.size()) + 
                           ") does not match number of agents (" + std::to_string(problem.num_agents) + ")");
        }
        
        if (problem.observations.size() != problem.num_agents) {
            errors.push_back("Number of observation sets (" + std::to_string(problem.observations.size()) + 
                           ") does not match number of agents (" + std::to_string(problem.num_agents) + ")");
        }
        
        if (problem.initial_belief.size() != problem.states.size()) {
            errors.push_back("Initial belief size (" + std::to_string(problem.initial_belief.size()) + 
                           ") does not match number of states (" + std::to_string(problem.states.size()) + ")");
        }
        
        double belief_sum = 0.0;
        for (double prob : problem.initial_belief) {
            if (prob < 0.0) {
                errors.push_back("Initial belief contains negative probability: " + std::to_string(prob));
            }
            belief_sum += prob;
        }
        if (std::abs(belief_sum - 1.0) > 1e-6) {
            errors.push_back("Initial belief probabilities do not sum to 1.0, got " + std::to_string(belief_sum));
        }
        
        return errors;
    }

    std::vector<std::string> POSGParser::split_line(const std::string& line) {
        std::vector<std::string> tokens;
        std::istringstream iss(line);
        std::string token;
        while (iss >> token) {
            tokens.push_back(token);
        }
        return tokens;
    }

    std::vector<double> POSGParser::parse_probabilities(const std::vector<std::string>& tokens, int start_idx) {
        std::vector<double> probabilities;
        for (size_t i = start_idx; i < tokens.size(); ++i) {
            try {
                probabilities.push_back(std::stod(tokens[i]));
            } catch (const std::exception& e) {
                throw std::runtime_error("Invalid probability value: " + tokens[i]);
            }
        }
        return probabilities;
    }

} // namespace posg_parser 