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
        if (num_agents != 2) {
            std::cout << "DEBUG: num_agents != 2: " << num_agents << std::endl;
            return false;
        }
        if (states.empty() || actions.size() != num_agents || observations.size() != num_agents) {
            std::cout << "DEBUG: states/actions/observations missing or wrong size" << std::endl;
            return false;
        }
        int num_states = states.size();
        int num_leader_actions = actions[0].size();
        int num_follower_actions = actions[1].size();
        int num_leader_obs = observations[0].size();
        int num_follower_obs = observations[1].size();
        // Check transitions
        for (int s = 0; s < num_states; ++s) {
            for (int aL = 0; aL < num_leader_actions; ++aL) {
                for (int aF = 0; aF < num_follower_actions; ++aF) {
                    posg_core::JointAction ja(posg_core::Action(aL, 0), posg_core::Action(aF, 1));
                    double sum = 0.0;
                    bool any_defined = false;
                    for (int s2 = 0; s2 < num_states; ++s2) {
                        double p = transition_model.get_transition_probability(s, ja, s2);
                        if (p < 0.0) {
                            std::cout << "DEBUG: Negative transition prob for (s=" << s << ",aL=" << aL << ",aF=" << aF << ",s'=" << s2 << ")" << std::endl;
                            return false;
                        }
                        sum += p;
                        if (p > 0.0) any_defined = true;
                    }
                    if (!any_defined) {
                        std::cout << "DEBUG: Missing transition for (s=" << s << ",aL=" << aL << ",aF=" << aF << ")" << std::endl;
                        return false;
                    }
                    if (std::abs(sum - 1.0) > 1e-6) {
                        std::cout << "DEBUG: Transition sum != 1 for (s=" << s << ",aL=" << aL << ",aF=" << aF << ") sum=" << sum << std::endl;
                        return false;
                    }
                }
            }
        }
        // Check observations
        for (int s2 = 0; s2 < num_states; ++s2) {
            for (int aL = 0; aL < num_leader_actions; ++aL) {
                for (int aF = 0; aF < num_follower_actions; ++aF) {
                    posg_core::JointAction ja(posg_core::Action(aL, 0), posg_core::Action(aF, 1));
                    double sum = 0.0;
                    bool any_defined = false;
                    for (int zL = 0; zL < num_leader_obs; ++zL) {
                        for (int zF = 0; zF < num_follower_obs; ++zF) {
                            posg_core::JointObservation jo(posg_core::Observation(zL, 0), posg_core::Observation(zF, 1));
                            double p = observation_model.get_observation_probability(s2, ja, jo);
                            if (p < 0.0) {
                                std::cout << "DEBUG: Negative observation prob for (s'=" << s2 << ",aL=" << aL << ",aF=" << aF << ",zL=" << zL << ",zF=" << zF << ")" << std::endl;
                                return false;
                            }
                            sum += p;
                            if (p > 0.0) any_defined = true;
                        }
                    }
                    if (!any_defined) {
                        std::cout << "DEBUG: Missing observation for (s'=" << s2 << ",aL=" << aL << ",aF=" << aF << ")" << std::endl;
                        return false;
                    }
                    if (std::abs(sum - 1.0) > 1e-6) {
                        std::cout << "DEBUG: Observation sum != 1 for (s'=" << s2 << ",aL=" << aL << ",aF=" << aF << ") sum=" << sum << std::endl;
                        return false;
                    }
                }
            }
        }
        // Check rewards
        for (int s = 0; s < num_states; ++s) {
            for (int aL = 0; aL < num_leader_actions; ++aL) {
                for (int aF = 0; aF < num_follower_actions; ++aF) {
                    int idx = aL + aF * num_leader_actions;
                    bool leader_defined = rewards_leader.size() > s && rewards_leader[s].size() > idx;
                    bool follower_defined = rewards_follower.size() > s && rewards_follower[s].size() > idx;
                    if (!leader_defined || !follower_defined) {
                        std::cout << "DEBUG: Missing reward for (s=" << s << ",aL=" << aL << ",aF=" << aF << ")" << std::endl;
                        return false;
                    }
                }
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

    POSGParser::POSGParser(const std::string& filename, int planning_horizon, bool allow_sparse)
        : filename(filename), planning_horizon_(planning_horizon), allow_sparse_(allow_sparse) {}

    POSGProblem POSGParser::parse() {
        std::ifstream file(filename);
        if (!file) {
            throw std::runtime_error("Could not open file: " + filename);
        }
        
        // Collect T:, O:, R: lines to process after model initialization
        std::vector<std::string> transition_lines;
        std::vector<std::string> observation_lines;
        std::vector<std::string> reward_lines;

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
                // Collect transition line to process after model initialization
                transition_lines.push_back(line);
            } else if (token.rfind("O:", 0) == 0) {
                currentState = ParseState::ObsModel;
                // Collect observation line to process after model initialization
                observation_lines.push_back(line);
            } else if (token.rfind("R:", 0) == 0) {
                currentState = ParseState::Rewards;
                // Collect reward line to process after model initialization
                reward_lines.push_back(line);
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

        // Initialize transition and observation models BEFORE parsing T:, O:, R: lines
        int num_states = problem.states.size();
        int num_leader_actions = problem.actions[0].size();
        int num_follower_actions = problem.actions[1].size();
        int num_leader_observations = problem.observations[0].size();
        int num_follower_observations = problem.observations[1].size();
        

        
        try {
            problem.transition_model = posg_core::TransitionModel(num_states, num_leader_actions, num_follower_actions);
            problem.observation_model = posg_core::ObservationModel(num_states, num_leader_actions, num_follower_actions, 
                                                                  num_leader_observations, num_follower_observations);
        } catch (const std::exception& e) {
            std::cout << "[DEBUG] Error initializing models: " << e.what() << std::endl;
            throw;
        }

        // Process collected transition lines
        for (const auto& line : transition_lines) {
            std::vector<std::string> tokens = split_line(line);
            if (tokens.size() >= 9) {
                int aL = std::stoi(tokens[1]);
                int aF = std::stoi(tokens[2]);
                int s = std::stoi(tokens[4]); // Skip the colon at tokens[3]
                int s_next = std::stoi(tokens[6]); // Skip the colon at tokens[5]
                double prob = std::stod(tokens[8]); // Skip the colon at tokens[7]
                
                posg_core::JointAction joint_action(posg_core::Action(aL, 0), posg_core::Action(aF, 1));
                problem.transition_model.set_transition_probability(s, joint_action, s_next, prob);
            }
        }

        // Process collected observation lines
        for (const auto& line : observation_lines) {
            std::vector<std::string> tokens = split_line(line);
            if (tokens.size() >= 10) {
                int aL = std::stoi(tokens[1]);
                int aF = std::stoi(tokens[2]);
                int s = std::stoi(tokens[4]); // Skip the colon at tokens[3]
                int zL = std::stoi(tokens[6]); // Skip the colon at tokens[5]
                int zF = std::stoi(tokens[7]);
                double prob = std::stod(tokens[9]); // Skip the colon at tokens[8]
                
                posg_core::JointAction joint_action(posg_core::Action(aL, 0), posg_core::Action(aF, 1));
                posg_core::JointObservation joint_obs(posg_core::Observation(zL, 0), posg_core::Observation(zF, 1));
                problem.observation_model.set_observation_probability(s, joint_action, joint_obs, prob);
            }
        }

        // Process collected reward lines
        for (const auto& line : reward_lines) {
            std::vector<std::string> tokens = split_line(line);
            if (tokens.size() >= 12) {
                int aL = std::stoi(tokens[1]);
                int aF = std::stoi(tokens[2]);
                int s = std::stoi(tokens[4]); // Skip the colon at tokens[3]
                double rL = std::stod(tokens[10]); // Skip the colons and asterisks
                double rF = std::stod(tokens[11]);
                
                // Store rewards in the problem structure
                if (problem.rewards_leader.size() <= s) {
                    problem.rewards_leader.resize(s + 1);
                }
                if (problem.rewards_follower.size() <= s) {
                    problem.rewards_follower.resize(s + 1);
                }
                
                int action_index = aL + aF * problem.actions[0].size();
                if (problem.rewards_leader[s].size() <= action_index) {
                    problem.rewards_leader[s].resize(action_index + 1);
                }
                if (problem.rewards_follower[s].size() <= action_index) {
                    problem.rewards_follower[s].resize(action_index + 1);
                }
                
                problem.rewards_leader[s][action_index] = rL;
                problem.rewards_follower[s][action_index] = rF;
            }
        }

        // DENSE MODEL VALIDATION: Check all transitions and observations are defined and sum to 1
        if (!problem.transition_model.is_valid()) {
            std::vector<std::string> errors = get_validation_errors();
            std::string error_msg = "Transition model validation failed: ";
            for (const auto& error : errors) {
                error_msg += error + "; ";
            }
            throw std::runtime_error(error_msg);
        }
        
        problem.horizon = planning_horizon_;

        // After all observations are parsed and problem.observation_model is fully populated:
        problem.observation_model.normalize();
        
        // After processing all lines and setting transitions/observations/rewards:
        // Fill in missing transitions if allow_sparse_
        if (allow_sparse_) {
            int num_states = problem.states.size();
            int num_leader_actions = problem.actions[0].size();
            int num_follower_actions = problem.actions[1].size();
            for (int s = 0; s < num_states; ++s) {
                for (int aL = 0; aL < num_leader_actions; ++aL) {
                    for (int aF = 0; aF < num_follower_actions; ++aF) {
                        posg_core::JointAction joint_action(posg_core::Action(aL, 0), posg_core::Action(aF, 1));
                        double sum = 0.0;
                        for (int s_next = 0; s_next < num_states; ++s_next) {
                            sum += problem.transition_model.get_transition_probability(s, joint_action, s_next);
                        }
                        if (std::abs(sum) < 1e-8) {
                            // No transition specified, use identity
                            for (int s_next = 0; s_next < num_states; ++s_next) {
                                double val = (s_next == s) ? 1.0 : 0.0;
                                problem.transition_model.set_transition_probability(s, joint_action, s_next, val);
                            }
                            std::cerr << "[SPARSE WARNING] Defaulted transition for (s=" << s << ", aL=" << aL << ", aF=" << aF << ") to identity." << std::endl;
                        }
                    }
                }
            }
            // Fill in missing observations
            int num_leader_obs = problem.observations[0].size();
            int num_follower_obs = problem.observations[1].size();
            for (int s = 0; s < num_states; ++s) {
                for (int aL = 0; aL < num_leader_actions; ++aL) {
                    for (int aF = 0; aF < num_follower_actions; ++aF) {
                        posg_core::JointAction joint_action(posg_core::Action(aL, 0), posg_core::Action(aF, 1));
                        for (int s_next = 0; s_next < num_states; ++s_next) {
                            double obs_sum = 0.0;
                            for (int zL = 0; zL < num_leader_obs; ++zL) {
                                for (int zF = 0; zF < num_follower_obs; ++zF) {
                                    posg_core::JointObservation joint_obs(posg_core::Observation(zL, 0), posg_core::Observation(zF, 1));
                                    obs_sum += problem.observation_model.get_observation_probability(s_next, joint_action, joint_obs);
                                }
                            }
                            if (std::abs(obs_sum) < 1e-8) {
                                // No observation specified, set (0,0)=1
                                for (int zL = 0; zL < num_leader_obs; ++zL) {
                                    for (int zF = 0; zF < num_follower_obs; ++zF) {
                                        double val = (zL == 0 && zF == 0) ? 1.0 : 0.0;
                                        posg_core::JointObservation joint_obs(posg_core::Observation(zL, 0), posg_core::Observation(zF, 1));
                                        problem.observation_model.set_observation_probability(s_next, joint_action, joint_obs, val);
                                    }
                                }
                                std::cerr << "[SPARSE WARNING] Defaulted observation for (s'=" << s_next << ", aL=" << aL << ", aF=" << aF << ") to (zL=0,zF=0)." << std::endl;
                            }
                        }
                    }
                }
            }
            // Fill in missing rewards
            for (int s = 0; s < num_states; ++s) {
                for (int aL = 0; aL < num_leader_actions; ++aL) {
                    for (int aF = 0; aF < num_follower_actions; ++aF) {
                        int action_index = aL + aF * num_leader_actions;
                        if (problem.rewards_leader.size() <= s || problem.rewards_leader[s].size() <= action_index) {
                            if (problem.rewards_leader.size() <= s) problem.rewards_leader.resize(s+1);
                            if (problem.rewards_leader[s].size() <= action_index) problem.rewards_leader[s].resize(action_index+1, 0.0);
                            std::cerr << "[SPARSE WARNING] Defaulted leader reward for (s=" << s << ", aL=" << aL << ", aF=" << aF << ") to 0." << std::endl;
                        }
                        if (problem.rewards_follower.size() <= s || problem.rewards_follower[s].size() <= action_index) {
                            if (problem.rewards_follower.size() <= s) problem.rewards_follower.resize(s+1);
                            if (problem.rewards_follower[s].size() <= action_index) problem.rewards_follower[s].resize(action_index+1, 0.0);
                            std::cerr << "[SPARSE WARNING] Defaulted follower reward for (s=" << s << ", aL=" << aL << ", aF=" << aF << ") to 0." << std::endl;
                        }
                    }
                }
            }
        }

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