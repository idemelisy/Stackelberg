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

        file.clear();
        file.seekg(0, std::ios::beg);
        
        problem.transition_model = posg_core::TransitionModel(problem.states.size(), problem.actions[0].size(), problem.actions[1].size());
        problem.observation_model = posg_core::ObservationModel(problem.states.size(), problem.actions[0].size(), problem.actions[1].size(), problem.observations[0].size(), problem.observations[1].size());
        problem.rewards_leader.resize(problem.states.size(), std::vector<double>(problem.actions[0].size() * problem.actions[1].size()));
        problem.rewards_follower.resize(problem.states.size(), std::vector<double>(problem.actions[0].size() * problem.actions[1].size()));

        // Second pass for T, O, R which require full context
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            std::string token;
            iss >> token;
            
            if (token == "T:") {
                std::string rest_of_line;
                std::getline(iss, rest_of_line);
                std::istringstream rest(rest_of_line);
                std::string a1, a2, colon1, s_str, colon2, s_next_str, prob_str;
                rest >> a1;
                if (a1 == "*") {
                    std::string colon, uniform_str;
                    rest >> colon >> uniform_str;
                    if (uniform_str == "uniform") {
                        // T: * : uniform
                        int num_states = problem.states.size();
                        int num_leader_actions = problem.actions[0].size();
                        int num_follower_actions = problem.actions[1].size();
                        double uniform_prob = 1.0 / num_states;
                        for (int la = 0; la < num_leader_actions; ++la) {
                            for (int fa = 0; fa < num_follower_actions; ++fa) {
                                posg_core::JointAction joint_action(posg_core::Action(la,0), posg_core::Action(fa,1));
                                for (int s = 0; s < num_states; ++s) {
                                    for (int s_next = 0; s_next < num_states; ++s_next) {
                                        problem.transition_model.set_transition_probability(s, joint_action, s_next, uniform_prob);
                                    }
                                }
                            }
                        }
                        // std::cout << "[PARSER] Set all transitions to uniform." << std::endl;
                    }
                } else {
                    // Try to parse as numbers
                    int leader_action = std::stoi(a1);
                    rest >> a2;
                    int follower_action = std::stoi(a2);
                    rest >> colon1 >> s_str;
                    if (s_str == "identity") {
                        // T: <la> <fa> : identity
                        int num_states = problem.states.size();
                        posg_core::JointAction joint_action(posg_core::Action(leader_action,0), posg_core::Action(follower_action,1));
                        for (int s = 0; s < num_states; ++s) {
                            for (int s_next = 0; s_next < num_states; ++s_next) {
                                double prob = (s == s_next) ? 1.0 : 0.0;
                                problem.transition_model.set_transition_probability(s, joint_action, s_next, prob);
                            }
                        }
                        // std::cout << "[PARSER] Set identity transition for la=" << leader_action << ", fa=" << follower_action << std::endl;
                    } else {
                        // T: <la> <fa> : <s> : <s_next> : <prob>
                        int start_state = std::stoi(s_str);
                        char colon2_char, colon3_char;
                        rest >> colon2_char >> s_next_str >> colon3_char >> prob_str;
                        if (colon2_char != ':' || colon3_char != ':') {
                            throw std::runtime_error("Invalid transition format: expected ':' separators");
                        }
                        int end_state = std::stoi(s_next_str);
                        double prob;
                        try {
                            prob = std::stod(prob_str);
                        } catch (const std::exception& e) {
                            throw std::runtime_error("Invalid probability value in transition: " + prob_str);
                        }
                        problem.transition_model.set_transition_probability(start_state, posg_core::JointAction(posg_core::Action(leader_action,0), posg_core::Action(follower_action,1)), end_state, prob);
                        // std::cout << "[PARSER] Parsed T: la=" << leader_action << ", fa=" << follower_action
                                  // << ", s=" << start_state << ", s'=" << end_state << ", prob=" << prob << std::endl;
                    }
                }
            } else if (token == "O:") {
                std::string rest_of_line;
                std::getline(iss, rest_of_line);
                std::istringstream rest(rest_of_line);
                std::string a1, a2, colon1, s_str, colon2, o1_str, o2_str, colon3, prob_str;
                rest >> a1;
                if (a1 == "*") {
                    std::string colon, uniform_str;
                    rest >> colon >> uniform_str;
                    if (uniform_str == "uniform") {
                        // O: * : uniform
                        int num_states = problem.states.size();
                        int num_leader_actions = problem.actions[0].size();
                        int num_follower_actions = problem.actions[1].size();
                        int num_leader_obs = problem.observations[0].size();
                        int num_follower_obs = problem.observations[1].size();
                        int num_joint_obs = num_leader_obs * num_follower_obs;
                        for (int s = 0; s < num_states; ++s) {
                            for (int a = 0; a < num_leader_actions; ++a) {
                                for (int f = 0; f < num_follower_actions; ++f) {
                                    posg_core::JointAction joint_action(posg_core::Action(a, 0), posg_core::Action(f, 1));
                                    for (int o1 = 0; o1 < num_leader_obs; ++o1) {
                                        for (int o2 = 0; o2 < num_follower_obs; ++o2) {
                                            posg_core::JointObservation joint_obs(posg_core::Observation(o1, 0), posg_core::Observation(o2, 1));
                                            problem.observation_model.set_observation_probability(s, joint_action, joint_obs, 1.0 / num_joint_obs);
                                        }
                                    }
                                }
                            }
                        }
                        // std::cout << "[PARSER] Set all observation probabilities to uniform." << std::endl;
                    }
                } else {
                    // Try to parse O: <la> <fa> : identity
                    int leader_action, follower_action;
                    std::istringstream a1a2(a1);
                    if (a1a2 >> leader_action) {
                        rest >> a2;
                        std::istringstream a2ss(a2);
                        if (a2ss >> follower_action) {
                            std::string colon, identity_str;
                            rest >> colon >> identity_str;
                            if (identity_str == "identity") {
                                int num_states = problem.states.size();
                                int num_leader_obs = problem.observations[0].size();
                                int num_follower_obs = problem.observations[1].size();
                                posg_core::JointAction joint_action(posg_core::Action(leader_action, 0), posg_core::Action(follower_action, 1));
                                for (int s = 0; s < num_states; ++s) {
                                    for (int o1 = 0; o1 < num_leader_obs; ++o1) {
                                        for (int o2 = 0; o2 < num_follower_obs; ++o2) {
                                            posg_core::JointObservation joint_obs(posg_core::Observation(o1, 0), posg_core::Observation(o2, 1));
                                            // Identity: observation matches state index
                                            double prob = (o1 == s && o2 == s) ? 1.0 : 0.0;
                                            problem.observation_model.set_observation_probability(s, joint_action, joint_obs, prob);
                                        }
                                    }
                                }
                                // std::cout << "[PARSER] Set identity observation for la=" << leader_action << ", fa=" << follower_action << std::endl;
                                continue;
                            }
                        }
                    }
                    // Try to parse O: <la> <fa> : <s'> : <o1> <o2> : <prob>
                    rest.clear();
                    rest.str(rest_of_line);
                    int la, fa, s_next, o1, o2;
                    char c;
                    std::string prob_str;
                    if (rest >> la >> fa >> c && c == ':' && rest >> s_next >> c && c == ':' && rest >> o1 >> o2 >> c && c == ':' && rest >> prob_str) {
                        double prob;
                        try {
                            prob = std::stod(prob_str);
                        } catch (const std::exception& e) {
                            throw std::runtime_error("Invalid probability value in observation: " + prob_str);
                        }
                        posg_core::JointAction joint_action(posg_core::Action(la, 0), posg_core::Action(fa, 1));
                        posg_core::JointObservation joint_obs(posg_core::Observation(o1, 0), posg_core::Observation(o2, 1));
                        problem.observation_model.set_observation_probability(s_next, joint_action, joint_obs, prob);
                        // std::cout << "[PARSER] Set O: la=" << la << ", fa=" << fa << ", s'=" << s_next << ", o1=" << o1 << ", o2=" << o2 << ", prob=" << prob << std::endl;
                    }
                }
            } else if (token == "R:") {
                 int leader_action, follower_action;
                 char colon;
                 std::string state_token, star1, star2;
                 double r_leader, r_follower;
                 iss >> leader_action >> follower_action >> colon >> state_token >> colon >> star1 >> colon >> star2 >> colon >> r_leader >> r_follower;
                 int joint_action_idx = leader_action + follower_action * problem.actions[0].size();
                 if (state_token == "*") {
                     for (size_t s = 0; s < problem.states.size(); ++s) {
                         problem.rewards_leader[s][joint_action_idx] = r_leader;
                         problem.rewards_follower[s][joint_action_idx] = r_follower;
                     }
                 } else {
                     int start_state = std::stoi(state_token);
                     problem.rewards_leader[start_state][joint_action_idx] = r_leader;
                     problem.rewards_follower[start_state][joint_action_idx] = r_follower;
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