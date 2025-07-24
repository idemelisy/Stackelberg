// conditional_occupancy_state.cpp
// ---------------------------------
// Implements the ConditionalOccupancyState class for the AAAI 2026 paper.
// What: Manages conditional occupancy states (distributions over state, leader_history conditioned on follower_history)
// Why: This enables the decomposition used in the CMDP approach for value function computation
// Fit: Used in value function computation and PBVI algorithms

#include "../include/conditional_occupancy_state.hpp"
#include "../include/common.hpp"
#include "../include/transition_model.hpp"
#include "../include/observation_model.hpp"
#include "../include/credible_mdp.hpp"
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <iostream> // Added for debug output

namespace posg_core {

    ConditionalOccupancyState::ConditionalOccupancyState() 
        : follower_history(1), timestep(0), is_normalized(true) {}

    ConditionalOccupancyState::ConditionalOccupancyState(const AgentHistory& follower_history) 
        : follower_history(follower_history), timestep(0), is_normalized(true) {}

    ConditionalOccupancyState::ConditionalOccupancyState(const AgentHistory& follower_history, 
                                                        const std::vector<double>& initial_belief)
        : follower_history(follower_history), timestep(0), is_normalized(false) {
        // Create empty leader history for initial state
        AgentHistory empty_leader_history(0);
        
        // Set initial conditional occupancy: Pr(state, empty_leader_history | follower_history) = initial_belief[state]
        for (size_t state = 0; state < initial_belief.size(); ++state) {
            if (initial_belief[state] > 0.0) {
                set_conditional_occupancy(state, empty_leader_history, initial_belief[state]);
            }
        }
        normalize();
    }

    ConditionalOccupancyState ConditionalOccupancyState::update(const Action& leader_action, 
                                                               const Action& follower_action,
                                                               const Observation& leader_obs, 
                                                               const Observation& follower_obs,
                                                               const TransitionModel& transition_model, 
                                                               const ObservationModel& observation_model) const {
        ConditionalOccupancyState new_conditional(follower_history);
        new_conditional.timestep = timestep + 1;
        
        double normalization_constant = 0.0;
        
        // For each possible next state
        for (int next_state = 0; next_state < transition_model.get_num_states(); ++next_state) {
            // For each current conditional occupancy state
            for (const auto& [current_state, leader_histories] : conditional_distribution) {
                for (const auto& [leader_history, current_prob] : leader_histories) {
                    if (current_prob <= 0.0) continue;
                    
                    // Create joint action and observation
                    JointAction joint_action(leader_action, follower_action);
                    JointObservation joint_obs(leader_obs, follower_obs);
                    
                    // Compute transition and observation probabilities
                    double transition_prob = transition_model.get_transition_probability(
                        current_state, joint_action, next_state);
                    double obs_prob = observation_model.get_observation_probability(
                        next_state, joint_action, joint_obs);
                    
                    // Create new leader history
                    AgentHistory new_leader_history = leader_history;
                    new_leader_history.add_action(leader_action);
                    new_leader_history.add_observation(leader_obs);
                    
                    // Update conditional occupancy probability
                    double new_prob = current_prob * transition_prob * obs_prob;
                    new_conditional.conditional_distribution[next_state][new_leader_history] += new_prob;
                    normalization_constant += new_prob;
                }
            }
        }
        
        // Normalize
        if (normalization_constant > 0.0) {
            for (auto& [state, leader_histories] : new_conditional.conditional_distribution) {
                for (auto& [leader_history, prob] : leader_histories) {
                    prob /= normalization_constant;
                }
            }
        }
        
        new_conditional.is_normalized = true;
        return new_conditional;
    }

    ConditionalOccupancyState ConditionalOccupancyState::tau_zF(const LeaderDecisionRule& sigma_L, 
                                                                const Action& a_F,
                                                                const Observation& z_F, 
                                                                const TransitionModel& transition_model,
                                                                const ObservationModel& observation_model) const {
        /**
         * From Paper: Lemma 4.2
         * c'(s', h'L) ∝ Σ_s c(s, hL)σL(aL|hL)p(s', zL, zF|s, aL, aF)
         * where h'L = (hL, aL, zL)
         */
        
        ConditionalOccupancyState new_conditional(follower_history);
        new_conditional.timestep = timestep + 1;
        
        // For each possible next state s'
        for (int next_state = 0; next_state < transition_model.get_num_states(); ++next_state) {
            // For each current conditional occupancy state (s, hL)
            for (const auto& [current_state, leader_histories] : conditional_distribution) {
                for (const auto& [leader_history, current_prob] : leader_histories) {
                    if (current_prob <= 0.0) continue;
                    
                    // For each possible leader action aL
                    for (int aL_id = 0; aL_id < transition_model.get_num_leader_actions(); ++aL_id) {
                        Action aL(aL_id, 0); // Leader action
                        
                        // Get leader action probability: σL(aL|hL)
                        double sigma_prob = sigma_L.get_action_probability(leader_history, aL);
                        if (sigma_prob <= 0.0) continue;
                        
                        // For each possible leader observation zL
                        for (int zL_id = 0; zL_id < observation_model.get_num_leader_observations(); ++zL_id) {
                            Observation zL(zL_id, 0); // Leader observation
                            
                            // Create joint action and observation
                            JointAction joint_action(aL, a_F);
                            JointObservation joint_obs(zL, z_F);
                            
                            // Compute transition and observation probabilities
                            double transition_prob = transition_model.get_transition_probability(
                                current_state, joint_action, next_state);
                            double obs_prob = observation_model.get_observation_probability(
                                next_state, joint_action, joint_obs);
                            
                            // Create new leader history: h'L = (hL, aL, zL)
                            AgentHistory new_leader_history = leader_history;
                            new_leader_history.add_action(aL);
                            new_leader_history.add_observation(zL);
                            
                            // Update conditional occupancy probability
                            double new_prob = current_prob * sigma_prob * transition_prob * obs_prob;
                            new_conditional.conditional_distribution[next_state][new_leader_history] += new_prob;
                        }
                    }
                }
            }
        }
        
        // Normalize
        new_conditional.normalize();
        return new_conditional;
    }

    ConditionalOccupancyState ConditionalOccupancyState::tau_tilde_zF(const LeaderDecisionRule& sigma_L, 
                                                                      const Action& a_F,
                                                                      const Observation& z_F, 
                                                                      const TransitionModel& transition_model,
                                                                      const ObservationModel& observation_model) const {
        /**
         * From Paper: τ̃zF(c, σL, aF) = ηzF(c, σL, aF) · τzF(c, σL, aF)
         */
        
        // Compute normalization constant
        double eta = eta_zF(sigma_L, a_F, z_F, transition_model, observation_model);
        
        // Compute τzF
        ConditionalOccupancyState tau_result = tau_zF(sigma_L, a_F, z_F, transition_model, observation_model);
        
        // Multiply by normalization constant
        for (auto& [state, leader_histories] : tau_result.conditional_distribution) {
            for (auto& [leader_history, prob] : leader_histories) {
                prob *= eta;
            }
        }
        
        tau_result.is_normalized = false; // No longer normalized after scaling
        return tau_result;
    }

    double ConditionalOccupancyState::eta_zF(const LeaderDecisionRule& sigma_L, 
                                             const Action& a_F,
                                             const Observation& z_F, 
                                             const TransitionModel& transition_model,
                                             const ObservationModel& observation_model) const {
        /**
         * From Paper: ηzF(c, σL, aF) = Σ_s,s',hL,aL p(s', zL, zF|s, aL, aF) · σL(aL|hL) · c(s, hL)
         */
        
        double eta = 0.0;
        
        // For each current conditional occupancy state (s, hL)
        for (const auto& [current_state, leader_histories] : conditional_distribution) {
            for (const auto& [leader_history, current_prob] : leader_histories) {
                if (current_prob <= 0.0) continue;
                
                // For each possible leader action aL
                for (int aL_id = 0; aL_id < transition_model.get_num_leader_actions(); ++aL_id) {
                    Action aL(aL_id, 0); // Leader action
                    
                    // Get leader action probability: σL(aL|hL)
                    double sigma_prob = sigma_L.get_action_probability(leader_history, aL);
                    if (sigma_prob <= 0.0) continue;
                    
                    // For each possible next state s'
                    for (int next_state = 0; next_state < transition_model.get_num_states(); ++next_state) {
                        // For each possible leader observation zL
                        for (int zL_id = 0; zL_id < observation_model.get_num_leader_observations(); ++zL_id) {
                            Observation zL(zL_id, 0); // Leader observation
                            
                            // Create joint action and observation
                            JointAction joint_action(aL, a_F);
                            JointObservation joint_obs(zL, z_F);
                            
                            // Compute transition and observation probabilities
                            double transition_prob = transition_model.get_transition_probability(
                                current_state, joint_action, next_state);
                            double obs_prob = observation_model.get_observation_probability(
                                next_state, joint_action, joint_obs);
                            
                            // Add to normalization constant
                            eta += current_prob * sigma_prob * transition_prob * obs_prob;
                        }
                    }
                }
            }
        }
        
        return eta;
    }

    double ConditionalOccupancyState::rho_i(int player_id, const LeaderDecisionRule& sigma_L, 
                                           const Action& a_F,
                                           const std::function<double(int, const Action&, const Action&)>& reward_function) const {
        /**
         * From Paper: ρi(c, σL, aF) = E[ri(s, aL, aF)|c, σL, aF]
         * = Σ_s,hL c(s, hL) · Σ_aL σL(aL|hL) · ri(s, aL, aF)
         */
        
        double expected_payoff = 0.0;
        
        // For each conditional occupancy state (s, hL)
        for (const auto& [state, leader_histories] : conditional_distribution) {
            for (const auto& [leader_history, conditional_prob] : leader_histories) {
                if (conditional_prob <= 0.0) continue;
                // Loop over all actions with nonzero probability for this leader_history
                // TODO: In the future, enumerate all possible leader actions if needed
                for (const auto& [aL, sigma_prob] : sigma_L.get_action_probabilities(leader_history)) {
                    if (sigma_prob <= 0.0) continue;
                    // Compute reward: ri(s, aL, aF)
                    double reward = reward_function(state, aL, a_F);
                    // Add to expected payoff
                    expected_payoff += conditional_prob * sigma_prob * reward;
                }
            }
        }
        
        return expected_payoff;
    }

    double ConditionalOccupancyState::get_conditional_occupancy(int state, 
                                                               const AgentHistory& leader_history) const {
        auto state_it = conditional_distribution.find(state);
        if (state_it == conditional_distribution.end()) return 0.0;
        
        auto leader_it = state_it->second.find(leader_history);
        return (leader_it != state_it->second.end()) ? leader_it->second : 0.0;
    }

    void ConditionalOccupancyState::set_conditional_occupancy(int state, 
                                                             const AgentHistory& leader_history, 
                                                             double probability) {
        if (probability > 0.0) {
            conditional_distribution[state][leader_history] = probability;
        } else {
            auto state_it = conditional_distribution.find(state);
            if (state_it != conditional_distribution.end()) {
                state_it->second.erase(leader_history);
                if (state_it->second.empty()) {
                    conditional_distribution.erase(state);
                }
            }
        }
        is_normalized = false;
    }

    bool ConditionalOccupancyState::is_valid(bool allow_empty) const {
        if (!is_normalized) {
            const_cast<ConditionalOccupancyState*>(this)->normalize();
        }
        if (conditional_distribution.empty()) return allow_empty;
        double sum = 0.0;
        for (const auto& [state, leader_histories] : conditional_distribution) {
            for (const auto& [leader_history, prob] : leader_histories) {
                if (prob < 0.0) return false;
                sum += prob;
            }
        }
        return std::abs(sum - 1.0) < 1e-6;
    }

    bool ConditionalOccupancyState::is_valid() const {
        return is_valid(true);
    }

    std::unordered_map<int, double> ConditionalOccupancyState::get_state_marginal() const {
        if (!is_normalized) {
            const_cast<ConditionalOccupancyState*>(this)->normalize();
        }
        std::unordered_map<int, double> marginal;
        double sum = 0.0;
        for (const auto& [state, leader_histories] : conditional_distribution) {
            double state_prob = 0.0;
            for (const auto& [leader_history, prob] : leader_histories) {
                state_prob += prob;
            }
            if (state_prob > 0.0) {
                marginal[state] = state_prob;
                sum += state_prob;
            }
        }
        // Normalize
        if (sum > 0.0) {
            for (auto& [state, prob] : marginal) {
                prob /= sum;
            }
        }
        return marginal;
    }

    std::unordered_map<AgentHistory, double> ConditionalOccupancyState::get_leader_history_marginal() const {
        if (!is_normalized) {
            const_cast<ConditionalOccupancyState*>(this)->normalize();
        }
        std::unordered_map<AgentHistory, double> marginal;
        double sum = 0.0;
        for (const auto& [state, leader_histories] : conditional_distribution) {
            for (const auto& [leader_history, prob] : leader_histories) {
                if (prob > 0.0) {
                    marginal[leader_history] += prob;
                    sum += prob;
                }
            }
        }
        // Normalize
        if (sum > 0.0) {
            for (auto& [history, prob] : marginal) {
                prob /= sum;
            }
        }
        return marginal;
    }

    double ConditionalOccupancyState::entropy() const {
        if (!is_normalized) {
            const_cast<ConditionalOccupancyState*>(this)->normalize();
        }
        double entropy_val = 0.0;
        for (const auto& [state, leader_histories] : conditional_distribution) {
            for (const auto& [leader_history, prob] : leader_histories) {
                if (prob > 0.0) {
                    entropy_val -= prob * std::log(prob); // Use natural log
                }
            }
        }
        return entropy_val;
    }

    double ConditionalOccupancyState::distance_to(const ConditionalOccupancyState& other) const {
        if (follower_history != other.follower_history) {
            std::cerr << "[DISTANCE ERROR] Mismatched follower histories detected.\n";
            std::cerr << "This: ";
            for (const auto& [state, leader_histories] : conditional_distribution) {
                for (const auto& [leader_history, _] : leader_histories) {
                    std::cerr << "(s=" << state << ", hL=" << leader_history.to_string() << ", hF=" << follower_history.to_string() << ") ";
                }
            }
            std::cerr << "\nOther: ";
            for (const auto& [state, leader_histories] : other.conditional_distribution) {
                for (const auto& [leader_history, _] : leader_histories) {
                    std::cerr << "(s=" << state << ", hL=" << leader_history.to_string() << ", hF=" << other.follower_history.to_string() << ") ";
                }
            }
            std::cerr << std::endl;
            throw std::runtime_error("distance_to: different follower histories");
        }
        double total_distance = 0.0;
        std::set<std::pair<int, AgentHistory>> all_tuples;
        for (const auto& [state, leader_histories] : conditional_distribution) {
            for (const auto& [leader_history, _] : leader_histories) {
                all_tuples.insert({state, leader_history});
            }
        }
        for (const auto& [state, leader_histories] : other.conditional_distribution) {
            for (const auto& [leader_history, _] : leader_histories) {
                all_tuples.insert({state, leader_history});
            }
        }
        for (const auto& [state, leader_history] : all_tuples) {
            double p1 = get_conditional_occupancy(state, leader_history);
            double p2 = other.get_conditional_occupancy(state, leader_history);
            if (p1 > 0.0 && p2 == 0.0)
                std::cout << "[DISTANCE DEBUG] Missing key in other: (s=" << state << ", hL=" << leader_history.to_string() << ")\n";
            if (p2 > 0.0 && p1 == 0.0)
                std::cout << "[DISTANCE DEBUG] Missing key in this: (s=" << state << ", hL=" << leader_history.to_string() << ")\n";
            total_distance += std::abs(p1 - p2);
        }
        return total_distance;
    }

    void ConditionalOccupancyState::normalize() {
        double sum = 0.0;
        for (const auto& [state, leader_histories] : conditional_distribution) {
            for (const auto& [leader_history, prob] : leader_histories) {
                sum += prob;
            }
        }
        if (sum > 0.0) {
            for (auto state_it = conditional_distribution.begin(); state_it != conditional_distribution.end(); ) {
                auto& leader_histories = state_it->second;
                for (auto leader_it = leader_histories.begin(); leader_it != leader_histories.end(); ) {
                    leader_it->second /= sum;
                    if (leader_it->second == 0.0) {
                        leader_it = leader_histories.erase(leader_it);
                    } else {
                        ++leader_it;
                    }
                }
                if (leader_histories.empty()) {
                    state_it = conditional_distribution.erase(state_it);
                } else {
                    ++state_it;
                }
            }
        } else {
            conditional_distribution.clear();
        }
        is_normalized = true;
    }

    std::string ConditionalOccupancyState::to_string() const {
        std::ostringstream oss;
        oss << "ConditionalOccupancyState(t=" << timestep 
            << ", follower_history=" << follower_history.to_string() << ", conditional={";
        
        bool first = true;
        for (const auto& [state, leader_histories] : conditional_distribution) {
            for (const auto& [leader_history, prob] : leader_histories) {
                if (!first) oss << ", ";
                oss << "(" << state << ", " << leader_history.to_string() << "):" 
                    << std::fixed << std::setprecision(3) << prob;
                first = false;
            }
        }
        oss << "})";
        return oss.str();
    }

    bool ConditionalOccupancyState::operator==(const ConditionalOccupancyState& other) const {
        if (timestep != other.timestep) return false;
        if (follower_history != other.follower_history) return false;
        return distance_to(other) < 1e-6;
    }

    bool ConditionalOccupancyState::operator!=(const ConditionalOccupancyState& other) const {
        return !(*this == other);
    }

    std::size_t ConditionalOccupancyState::Hash::operator()(const ConditionalOccupancyState& conditional_occupancy) const {
        std::size_t hash = std::hash<AgentHistory>{}(conditional_occupancy.follower_history);
        for (const auto& [state, leader_histories] : conditional_occupancy.conditional_distribution) {
            for (const auto& [leader_history, prob] : leader_histories) {
                hash ^= std::hash<int>{}(state) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                hash ^= std::hash<AgentHistory>{}(leader_history) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                hash ^= std::hash<double>{}(prob) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            }
        }
        return hash;
    }

} // namespace posg_core 