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
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <sstream>
#include <iomanip>

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

    std::unordered_map<int, double> ConditionalOccupancyState::get_state_marginal() const {
        std::unordered_map<int, double> marginal;
        for (const auto& [state, leader_histories] : conditional_distribution) {
            double state_prob = 0.0;
            for (const auto& [leader_history, prob] : leader_histories) {
                state_prob += prob;
            }
            if (state_prob > 0.0) {
                marginal[state] = state_prob;
            }
        }
        return marginal;
    }

    std::unordered_map<AgentHistory, double> ConditionalOccupancyState::get_leader_history_marginal() const {
        std::unordered_map<AgentHistory, double> marginal;
        for (const auto& [state, leader_histories] : conditional_distribution) {
            for (const auto& [leader_history, prob] : leader_histories) {
                if (prob > 0.0) {
                    marginal[leader_history] += prob;
                }
            }
        }
        return marginal;
    }

    double ConditionalOccupancyState::entropy() const {
        double entropy_val = 0.0;
        for (const auto& [state, leader_histories] : conditional_distribution) {
            for (const auto& [leader_history, prob] : leader_histories) {
                if (prob > 0.0) {
                    entropy_val -= prob * std::log2(prob);
                }
            }
        }
        return entropy_val;
    }

    double ConditionalOccupancyState::distance_to(const ConditionalOccupancyState& other) const {
        if (follower_history != other.follower_history) {
            throw std::invalid_argument("Cannot compute distance between conditional occupancy states with different follower histories");
        }
        
        double total_distance = 0.0;
        
        // Collect all unique (state, leader_history) tuples
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
        
        // Compute L1 distance
        for (const auto& [state, leader_history] : all_tuples) {
            double diff = std::abs(get_conditional_occupancy(state, leader_history) - 
                                 other.get_conditional_occupancy(state, leader_history));
            total_distance += diff;
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
            for (auto& [state, leader_histories] : conditional_distribution) {
                for (auto& [leader_history, prob] : leader_histories) {
                    prob /= sum;
                }
            }
        }
        is_normalized = true;
    }

    bool ConditionalOccupancyState::is_valid() const {
        double sum = 0.0;
        for (const auto& [state, leader_histories] : conditional_distribution) {
            for (const auto& [leader_history, prob] : leader_histories) {
                if (prob < 0.0) return false;
                sum += prob;
            }
        }
        return std::abs(sum - 1.0) < 1e-6;
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