// occupancy_state.cpp
// ------------------
// Implements the OccupancyState and AgentHistory classes for the AAAI 2026 paper.
// What: Manages occupancy states (distributions over state, leader_history, follower_history)
// Why: This is the core concept that enables the CMDP approach in leader-follower POSGs
// Fit: Used throughout the CMDP solution for value iteration, policy selection, and simulation.

#include "../include/occupancy_state.hpp"
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

    // ============================================================================
    // AgentHistory Implementation
    // ============================================================================

    AgentHistory::AgentHistory(int agent_id) : agent_id(agent_id) {}

    void AgentHistory::add_action(const Action& action) {
        if (action.get_agent_id() != agent_id) {
            throw std::invalid_argument("Action agent ID doesn't match history agent ID");
        }
        actions.push_back(action);
    }

    void AgentHistory::add_observation(const Observation& observation) {
        if (observation.get_agent_id() != agent_id) {
            throw std::invalid_argument("Observation agent ID doesn't match history agent ID");
        }
        observations.push_back(observation);
    }

    std::string AgentHistory::to_string() const {
        std::ostringstream oss;
        oss << "AgentHistory(agent=" << agent_id << ", actions=[";
        for (size_t i = 0; i < actions.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << actions[i].to_string();
        }
        oss << "], observations=[";
        for (size_t i = 0; i < observations.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << observations[i].to_string();
        }
        oss << "])";
        return oss.str();
    }

    bool AgentHistory::operator==(const AgentHistory& other) const {
        return agent_id == other.agent_id && 
               actions == other.actions && 
               observations == other.observations;
    }

    bool AgentHistory::operator<(const AgentHistory& other) const {
        if (agent_id != other.agent_id) return agent_id < other.agent_id;
        if (actions != other.actions) return actions < other.actions;
        return observations < other.observations;
    }

    bool AgentHistory::operator!=(const AgentHistory& other) const {
        return !(*this == other);
    }

    // ============================================================================
    // OccupancyState Implementation
    // ============================================================================

    OccupancyState::OccupancyState() : timestep(0), is_normalized(true) {}

    OccupancyState::OccupancyState(const std::vector<double>& initial_belief) 
        : timestep(0), is_normalized(false) {
        // Create empty histories for initial state
        AgentHistory empty_leader_history(0);
        AgentHistory empty_follower_history(1);
        
        // Set initial occupancy: Pr(state, empty_leader_history, empty_follower_history) = initial_belief[state]
        for (size_t state = 0; state < initial_belief.size(); ++state) {
            if (initial_belief[state] > 0.0) {
                set_occupancy(state, empty_leader_history, empty_follower_history, initial_belief[state]);
            }
        }
        normalize();
    }

    void OccupancyState::update(const Action& leader_action, const Action& follower_action,
                               const Observation& leader_obs, const Observation& follower_obs,
                               const TransitionModel& transition_model, 
                               const ObservationModel& observation_model) {
        std::unordered_map<int, std::unordered_map<AgentHistory, 
            std::unordered_map<AgentHistory, double>>> new_occupancy;
        
        double normalization_constant = 0.0;
        
        // For each possible next state
        for (int next_state = 0; next_state < transition_model.get_num_states(); ++next_state) {
            // For each current occupancy state
            for (const auto& [current_state, leader_histories] : occupancy_distribution) {
                for (const auto& [leader_history, follower_histories] : leader_histories) {
                    for (const auto& [follower_history, current_prob] : follower_histories) {
                        if (current_prob <= 0.0) continue;
                        
                        // Create joint action and observation
                        JointAction joint_action(leader_action, follower_action);
                        JointObservation joint_obs(leader_obs, follower_obs);
                        
                        // Compute transition and observation probabilities
                        double transition_prob = transition_model.get_transition_probability(
                            current_state, joint_action, next_state);
                        double obs_prob = observation_model.get_observation_probability(
                            next_state, joint_action, joint_obs);
                        
                        // Create new histories
                        AgentHistory new_leader_history = leader_history;
                        new_leader_history.add_action(leader_action);
                        new_leader_history.add_observation(leader_obs);
                        
                        AgentHistory new_follower_history = follower_history;
                        new_follower_history.add_action(follower_action);
                        new_follower_history.add_observation(follower_obs);
                        
                        // Update occupancy probability
                        double new_prob = current_prob * transition_prob * obs_prob;
                        new_occupancy[next_state][new_leader_history][new_follower_history] += new_prob;
                        normalization_constant += new_prob;
                    }
                }
            }
        }
        
        // Normalize and update
        if (normalization_constant > 0.0) {
            for (auto& [state, leader_histories] : new_occupancy) {
                for (auto& [leader_history, follower_histories] : leader_histories) {
                    for (auto& [follower_history, prob] : follower_histories) {
                        prob /= normalization_constant;
                    }
                }
            }
        }
        
        occupancy_distribution = new_occupancy;
        timestep++;
        is_normalized = true;
    }

    double OccupancyState::get_occupancy(int state, const AgentHistory& leader_history, 
                                        const AgentHistory& follower_history) const {
        auto state_it = occupancy_distribution.find(state);
        if (state_it == occupancy_distribution.end()) return 0.0;
        
        auto leader_it = state_it->second.find(leader_history);
        if (leader_it == state_it->second.end()) return 0.0;
        
        auto follower_it = leader_it->second.find(follower_history);
        return (follower_it != leader_it->second.end()) ? follower_it->second : 0.0;
    }

    void OccupancyState::set_occupancy(int state, const AgentHistory& leader_history,
                                      const AgentHistory& follower_history, double probability) {
        if (probability > 0.0) {
            occupancy_distribution[state][leader_history][follower_history] = probability;
        } else {
            auto state_it = occupancy_distribution.find(state);
            if (state_it != occupancy_distribution.end()) {
                auto leader_it = state_it->second.find(leader_history);
                if (leader_it != state_it->second.end()) {
                    leader_it->second.erase(follower_history);
                    if (leader_it->second.empty()) {
                        state_it->second.erase(leader_history);
                    }
                }
                if (state_it->second.empty()) {
                    occupancy_distribution.erase(state);
                }
            }
        }
        is_normalized = false;
    }

    /**
     * @brief Add or update an entry in the occupancy distribution.
     *
     * If the entry already exists, its probability is overwritten. If the probability is negative,
     * throws std::invalid_argument. If probability is zero, the entry is removed if present.
     *
     * @param state The environment state index
     * @param leader_history The leader's AgentHistory
     * @param follower_history The follower's AgentHistory
     * @param probability The probability to assign (must be >= 0)
     */
    void OccupancyState::add_entry(int state, const AgentHistory& leader_history, const AgentHistory& follower_history, double probability) {
        if (probability < 0.0) {
            throw std::invalid_argument("OccupancyState::add_entry: Probability cannot be negative");
        }
        if (probability == 0.0) {
            // Remove the entry if it exists
            auto state_it = occupancy_distribution.find(state);
            if (state_it != occupancy_distribution.end()) {
                auto& leader_map = state_it->second;
                auto leader_it = leader_map.find(leader_history);
                if (leader_it != leader_map.end()) {
                    auto& follower_map = leader_it->second;
                    follower_map.erase(follower_history);
                    if (follower_map.empty()) {
                        leader_map.erase(leader_it);
                    }
                }
                if (leader_map.empty()) {
                    occupancy_distribution.erase(state_it);
                }
            }
        } else {
            // Add or update the entry
            occupancy_distribution[state][leader_history][follower_history] = probability;
        }
        is_normalized = false;
    }

    std::unordered_map<int, double> OccupancyState::get_state_marginal() const {
        std::unordered_map<int, double> marginal;
        double sum = 0.0;
        for (const auto& [state, leader_histories] : occupancy_distribution) {
            double state_prob = 0.0;
            for (const auto& [leader_history, follower_histories] : leader_histories) {
                for (const auto& [follower_history, prob] : follower_histories) {
                    state_prob += prob;
                }
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

    std::unordered_map<AgentHistory, double> OccupancyState::get_leader_history_marginal() const {
        std::unordered_map<AgentHistory, double> marginal;
        double sum = 0.0;
        for (const auto& [state, leader_histories] : occupancy_distribution) {
            for (const auto& [leader_history, follower_histories] : leader_histories) {
                double history_prob = 0.0;
                for (const auto& [follower_history, prob] : follower_histories) {
                    history_prob += prob;
                }
                if (history_prob > 0.0) {
                    marginal[leader_history] += history_prob;
                    sum += history_prob;
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

    std::unordered_map<AgentHistory, double> OccupancyState::get_follower_history_marginal() const {
        std::unordered_map<AgentHistory, double> marginal;
        double sum = 0.0;
        for (const auto& [state, leader_histories] : occupancy_distribution) {
            for (const auto& [leader_history, follower_histories] : leader_histories) {
                for (const auto& [follower_history, prob] : follower_histories) {
                    if (prob > 0.0) {
                        marginal[follower_history] += prob;
                        sum += prob;
                    }
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

    double OccupancyState::entropy() const {
        double entropy_val = 0.0;
        for (const auto& [state, leader_histories] : occupancy_distribution) {
            for (const auto& [leader_history, follower_histories] : leader_histories) {
                for (const auto& [follower_history, prob] : follower_histories) {
                    if (prob > 0.0) {
                        entropy_val -= prob * std::log(prob); // Use natural log
                    }
                }
            }
        }
        return entropy_val;
    }

    double OccupancyState::distance_to(const OccupancyState& other) const {
        double total_distance = 0.0;
        
        // Collect all unique (state, leader_history, follower_history) tuples
        std::set<std::tuple<int, AgentHistory, AgentHistory>> all_tuples;
        
        for (const auto& [state, leader_histories] : occupancy_distribution) {
            for (const auto& [leader_history, follower_histories] : leader_histories) {
                for (const auto& [follower_history, _] : follower_histories) {
                    all_tuples.insert({state, leader_history, follower_history});
                }
            }
        }
        
        for (const auto& [state, leader_histories] : other.occupancy_distribution) {
            for (const auto& [leader_history, follower_histories] : leader_histories) {
                for (const auto& [follower_history, _] : follower_histories) {
                    all_tuples.insert({state, leader_history, follower_history});
                }
            }
        }
        
        // Compute L1 distance
        for (const auto& [state, leader_history, follower_history] : all_tuples) {
            double diff = std::abs(get_occupancy(state, leader_history, follower_history) - 
                                 other.get_occupancy(state, leader_history, follower_history));
            total_distance += diff;
        }
        
        return total_distance;
    }

    void OccupancyState::normalize() {
        double sum = 0.0;
        for (const auto& [state, leader_histories] : occupancy_distribution) {
            for (const auto& [leader_history, follower_histories] : leader_histories) {
                for (const auto& [follower_history, prob] : follower_histories) {
                    sum += prob;
                }
            }
        }
        if (sum > 0.0) {
            for (auto state_it = occupancy_distribution.begin(); state_it != occupancy_distribution.end(); ) {
                auto& leader_histories = state_it->second;
                for (auto leader_it = leader_histories.begin(); leader_it != leader_histories.end(); ) {
                    auto& follower_histories = leader_it->second;
                    for (auto follower_it = follower_histories.begin(); follower_it != follower_histories.end(); ) {
                        follower_it->second /= sum;
                        if (follower_it->second == 0.0) {
                            follower_it = follower_histories.erase(follower_it);
                        } else {
                            ++follower_it;
                        }
                    }
                    if (follower_histories.empty()) {
                        leader_it = leader_histories.erase(leader_it);
                    } else {
                        ++leader_it;
                    }
                }
                if (leader_histories.empty()) {
                    state_it = occupancy_distribution.erase(state_it);
                } else {
                    ++state_it;
                }
            }
        } else {
            occupancy_distribution.clear();
        }
        is_normalized = true;
    }

    bool OccupancyState::is_valid(bool allow_empty) const {
        if (!is_normalized) {
            const_cast<OccupancyState*>(this)->normalize();
        }
        if (occupancy_distribution.empty()) return allow_empty;
        double sum = 0.0;
        for (const auto& [state, leader_histories] : occupancy_distribution) {
            for (const auto& [leader_history, follower_histories] : leader_histories) {
                for (const auto& [follower_history, prob] : follower_histories) {
                    if (prob < 0.0) return false;
                    sum += prob;
                }
            }
        }
        return std::abs(sum - 1.0) < 1e-6;
    }

    bool OccupancyState::is_valid() const {
        return is_valid(true);
    }

    std::string OccupancyState::to_string() const {
        std::ostringstream oss;
        oss << "OccupancyState(t=" << timestep << ", occupancy={";
        
        bool first = true;
        for (const auto& [state, leader_histories] : occupancy_distribution) {
            for (const auto& [leader_history, follower_histories] : leader_histories) {
                for (const auto& [follower_history, prob] : follower_histories) {
                    if (!first) oss << ", ";
                    oss << "(" << state << ", " << leader_history.to_string() << ", " 
                        << follower_history.to_string() << "):" << std::fixed 
                        << std::setprecision(3) << prob;
                    first = false;
                }
            }
        }
        oss << "})";
        return oss.str();
    }

    bool OccupancyState::operator==(const OccupancyState& other) const {
        if (timestep != other.timestep) return false;
        return distance_to(other) < 1e-6;
    }

    bool OccupancyState::operator!=(const OccupancyState& other) const {
        return !(*this == other);
    }

    bool OccupancyState::operator<(const OccupancyState& other) const {
        if (timestep != other.timestep) return timestep < other.timestep;
        // Flatten to sorted vector of tuples for comparison
        using Tuple = std::tuple<int, AgentHistory, AgentHistory, double>;
        std::vector<Tuple> this_vec, other_vec;
        for (const auto& [state, leader_histories] : occupancy_distribution) {
            for (const auto& [leader_history, follower_histories] : leader_histories) {
                for (const auto& [follower_history, prob] : follower_histories) {
                    this_vec.emplace_back(state, leader_history, follower_history, prob);
                }
            }
        }
        for (const auto& [state, leader_histories] : other.occupancy_distribution) {
            for (const auto& [leader_history, follower_histories] : leader_histories) {
                for (const auto& [follower_history, prob] : follower_histories) {
                    other_vec.emplace_back(state, leader_history, follower_history, prob);
                }
            }
        }
        std::sort(this_vec.begin(), this_vec.end());
        std::sort(other_vec.begin(), other_vec.end());
        return this_vec < other_vec;
    }

    std::size_t OccupancyState::Hash::operator()(const OccupancyState& occupancy) const {
        std::size_t hash = 0;
        for (const auto& [state, leader_histories] : occupancy.occupancy_distribution) {
            for (const auto& [leader_history, follower_histories] : leader_histories) {
                for (const auto& [follower_history, prob] : follower_histories) {
                    hash ^= std::hash<int>{}(state) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                    hash ^= std::hash<AgentHistory>{}(leader_history) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                    hash ^= std::hash<AgentHistory>{}(follower_history) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                    hash ^= std::hash<double>{}(prob) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
            }
        }
        return hash;
    }

    ConditionalOccupancyState OccupancyState::conditional_decompose(const AgentHistory& follower_history) const {
        /**
         * From Paper: o = Σ_hF Pr(hF | o) · c(o,hF) ⊗ e_hF
         * where c(o,hF) is the conditional occupancy state given follower history hF
         * 
         * We compute c(o,hF) by extracting Pr(state, leader_history | follower_history)
         */
        
        ConditionalOccupancyState conditional_state(follower_history);
        conditional_state.set_timestep(timestep);
        
        // Get marginal probability of this follower history
        auto follower_marginal = get_follower_history_marginal();
        auto follower_prob_it = follower_marginal.find(follower_history);
        
        if (follower_prob_it == follower_marginal.end() || follower_prob_it->second <= 0.0) {
            // No probability mass for this follower history, return empty conditional state
            return conditional_state;
        }
        
        double follower_prob = follower_prob_it->second;
        
        // Extract conditional probabilities: Pr(state, leader_history | follower_history)
        for (const auto& [state, leader_histories] : occupancy_distribution) {
            for (const auto& [leader_history, follower_histories] : leader_histories) {
                auto follower_it = follower_histories.find(follower_history);
                if (follower_it != follower_histories.end() && follower_it->second > 0.0) {
                    // Pr(state, leader_history | follower_history) = Pr(state, leader_history, follower_history) / Pr(follower_history)
                    double conditional_prob = follower_it->second / follower_prob;
                    conditional_state.set_conditional_occupancy(state, leader_history, conditional_prob);
                }
            }
        }
        
        conditional_state.normalize();
        return conditional_state;
    }

} // namespace posg_core 