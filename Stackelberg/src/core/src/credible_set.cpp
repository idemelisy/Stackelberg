// credible_set.cpp
// ---------------
// Implements the CredibleSet class for the AAAI 2026 paper.
// What: Manages credible sets (collections of occupancy states)
// Why: This is the state space of the Credible MDP, representing reachable occupancy states under follower responses
// Fit: Used as the state space for CMDP value iteration and policy computation

#include "../include/credible_set.hpp"
#include "../include/common.hpp"
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace posg_core {

    CredibleSet::CredibleSet() : timestep(0) {}

    CredibleSet::CredibleSet(const std::vector<OccupancyState>& states) : timestep(0) {
        for (const auto& state : states) {
            add_occupancy_state(state);
        }
    }

    CredibleSet::CredibleSet(const OccupancyState& single_state) : timestep(0) {
        add_occupancy_state(single_state);
    }

    void CredibleSet::add_occupancy_state(const OccupancyState& occupancy_state) {
        occupancy_states.insert(occupancy_state);
        // Clear cache when adding new states
        conditional_occupancy_cache.clear();
    }

    void CredibleSet::remove_occupancy_state(const OccupancyState& occupancy_state) {
        occupancy_states.erase(occupancy_state);
        // Clear cache when removing states
        conditional_occupancy_cache.clear();
    }

    std::vector<ConditionalOccupancyState> CredibleSet::get_conditional_occupancy_states() const {
        std::vector<ConditionalOccupancyState> conditional_states;
        std::set<AgentHistory> all_follower_histories;
        
        // Collect all unique follower histories from all occupancy states
        for (const auto& occupancy_state : occupancy_states) {
            auto follower_marginal = occupancy_state.get_follower_history_marginal();
            for (const auto& [follower_history, _] : follower_marginal) {
                all_follower_histories.insert(follower_history);
            }
        }
        
        // Create conditional occupancy states for each follower history
        for (const auto& follower_history : all_follower_histories) {
            auto conditional_states_for_history = get_conditional_occupancy_states(follower_history);
            conditional_states.insert(conditional_states.end(), 
                                    conditional_states_for_history.begin(), 
                                    conditional_states_for_history.end());
        }
        
        return conditional_states;
    }

    std::vector<ConditionalOccupancyState> CredibleSet::get_conditional_occupancy_states(
        const AgentHistory& follower_history) const {
        
        // Check cache first
        auto cache_it = conditional_occupancy_cache.find(follower_history);
        if (cache_it != conditional_occupancy_cache.end()) {
            return cache_it->second;
        }
        
        std::vector<ConditionalOccupancyState> conditional_states;
        
        // For each occupancy state, extract the conditional distribution
        for (const auto& occupancy_state : occupancy_states) {
            // Get the marginal probability of this follower history
            auto follower_marginal = occupancy_state.get_follower_history_marginal();
            auto follower_prob_it = follower_marginal.find(follower_history);
            
            if (follower_prob_it != follower_marginal.end() && follower_prob_it->second > 0.0) {
                // Create conditional occupancy state for this follower history
                ConditionalOccupancyState conditional_state(follower_history);
                
                // Extract conditional probabilities: Pr(state, leader_history | follower_history)
                for (const auto& [state, leader_histories] : occupancy_state.get_occupancy_distribution()) {
                    for (const auto& [leader_history, follower_histories] : leader_histories) {
                        auto follower_it = follower_histories.find(follower_history);
                        if (follower_it != follower_histories.end() && follower_it->second > 0.0) {
                            // Pr(state, leader_history | follower_history) = Pr(state, leader_history, follower_history) / Pr(follower_history)
                            double conditional_prob = follower_it->second / follower_prob_it->second;
                            conditional_state.set_conditional_occupancy(state, leader_history, conditional_prob);
                        }
                    }
                }
                
                conditional_states.push_back(conditional_state);
            }
        }
        
        // Cache the result
        conditional_occupancy_cache[follower_history] = conditional_states;
        
        return conditional_states;
    }

    double CredibleSet::entropy() const {
        if (occupancy_states.empty()) return 0.0;
        
        double total_entropy = 0.0;
        for (const auto& occupancy_state : occupancy_states) {
            total_entropy += occupancy_state.entropy();
        }
        return total_entropy / occupancy_states.size();
    }

    double CredibleSet::distance_to(const CredibleSet& other) const {
        if (occupancy_states.empty() && other.occupancy_states.empty()) return 0.0;
        if (occupancy_states.empty()) return std::numeric_limits<double>::infinity();
        if (other.occupancy_states.empty()) return std::numeric_limits<double>::infinity();
        
        double total_distance = 0.0;
        size_t comparisons = 0;
        
        for (const auto& state1 : occupancy_states) {
            double min_distance = std::numeric_limits<double>::infinity();
            for (const auto& state2 : other.occupancy_states) {
                double distance = state1.distance_to(state2);
                min_distance = std::min(min_distance, distance);
            }
            total_distance += min_distance;
            comparisons++;
        }
        
        return comparisons > 0 ? total_distance / comparisons : 0.0;
    }

    double CredibleSet::hausdorff_distance(const CredibleSet& other) const {
        if (occupancy_states.empty() && other.occupancy_states.empty()) return 0.0;
        if (occupancy_states.empty() || other.occupancy_states.empty()) {
            return std::numeric_limits<double>::infinity();
        }
        
        // Compute sup_{x in this} inf_{y in other} d(x, y)
        double max_min_distance_this_to_other = 0.0;
        for (const auto& state_this : occupancy_states) {
            double min_distance = std::numeric_limits<double>::infinity();
            for (const auto& state_other : other.occupancy_states) {
                double distance = state_this.distance_to(state_other);
                min_distance = std::min(min_distance, distance);
            }
            max_min_distance_this_to_other = std::max(max_min_distance_this_to_other, min_distance);
        }
        
        // Compute sup_{y in other} inf_{x in this} d(x, y)
        double max_min_distance_other_to_this = 0.0;
        for (const auto& state_other : other.occupancy_states) {
            double min_distance = std::numeric_limits<double>::infinity();
            for (const auto& state_this : occupancy_states) {
                double distance = state_other.distance_to(state_this);
                min_distance = std::min(min_distance, distance);
            }
            max_min_distance_other_to_this = std::max(max_min_distance_other_to_this, min_distance);
        }
        
        // Hausdorff distance is the maximum of the two
        return std::max(max_min_distance_this_to_other, max_min_distance_other_to_this);
    }

    bool CredibleSet::contains(const OccupancyState& occupancy_state) const {
        return occupancy_states.find(occupancy_state) != occupancy_states.end();
    }

    std::string CredibleSet::to_string() const {
        std::ostringstream oss;
        oss << "CredibleSet(t=" << timestep << ", size=" << occupancy_states.size() << ", states={";
        
        bool first = true;
        for (const auto& occupancy_state : occupancy_states) {
            if (!first) oss << ", ";
            oss << occupancy_state.to_string();
            first = false;
        }
        oss << "})";
        return oss.str();
    }

    bool CredibleSet::operator==(const CredibleSet& other) const {
        if (timestep != other.timestep) return false;
        if (occupancy_states.size() != other.occupancy_states.size()) return false;
        
        // Check if all states in this set are in the other set
        for (const auto& state : occupancy_states) {
            if (other.occupancy_states.find(state) == other.occupancy_states.end()) {
                return false;
            }
        }
        
        return true;
    }

    bool CredibleSet::operator!=(const CredibleSet& other) const {
        return !(*this == other);
    }

    bool CredibleSet::operator<(const CredibleSet& other) const {
        if (timestep != other.timestep) return timestep < other.timestep;
        if (occupancy_states.size() != other.occupancy_states.size()) {
            return occupancy_states.size() < other.occupancy_states.size();
        }
        
        // Lexicographic comparison of occupancy states
        auto it1 = occupancy_states.begin();
        auto it2 = other.occupancy_states.begin();
        
        while (it1 != occupancy_states.end() && it2 != other.occupancy_states.end()) {
            if (*it1 != *it2) {
                return *it1 < *it2;
            }
            ++it1;
            ++it2;
        }
        
        return false; // Sets are equal
    }

    std::size_t CredibleSet::Hash::operator()(const CredibleSet& credible_set) const {
        std::size_t hash = 0;
        for (const auto& occupancy_state : credible_set.occupancy_states) {
            hash ^= std::hash<OccupancyState>{}(occupancy_state) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }

} // namespace posg_core 