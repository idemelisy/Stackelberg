#pragma once

#include "common.hpp"
#include <unordered_map>
#include <vector>
#include <set>
#include <string>

namespace posg_core {

    // Forward declarations
    class Action;
    class Observation;
    class JointAction;
    class JointObservation;
    class TransitionModel;
    class ObservationModel;

    /**
     * @brief Represents a history of actions and observations for an agent
     */
    class AgentHistory {
    private:
        std::vector<Action> actions;
        std::vector<Observation> observations;
        int agent_id;

    public:
        AgentHistory(int agent_id);
        
        void add_action(const Action& action);
        void add_observation(const Observation& observation);
        
        const std::vector<Action>& get_actions() const { return actions; }
        const std::vector<Observation>& get_observations() const { return observations; }
        int get_agent_id() const { return agent_id; }
        size_t length() const { return actions.size(); }
        
        std::string to_string() const;
        bool operator==(const AgentHistory& other) const;
        bool operator!=(const AgentHistory& other) const;
        bool operator<(const AgentHistory& other) const;
    };

} // namespace posg_core

// Hash specialization for AgentHistory
namespace std {
    template<>
    struct hash<posg_core::AgentHistory> {
        std::size_t operator()(const posg_core::AgentHistory& h) const {
            std::size_t hash = std::hash<int>{}(h.get_agent_id());
            for (const auto& a : h.get_actions()) {
                hash ^= std::hash<int>{}(a.get_action_id()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            }
            for (const auto& o : h.get_observations()) {
                hash ^= std::hash<int>{}(o.get_observation_id()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            }
            return hash;
        }
    };
}

namespace posg_core {

    /**
     * @brief Represents an occupancy state - distribution over (state, leader_history, follower_history)
     * 
     * This is the core concept from the AAAI 2026 paper. An occupancy state represents
     * the leader's belief about the joint state of the environment and both agents' histories.
     * This is fundamentally different from POMDP belief states which only track state distributions.
     */
    class OccupancyState {
    private:
        // Map from (state, leader_history, follower_history) to probability
        std::unordered_map<int, std::unordered_map<AgentHistory, 
            std::unordered_map<AgentHistory, double>>> occupancy_distribution;
        
        // Timestep tracking
        int timestep;
        
        // Cache for normalization
        mutable bool is_normalized;

    public:
        // Constructors
        OccupancyState();
        OccupancyState(const std::vector<double>& initial_belief);
        
        // Core occupancy state operations
        void update(const Action& leader_action, const Action& follower_action,
                   const Observation& leader_obs, const Observation& follower_obs,
                   const TransitionModel& model, const ObservationModel& obs_model);
        
        // Occupancy state access
        double get_occupancy(int state, const AgentHistory& leader_history, 
                           const AgentHistory& follower_history) const;
        void set_occupancy(int state, const AgentHistory& leader_history,
                          const AgentHistory& follower_history, double probability);
        
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
        void add_entry(int state, const AgentHistory& leader_history, const AgentHistory& follower_history, double probability);

        // Get marginal distributions
        std::unordered_map<int, double> get_state_marginal() const;
        std::unordered_map<AgentHistory, double> get_leader_history_marginal() const;
        std::unordered_map<AgentHistory, double> get_follower_history_marginal() const;
        
        // Access internal distribution (for CredibleSet use)
        const std::unordered_map<int, std::unordered_map<AgentHistory, 
            std::unordered_map<AgentHistory, double>>>& get_occupancy_distribution() const {
            return occupancy_distribution;
        }
        
        // Accessor for timestep (needed for hashing and external logic)
        int get_timestep() const { return timestep; }

        // Information measures
        double entropy() const;
        double distance_to(const OccupancyState& other) const;
        
        // Utility methods
        void normalize();
        bool is_valid() const;
        std::string to_string() const;
        
        // Comparison operators
        bool operator==(const OccupancyState& other) const;
        bool operator!=(const OccupancyState& other) const;
        bool operator<(const OccupancyState& other) const;
        
        // Hash function for unordered_map
        struct Hash {
            std::size_t operator()(const OccupancyState& occupancy) const;
        };
    };

    using OccupancyStateHash = OccupancyState::Hash;

} // namespace posg_core 

// Hash specialization for OccupancyState (must be after full class definition)
#include <functional>
namespace std {
    /**
     * @brief Hash specialization for posg_core::OccupancyState
     *
     * Hashes the occupancy distribution and timestep for use in unordered_map/set.
     * Combines the hashes of all (state, leader_history, follower_history, probability) entries.
     */
    template<>
    struct hash<posg_core::OccupancyState> {
        std::size_t operator()(const posg_core::OccupancyState& o) const noexcept {
            std::size_t hash_val = std::hash<int>{}(o.get_timestep());
            const auto& dist = o.get_occupancy_distribution();
            for (const auto& [state, leader_map] : dist) {
                std::size_t state_hash = std::hash<int>{}(state);
                for (const auto& [leader_hist, follower_map] : leader_map) {
                    std::size_t leader_hash = std::hash<posg_core::AgentHistory>{}(leader_hist);
                    for (const auto& [follower_hist, prob] : follower_map) {
                        std::size_t follower_hash = std::hash<posg_core::AgentHistory>{}(follower_hist);
                        std::size_t prob_hash = std::hash<double>{}(prob);
                        // Combine all
                        hash_val ^= state_hash ^ (leader_hash << 1) ^ (follower_hash << 2) ^ (prob_hash << 3);
                    }
                }
            }
            return hash_val;
        }
    };
} 