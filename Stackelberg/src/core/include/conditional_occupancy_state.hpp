#pragma once

#include "occupancy_state.hpp"
#include <unordered_map>
#include <vector>

namespace posg_core {

    // Forward declarations
    class Action;
    class Observation;
    class JointAction;
    class JointObservation;
    class TransitionModel;
    class ObservationModel;

    /**
     * @brief Represents a conditional occupancy state - distribution over (state, leader_history) 
     * conditioned on follower history
     * 
     * This is a key concept from the AAAI 2026 paper. A conditional occupancy state represents
     * the leader's belief about the environment state and its own history, given a specific
     * follower history. This enables the decomposition used in the CMDP approach.
     */
    class ConditionalOccupancyState {
    private:
        // Map from (state, leader_history) to probability, conditioned on follower_history
        std::unordered_map<int, std::unordered_map<AgentHistory, double>> conditional_distribution;
        
        // The follower history this distribution is conditioned on
        AgentHistory follower_history;
        
        // Timestep tracking
        int timestep;
        
        // Cache for normalization
        mutable bool is_normalized;

    public:
        // Constructors
        ConditionalOccupancyState();
        ConditionalOccupancyState(const AgentHistory& follower_history);
        ConditionalOccupancyState(const AgentHistory& follower_history, 
                                const std::vector<double>& initial_belief);
        
        // Core conditional occupancy state operations
        ConditionalOccupancyState update(const Action& leader_action, const Action& follower_action,
                                        const Observation& leader_obs, const Observation& follower_obs,
                                        const TransitionModel& model, const ObservationModel& obs_model) const;
        
        // Conditional occupancy state access
        double get_conditional_occupancy(int state, const AgentHistory& leader_history) const;
        void set_conditional_occupancy(int state, const AgentHistory& leader_history, double probability);
        
        // Get marginal distributions
        std::unordered_map<int, double> get_state_marginal() const;
        std::unordered_map<AgentHistory, double> get_leader_history_marginal() const;
        
        // Get the follower history this is conditioned on
        const AgentHistory& get_follower_history() const { return follower_history; }
        
        // Information measures
        double entropy() const;
        double distance_to(const ConditionalOccupancyState& other) const;
        
        // Utility methods
        void normalize();
        bool is_valid() const;
        std::string to_string() const;
        
        // Comparison operators
        bool operator==(const ConditionalOccupancyState& other) const;
        bool operator!=(const ConditionalOccupancyState& other) const;
        
        // Hash function for unordered_map
        struct Hash {
            std::size_t operator()(const ConditionalOccupancyState& conditional_occupancy) const;
        };
    };

    using ConditionalOccupancyStateHash = ConditionalOccupancyState::Hash;

} // namespace posg_core 