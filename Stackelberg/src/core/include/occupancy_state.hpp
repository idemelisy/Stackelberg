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
    class ConditionalOccupancyState;

    /**
     * @brief Represents an occupancy state - distribution over (state, leader_history, follower_history)
     * 
     * From Paper: Definition 1 - Occupancy State
     * An occupancy state μ is a distribution over (s, hL, hF) where:
     * - s ∈ S is the environment state
     * - hL ∈ HL is the leader's history of actions and observations
     * - hF ∈ HF is the follower's history of actions and observations
     * 
     * This represents the leader's belief about the joint state of the environment 
     * and both agents' information states, enabling strategic reasoning over the
     * follower's rationality and information asymmetry.
     */
    class OccupancyState {
    private:
        // Map from (state, leader_history, follower_history) to probability
        // μ(s, hL, hF) = Pr(s, hL, hF | information available to leader)
        std::unordered_map<int, std::unordered_map<AgentHistory, 
            std::unordered_map<AgentHistory, double>>> occupancy_distribution;
        
        // Timestep tracking for temporal consistency
        int timestep;
        
        // Cache for normalization status
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
         * From Paper: μ(s, hL, hF) = probability
         * 
         * @param state The environment state s ∈ S
         * @param leader_history The leader's history hL ∈ HL
         * @param follower_history The follower's history hF ∈ HF
         * @param probability The probability μ(s, hL, hF) ≥ 0
         */
        void add_entry(int state, const AgentHistory& leader_history, const AgentHistory& follower_history, double probability);
        
        /**
         * @brief Decompose occupancy state into conditional occupancy state given follower history
         * 
         * From Paper: Lemma 4.1 - Conditional Decomposition
         * μ = Σ_hF Pr(hF | μ) · c(μ, hF) ⊗ e_hF
         * where c(μ, hF) is the conditional occupancy state given follower history hF
         * 
         * @param follower_history The follower history hF to condition on
         * @return Conditional occupancy state c(μ, hF)
         */
        ConditionalOccupancyState conditional_decompose(const AgentHistory& follower_history) const;

        // Marginal distributions
        /**
         * @brief Get state marginal distribution
         * 
         * From Paper: μ(s) = Σ_hL,hF μ(s, hL, hF)
         */
        std::unordered_map<int, double> get_state_marginal() const;
        
        /**
         * @brief Get leader history marginal distribution
         * 
         * From Paper: μ(hL) = Σ_s,hF μ(s, hL, hF)
         */
        std::unordered_map<AgentHistory, double> get_leader_history_marginal() const;
        
        /**
         * @brief Get follower history marginal distribution
         * 
         * From Paper: μ(hF) = Σ_s,hL μ(s, hL, hF)
         */
        std::unordered_map<AgentHistory, double> get_follower_history_marginal() const;
        
        // Access internal distribution (for CredibleSet use)
        const std::unordered_map<int, std::unordered_map<AgentHistory, 
            std::unordered_map<AgentHistory, double>>>& get_occupancy_distribution() const {
            return occupancy_distribution;
        }
        
        // Accessor for timestep (needed for hashing and external logic)
        int get_timestep() const { return timestep; }

        // Information measures
        /**
         * @brief Compute entropy of the occupancy state
         * 
         * From Paper: H(μ) = -Σ_s,hL,hF μ(s, hL, hF) log μ(s, hL, hF)
         */
        double entropy() const;
        
        /**
         * @brief Compute ℓ1-distance to another occupancy state
         * 
         * From Paper: ||μ - μ'||₁ = Σ_s,hL,hF |μ(s, hL, hF) - μ'(s, hL, hF)|
         */
        double distance_to(const OccupancyState& other) const;
        
        // Utility methods
        void normalize();
        bool is_valid(bool allow_empty = true) const;
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