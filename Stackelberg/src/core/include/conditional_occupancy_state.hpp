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
    class LeaderDecisionRule;

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
        
        /**
         * @brief Compute τzF(c, σL, aF) from Lemma 4.2
         * 
         * The next conditional occupancy state upon receiving follower observation zF 
         * after taking leader-follower decisions (σL, aF) starting in conditional occupancy state c.
         * 
         * From Paper: Lemma 4.2 - c'(s', h'L) ∝ Σ_s c(s, hL)σL(aL|hL)p(s', zL, zF|s, aL, aF)
         * 
         * @param sigma_L Leader decision rule
         * @param a_F Follower action
         * @param z_F Follower observation
         * @param transition_model Transition model
         * @param observation_model Observation model
         * @return Updated conditional occupancy state
         */
        ConditionalOccupancyState tau_zF(const LeaderDecisionRule& sigma_L, const Action& a_F,
                                        const Observation& z_F, const TransitionModel& transition_model,
                                        const ObservationModel& observation_model) const;
        
        /**
         * @brief Compute weighted conditional occupancy state τ̃zF(c, σL, aF)
         * 
         * From Paper: τ̃zF(c, σL, aF) = ηzF(c, σL, aF) · τzF(c, σL, aF)
         * 
         * @param sigma_L Leader decision rule
         * @param a_F Follower action
         * @param z_F Follower observation
         * @param transition_model Transition model
         * @param observation_model Observation model
         * @return Weighted conditional occupancy state
         */
        ConditionalOccupancyState tau_tilde_zF(const LeaderDecisionRule& sigma_L, const Action& a_F,
                                              const Observation& z_F, const TransitionModel& transition_model,
                                              const ObservationModel& observation_model) const;
        
        /**
         * @brief Compute normalization constant ηzF(c, σL, aF)
         * 
         * From Paper: ηzF(c, σL, aF) = Σ_s,s',hL,aL p(s', zL, zF|s, aL, aF) · σL(aL|hL) · c(s, hL)
         * 
         * @param sigma_L Leader decision rule
         * @param a_F Follower action
         * @param z_F Follower observation
         * @param transition_model Transition model
         * @param observation_model Observation model
         * @return Normalization constant
         */
        double eta_zF(const LeaderDecisionRule& sigma_L, const Action& a_F,
                     const Observation& z_F, const TransitionModel& transition_model,
                     const ObservationModel& observation_model) const;
        
        /**
         * @brief Compute immediate expected payoff ρi(c, σL, aF)
         * 
         * From Paper: ρi(c, σL, aF) = E[ri(s, aL, aF)|c, σL, aF]
         * 
         * @param player_id Player ID (0 for leader, 1 for follower)
         * @param sigma_L Leader decision rule
         * @param a_F Follower action
         * @param reward_function Reward function for the player
         * @return Immediate expected payoff
         */
        double rho_i(int player_id, const LeaderDecisionRule& sigma_L, const Action& a_F,
                    const std::function<double(int, const Action&, const Action&)>& reward_function) const;

        // Conditional occupancy state access
        double get_conditional_occupancy(int state, const AgentHistory& leader_history) const;
        void set_conditional_occupancy(int state, const AgentHistory& leader_history, double probability);
        
        // Get marginal distributions
        std::unordered_map<int, double> get_state_marginal() const;
        std::unordered_map<AgentHistory, double> get_leader_history_marginal() const;
        
        // Get the follower history this is conditioned on
        const AgentHistory& get_follower_history() const { return follower_history; }
        
        // Set timestep (needed for conditional decomposition)
        void set_timestep(int t) { timestep = t; }
        
        // Information measures
        double entropy() const;
        double distance_to(const ConditionalOccupancyState& other) const;
        
        // Utility methods
        void normalize();
        bool is_valid(bool allow_empty = true) const;
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