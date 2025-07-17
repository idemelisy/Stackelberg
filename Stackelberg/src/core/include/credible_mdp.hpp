#pragma once

#include "credible_set.hpp"
#include "occupancy_state.hpp"
#include "conditional_occupancy_state.hpp"
#include "common.hpp"
#include <unordered_map>
#include <vector>
#include <functional>
#include "transition_model.hpp"
#include "observation_model.hpp"

namespace posg_core {

    // Forward declarations
    class Action;
    class Observation;
    class JointAction;
    class JointObservation;

    /**
     * @brief Represents a leader decision rule (policy component)
     */
    class LeaderDecisionRule {
    private:
        // Map from leader history to action probabilities
        std::unordered_map<AgentHistory, std::unordered_map<Action, double>> decision_rule;
        
        // Timestep this rule applies to
        int timestep;

    public:
        LeaderDecisionRule(int timestep);
        
        void set_action_probability(const AgentHistory& leader_history, 
                                  const Action& action, double probability);
        double get_action_probability(const AgentHistory& leader_history, 
                                    const Action& action) const;
        
        // Sample an action given a leader history
        Action sample_action(const AgentHistory& leader_history) const;
        
        // Get all possible actions and their probabilities for a history
        std::vector<std::pair<Action, double>> get_action_probabilities(
            const AgentHistory& leader_history) const;
        
        int get_timestep() const { return timestep; }
        bool is_valid() const;
        void normalize();
        
        std::string to_string() const;
    };

    /**
     * @brief Represents a follower decision rule (policy component)
     */
    class FollowerDecisionRule {
    private:
        // Map from follower history to action probabilities
        std::unordered_map<AgentHistory, std::unordered_map<Action, double>> decision_rule;
        
        // Timestep this rule applies to
        int timestep;

    public:
        FollowerDecisionRule(int timestep);
        
        void set_action_probability(const AgentHistory& follower_history, 
                                  const Action& action, double probability);
        double get_action_probability(const AgentHistory& follower_history, 
                                    const Action& action) const;
        
        // Sample an action given a follower history
        Action sample_action(const AgentHistory& follower_history) const;
        
        // Get all possible actions and their probabilities for a history
        std::vector<std::pair<Action, double>> get_action_probabilities(
            const AgentHistory& follower_history) const;
        
        int get_timestep() const { return timestep; }
        bool is_valid() const;
        void normalize();
        
        std::string to_string() const;
    };

    /**
     * @brief Represents the Credible Markov Decision Process (CMDP)
     * 
     * This is the core contribution of the AAAI 2026 paper. The CMDP is a lossless
     * reduction of the leader-follower POSG that enables dynamic programming methods.
     * The state space consists of credible sets, and the transition function embeds
     * rational follower responses into the state dynamics.
     */
    class CredibleMDP {
    private:
        // State space: set of all credible sets
        std::vector<CredibleSet> state_space;
        
        // Action space: set of all leader decision rules
        std::vector<LeaderDecisionRule> action_space;
        
        // Transition function: maps (credible_set, leader_decision_rule) to next credible_set
        std::unordered_map<CredibleSet, std::unordered_map<LeaderDecisionRule, CredibleSet, 
            CredibleSetHash>, CredibleSetHash> transition_function;
        
        // Reward function: maps credible_set to leader reward
        std::unordered_map<CredibleSet, double, CredibleSetHash> reward_function;
        
        // Initial credible set
        CredibleSet initial_credible_set;
        
        // Planning horizon
        int horizon;
        
        // Models for computing transitions
        TransitionModel transition_model;
        ObservationModel observation_model;

        std::vector<FollowerDecisionRule> generate_possible_follower_rules(const OccupancyState&, const LeaderDecisionRule&) const;

    public:
        // Constructors
        CredibleMDP();
        CredibleMDP(const TransitionModel& transition_model, 
                   const ObservationModel& observation_model,
                   const std::vector<double>& initial_belief,
                   int horizon);
        
        // Core CMDP operations
        CredibleSet transition(const CredibleSet& current_set, 
                             const LeaderDecisionRule& leader_rule) const;
        
        double get_reward(const CredibleSet& credible_set) const;
        void set_reward(const CredibleSet& credible_set, double reward);
        
        // State space management
        void add_credible_set(const CredibleSet& credible_set);
        const std::vector<CredibleSet>& get_state_space() const { return state_space; }
        
        // Action space management
        void add_leader_decision_rule(const LeaderDecisionRule& rule);
        const std::vector<LeaderDecisionRule>& get_action_space() const { return action_space; }
        
        // Value function computation (as per paper's Section 4)
        double compute_optimal_value(const CredibleSet& credible_set, int timestep) const;
        LeaderDecisionRule compute_optimal_policy(const CredibleSet& credible_set, int timestep) const;
        
        // PBVI support (as per paper's Section 5)
        std::vector<CredibleSet> sample_credible_sets(int num_samples) const;
        void expand_credible_set(const CredibleSet& credible_set, 
                               std::vector<CredibleSet>& expanded_sets) const;
        
        // Getters
        const CredibleSet& get_initial_credible_set() const { return initial_credible_set; }
        int get_horizon() const { return horizon; }
        const TransitionModel& get_transition_model() const { return transition_model; }
        const ObservationModel& get_observation_model() const { return observation_model; }
        
        // Validation
        bool is_valid() const;
        
        // Debug
        std::string to_string() const;
    };

} // namespace posg_core 