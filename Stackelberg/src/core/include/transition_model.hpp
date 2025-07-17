#pragma once

#include "common.hpp"
#include <unordered_map>
#include <vector>

namespace posg_core {

    /**
     * @brief Represents the transition model of the environment
     * 
     * Defines how the state changes based on joint actions taken by the agents.
     */
    class TransitionModel {
    private:
        // Transition probabilities: T[state][joint_action][next_state] = probability
        std::vector<std::vector<std::vector<double>>> transition_probabilities;
        
        // Number of states, actions per agent
        int num_states;
        int num_leader_actions;
        int num_follower_actions;

    public:
        // Constructors
        TransitionModel();
        TransitionModel(int num_states, int num_leader_actions, int num_follower_actions);
        
        // Set transition probabilities
        void set_transition_probability(int current_state, const JointAction& joint_action, 
                                      int next_state, double probability);
        
        // Get transition probabilities
        double get_transition_probability(int current_state, const JointAction& joint_action, 
                                        int next_state) const;
        
        // Sample next state
        int sample_next_state(int current_state, const JointAction& joint_action) const;
        
        // Get all possible next states and their probabilities
        std::vector<std::pair<int, double>> get_next_states_and_probabilities(
            int current_state, const JointAction& joint_action) const;
        
        // Validation
        bool is_valid() const;
        void normalize();
        
        // Getters
        int get_num_states() const { return num_states; }
        int get_num_leader_actions() const { return num_leader_actions; }
        int get_num_follower_actions() const { return num_follower_actions; }
        
        // Debug
        void print() const;

        std::vector<int> get_states() const;
        std::vector<JointAction> get_joint_actions() const;
    };

} // namespace posg_core 