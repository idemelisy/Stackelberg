#pragma once

#include "common.hpp"
#include <unordered_map>
#include <vector>

namespace posg_core {

    /**
     * @brief Represents the observation model of the environment
     * 
     * Defines how observations are generated based on the true state and actions.
     * Each agent has its own observation model.
     */
    class ObservationModel {
    private:
        // Observation probabilities: O[state][joint_action][joint_observation] = probability
        std::vector<std::vector<std::vector<double>>> observation_probabilities;
        
        // Number of states, actions, observations per agent
        int num_states;
        int num_leader_actions;
        int num_follower_actions;
        int num_leader_observations;
        int num_follower_observations;

    public:
        // Constructors
        ObservationModel();
        ObservationModel(int num_states, int num_leader_actions, int num_follower_actions,
                        int num_leader_observations, int num_follower_observations);
        
        // Set observation probabilities
        void set_observation_probability(int state, const JointAction& joint_action,
                                       const JointObservation& joint_observation, double probability);
        
        // Get observation probabilities
        double get_observation_probability(int state, const JointAction& joint_action,
                                         const JointObservation& joint_observation) const;
        
        // Sample observations
        JointObservation sample_joint_observation(int state, const JointAction& joint_action) const;
        Observation sample_leader_observation(int state, const Action& leader_action) const;
        Observation sample_follower_observation(int state, const Action& follower_action) const;
        
        // Get observation probabilities for individual agents
        double get_leader_observation_probability(int state, const Action& leader_action,
                                                const Observation& leader_obs) const;
        double get_follower_observation_probability(int state, const Action& follower_action,
                                                  const Observation& follower_obs) const;
        
        // Get all possible observations and their probabilities
        std::vector<std::pair<JointObservation, double>> get_observations_and_probabilities(
            int state, const JointAction& joint_action) const;
        
        // Validation
        bool is_valid() const;
        void normalize();
        
        // Getters
        int get_num_states() const { return num_states; }
        int get_num_leader_actions() const { return num_leader_actions; }
        int get_num_follower_actions() const { return num_follower_actions; }
        int get_num_leader_observations() const { return num_leader_observations; }
        int get_num_follower_observations() const { return num_follower_observations; }
        
        // Debug
        void print() const;

        std::vector<Observation> get_observations_for_agent(int agent_id) const;
        std::vector<JointObservation> get_joint_observations() const;
    };

} // namespace posg_core 