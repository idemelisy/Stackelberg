// observation_model.cpp
// --------------------
// Implements the ObservationModel class for POSG.
// What: Manages the observation probabilities for each state and joint action.
// Why: In POSGs, agents do not observe the true state directly; they receive noisy observations. The observation model is essential for belief updates and value iteration.
// Fit: Used in belief updates, simulation, and value function backups throughout the POSG solution.

#include "../include/observation_model.hpp"
#include <stdexcept>
#include <sstream>
#include <random>

namespace posg_core {

    // Default constructor: creates an empty observation model.
    ObservationModel::ObservationModel() 
        : num_states(0), num_leader_actions(0), num_follower_actions(0),
          num_leader_observations(0), num_follower_observations(0) {}
    
    // Constructor: initializes the observation model with the given sizes.
    // What: Allocates space for observation probabilities for all state/action/observation tuples.
    // Why: Needed before setting or querying any observation probabilities.
    // Fit: Called after parsing the problem structure.
    ObservationModel::ObservationModel(int num_states, int num_leader_actions, int num_follower_actions,
                                     int num_leader_observations, int num_follower_observations)
        : num_states(num_states), num_leader_actions(num_leader_actions), num_follower_actions(num_follower_actions),
          num_leader_observations(num_leader_observations), num_follower_observations(num_follower_observations) {
        // Initialize observation probabilities to zero.
        observation_probabilities.resize(num_states);
        for (int s = 0; s < num_states; ++s) {
            observation_probabilities[s].resize(num_leader_actions * num_follower_actions);
            for (int a = 0; a < num_leader_actions * num_follower_actions; ++a) {
                observation_probabilities[s][a].resize(num_leader_observations * num_follower_observations, 0.0);
            }
        }
    }

    // Set the observation probability for a given (state, joint_action, joint_observation).
    // What: Stores the probability of observing joint_observation after taking joint_action and ending in state.
    // Why: Needed to define the agents' observation process for planning and simulation.
    // Fit: Called during problem parsing.
    void ObservationModel::set_observation_probability(int state, const JointAction& joint_action,
                                                     const JointObservation& joint_observation, double probability) {
        if (state < 0 || state >= num_states) {
            throw std::out_of_range("Invalid state");
        }
        if (probability < 0.0 || probability > 1.0) {
            throw std::invalid_argument("Probability must be between 0 and 1");
        }
        int action_index = joint_action.get_leader_action().get_action_id() + 
                          joint_action.get_follower_action().get_action_id() * num_leader_actions;
        int obs_index = joint_observation.get_leader_observation().get_observation_id() + 
                       joint_observation.get_follower_observation().get_observation_id() * num_leader_observations;
        observation_probabilities[state][action_index][obs_index] = probability;
    }

    // Get the observation probability for a given (state, joint_action, joint_observation).
    // What: Returns the probability of observing joint_observation after taking joint_action and ending in state.
    // Why: Used in belief updates and simulation.
    double ObservationModel::get_observation_probability(int state, const JointAction& joint_action, 
                                                       const JointObservation& joint_observation) const {
        if (state < 0 || state >= num_states) {
            return 0.0;
        }
        int action_index = joint_action.get_leader_action().get_action_id() + 
                          joint_action.get_follower_action().get_action_id() * num_leader_actions;
        int obs_index = joint_observation.get_leader_observation().get_observation_id() + 
                       joint_observation.get_follower_observation().get_observation_id() * num_leader_observations;
        return observation_probabilities[state][action_index][obs_index];
    }

    // Get all possible joint observations and their probabilities for a given (state, joint_action).
    // What: Returns a vector of (joint_observation, probability) pairs.
    // Why: Used in value iteration and belief updates.
    std::vector<std::pair<JointObservation, double>> ObservationModel::get_observations_and_probabilities(
        int state, const JointAction& joint_action) const {
        std::vector<std::pair<JointObservation, double>> joint_observations;
        if (state < 0 || state >= num_states) {
            return joint_observations;
        }
        int action_index = joint_action.get_leader_action().get_action_id() + 
                          joint_action.get_follower_action().get_action_id() * num_leader_actions;
        for (int leader_obs = 0; leader_obs < num_leader_observations; ++leader_obs) {
            for (int follower_obs = 0; follower_obs < num_follower_observations; ++follower_obs) {
                int obs_index = leader_obs + follower_obs * num_leader_observations;
                double prob = observation_probabilities[state][action_index][obs_index];
                if (prob > 0.0) {
                    joint_observations.emplace_back(
                        JointObservation(Observation(leader_obs, 0), Observation(follower_obs, 1)), prob);
                }
            }
        }
        return joint_observations;
    }

    // Check if the observation model is valid (all probabilities for each state/action sum to 1).
    // What: Ensures the model is a valid probability distribution.
    // Why: Required for correct belief updates and simulation.
    bool ObservationModel::is_valid() const {
        if (num_states == 0) return false;

        for (int s = 0; s < num_states; ++s) {
            for (int a = 0; a < num_leader_actions * num_follower_actions; ++a) {
                double sum = 0.0;
                for (int o = 0; o < num_leader_observations * num_follower_observations; ++o) {
                    sum += observation_probabilities[s][a][o];
                }
                // For empty models (no probabilities set), sum will be 0 and should be invalid
                // For complete models, sum should be 1
                if (std::abs(sum - 1.0) > 1e-6) {
                    return false;
                }
            }
        }
        return true;
    }

    // Normalize the observation probabilities for each state/action.
    // What: Ensures all probabilities sum to 1 (in case of rounding errors).
    void ObservationModel::normalize() {
        for (int s = 0; s < num_states; ++s) {
            for (int a = 0; a < num_leader_actions * num_follower_actions; ++a) {
                double sum = 0.0;
                for (int o = 0; o < num_leader_observations * num_follower_observations; ++o) {
                    sum += observation_probabilities[s][a][o];
                }
                if (sum > 0.0) {
                    for (int o = 0; o < num_leader_observations * num_follower_observations; ++o) {
                        observation_probabilities[s][a][o] /= sum;
                    }
                }
            }
        }
    }

    // Print a summary of the observation model (for debugging).
    void ObservationModel::print() const {
        std::cout << "ObservationModel(states=" << num_states 
                  << ", leader_actions=" << num_leader_actions 
                  << ", follower_actions=" << num_follower_actions
                  << ", leader_observations=" << num_leader_observations
                  << ", follower_observations=" << num_follower_observations << ")\n";
    }

    // Get the list of possible observations for a given agent.
    std::vector<Observation> ObservationModel::get_observations_for_agent(int agent_id) const {
        std::vector<Observation> observations;
        int num_obs = (agent_id == 0) ? num_leader_observations : num_follower_observations;
        for (int i = 0; i < num_obs; ++i) {
            observations.emplace_back(i, agent_id);
        }
        return observations;
    }

    // Get all possible joint observations.
    std::vector<JointObservation> ObservationModel::get_joint_observations() const {
        std::vector<JointObservation> joint_observations;
        for (int lo = 0; lo < num_leader_observations; ++lo) {
            for (int fo = 0; fo < num_follower_observations; ++fo) {
                joint_observations.emplace_back(Observation(lo, 0), Observation(fo, 1));
            }
        }
        return joint_observations;
    }

JointObservation ObservationModel::sample_joint_observation(int state, const JointAction& joint_action) const {
    // Get all possible joint observations and their probabilities
    auto obs_probs = get_observations_and_probabilities(state, joint_action);
    if (obs_probs.empty()) {
        // Return a default observation if none are defined
        return JointObservation(Observation(0, 0), Observation(0, 1));
    }
    // For now, just return the first nonzero observation (stub)
    return obs_probs.front().first;
}

} // namespace posg_core 