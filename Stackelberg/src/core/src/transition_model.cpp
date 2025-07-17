// transition_model.cpp
// -------------------
// Implements the TransitionModel class for POSG.
// What: Manages the environment's transition probabilities between states given joint actions.
// Why: In POSGs, the environment is stochastic and depends on both agents' actions. The transition model is essential for belief updates and value iteration.
// Fit: Used in belief updates, simulation, and value function backups throughout the POSG solution.

#include "../include/transition_model.hpp"
#include <stdexcept>
#include <sstream>
#include <random>
#include <iostream>

namespace posg_core {

    // Default constructor: creates an empty transition model.
    TransitionModel::TransitionModel() : num_states(0), num_leader_actions(0), num_follower_actions(0) {}
    
    // Constructor: initializes the transition model with the given sizes.
    // What: Allocates space for transition probabilities for all state/action pairs.
    // Why: Needed before setting or querying any transition probabilities.
    // Fit: Called after parsing the problem structure.
    TransitionModel::TransitionModel(int num_states, int num_leader_actions, int num_follower_actions)
        : num_states(num_states), num_leader_actions(num_leader_actions), num_follower_actions(num_follower_actions) {
        // Initialize transition probabilities to zero.
        transition_probabilities.resize(num_states);
        for (int s = 0; s < num_states; ++s) {
            transition_probabilities[s].resize(num_leader_actions * num_follower_actions);
            for (int a = 0; a < num_leader_actions * num_follower_actions; ++a) {
                transition_probabilities[s][a].resize(num_states, 0.0);
            }
        }
    }

    // Set the transition probability for a given (current_state, joint_action, next_state).
    // What: Stores the probability of transitioning from current_state to next_state under joint_action.
    // Why: Needed to define the environment's dynamics for planning and simulation.
    // Fit: Called during problem parsing.
    void TransitionModel::set_transition_probability(int current_state, const JointAction& joint_action, int next_state, double probability) {
        if (current_state < 0 || current_state >= num_states) {
            throw std::out_of_range("Invalid current state");
        }
        if (next_state < 0 || next_state >= num_states) {
            throw std::out_of_range("Invalid next state");
        }
        if (probability < 0.0 || probability > 1.0) {
            throw std::invalid_argument("Probability must be between 0 and 1");
        }
        int action_index = joint_action.get_leader_action().get_action_id() +
                          joint_action.get_follower_action().get_action_id() * num_leader_actions;
        transition_probabilities[current_state][action_index][next_state] = probability;
    }

    // Get the transition probability for a given (current_state, joint_action, next_state).
    // What: Returns the probability of transitioning from current_state to next_state under joint_action.
    // Why: Used in belief updates and simulation.
    double TransitionModel::get_transition_probability(int current_state, const JointAction& joint_action, int next_state) const {
        if (current_state < 0 || current_state >= num_states || next_state < 0 || next_state >= num_states) {
            return 0.0;
        }
        int action_index = joint_action.get_leader_action().get_action_id() +
                          joint_action.get_follower_action().get_action_id() * num_leader_actions;
        return transition_probabilities[current_state][action_index][next_state];
    }

    // Sample the next state given the current state and joint action.
    // What: Draws a next state according to the transition probabilities.
    // Why: Used in simulation and policy evaluation.
    int TransitionModel::sample_next_state(int current_state, const JointAction& joint_action) const {
        int action_index = joint_action.get_leader_action().get_action_id() +
                          joint_action.get_follower_action().get_action_id() * num_leader_actions;
        const std::vector<double>& probs = transition_probabilities[current_state][action_index];
        double r = ((double) rand() / RAND_MAX);
        double cumulative = 0.0;
        for (int s = 0; s < num_states; ++s) {
            cumulative += probs[s];
            if (r <= cumulative) return s;
        }
        return num_states - 1; // fallback
    }

    // Get all possible next states and their probabilities for a given (current_state, joint_action).
    // What: Returns a vector of (next_state, probability) pairs.
    // Why: Used in value iteration and belief updates.
    std::vector<std::pair<int, double>> TransitionModel::get_next_states_and_probabilities(int current_state, const JointAction& joint_action) const {
        std::vector<std::pair<int, double>> result;
        int action_index = joint_action.get_leader_action().get_action_id() +
                          joint_action.get_follower_action().get_action_id() * num_leader_actions;
        for (int s = 0; s < num_states; ++s) {
            double prob = transition_probabilities[current_state][action_index][s];
            if (prob > 0.0) {
                result.emplace_back(s, prob);
            }
        }
        return result;
    }

    // Check if the transition model is valid (all probabilities for each state/action sum to 1).
    // What: Ensures the model is a valid probability distribution.
    // Why: Required for correct belief updates and simulation.
    bool TransitionModel::is_valid() const {
        if (num_states == 0) return false;

        for (int s = 0; s < num_states; ++s) {
            for (int a = 0; a < num_leader_actions * num_follower_actions; ++a) {
                double sum = 0.0;
                for (int s_prime = 0; s_prime < num_states; ++s_prime) {
                    sum += transition_probabilities[s][a][s_prime];
                }
                // For empty models (no probabilities set), sum will be 0 and should be invalid
                // For complete models, sum should be 1
                if (std::abs(sum - 1.0) > 1e-6) {
                    std::cout << "DEBUG: Invalid transition sum for state " << s << ", action_idx " << a << " -> " << sum << std::endl;
                    return false;
                }
            }
        }
        return true;
    }

    // Normalize the transition probabilities for each state/action.
    // What: Ensures all probabilities sum to 1 (in case of rounding errors).
    void TransitionModel::normalize() {
        for (int s = 0; s < num_states; ++s) {
            for (int a = 0; a < num_leader_actions * num_follower_actions; ++a) {
                double sum = 0.0;
                for (int s_prime = 0; s_prime < num_states; ++s_prime) {
                    sum += transition_probabilities[s][a][s_prime];
                }
                if (sum > 0.0) {
                    for (int s_prime = 0; s_prime < num_states; ++s_prime) {
                        transition_probabilities[s][a][s_prime] /= sum;
                    }
                }
            }
        }
    }

    // Print a summary of the transition model (for debugging).
    void TransitionModel::print() const {
        std::cout << "TransitionModel(states=" << num_states 
                  << ", leader_actions=" << num_leader_actions 
                  << ", follower_actions=" << num_follower_actions << ")\n";
    }

    // Get the list of state indices.
    std::vector<int> TransitionModel::get_states() const {
        std::vector<int> states(num_states);
        for (int i = 0; i < num_states; ++i) {
            states[i] = i;
        }
        return states;
    }

    // Get all possible joint actions.
    std::vector<JointAction> TransitionModel::get_joint_actions() const {
        std::vector<JointAction> joint_actions;
        for (int la = 0; la < num_leader_actions; ++la) {
            for (int fa = 0; fa < num_follower_actions; ++fa) {
                joint_actions.emplace_back(Action(la, 0), Action(fa, 1));
            }
        }
        return joint_actions;
    }

} // namespace posg_core 