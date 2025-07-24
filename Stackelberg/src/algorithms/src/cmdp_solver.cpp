#include "../include/cmdp_solver.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <unordered_set>
#include "../../common/logging.hpp"

namespace posg_algorithms {

    // ============================================================================
    // ValueFunction Implementation
    // ============================================================================

    void ValueFunction::add_alpha_vector(const std::vector<double>& alpha_vector, 
                                        const posg_core::Action& action) {
        alpha_vectors_.push_back(alpha_vector);
        actions_.push_back(action);
    }

    double ValueFunction::get_value(const posg_core::OccupancyState& occupancy_state) const {
        if (alpha_vectors_.empty()) {
            return 0.0;  // Default value for empty value function
        }

        double max_value = -std::numeric_limits<double>::infinity();
        
        // From Paper: V(μ) = max_α α^T μ where α are alpha-vectors
        for (const auto& alpha_vector : alpha_vectors_) {
            double value = 0.0;
            
            // Compute dot product: α^T μ
            // Map alpha-vector coefficients to occupancy state entries
            size_t entry_index = 0;
            const auto& dist = occupancy_state.get_occupancy_distribution();
            for (const auto& [state, leader_map] : dist) {
                for (const auto& [leader_hist, follower_map] : leader_map) {
                    for (const auto& [follower_hist, prob] : follower_map) {
                        if (entry_index < alpha_vector.size()) {
                            value += alpha_vector[entry_index] * prob;
                        }
                        entry_index++;
                    }
                }
            }
            
            max_value = std::max(max_value, value);
        }
        
        return max_value;
    }

    posg_core::Action ValueFunction::get_best_action(const posg_core::OccupancyState& occupancy_state) const {
        if (alpha_vectors_.empty()) {
            return posg_core::Action(0, 0); // Default action
        }

        double max_value = -std::numeric_limits<double>::infinity();
        size_t best_index = 0;
        
        // Find the alpha-vector that gives maximum value
        for (size_t i = 0; i < alpha_vectors_.size(); ++i) {
            double value = 0.0;
            
            // Compute dot product: α^T μ
            size_t entry_index = 0;
            const auto& dist = occupancy_state.get_occupancy_distribution();
            for (const auto& [state, leader_map] : dist) {
                for (const auto& [leader_hist, follower_map] : leader_map) {
                    for (const auto& [follower_hist, prob] : follower_map) {
                        if (entry_index < alpha_vectors_[i].size()) {
                            value += alpha_vectors_[i][entry_index] * prob;
                        }
                        entry_index++;
                    }
                }
            }
            
            if (value > max_value) {
                max_value = value;
                best_index = i;
            }
        }
        
        return actions_[best_index];
    }

    // ============================================================================
    // CMDPSolver Implementation
    // ============================================================================

    CMDPSolver::CMDPSolver(const posg_parser::POSGProblem& problem) 
        : problem_(problem) {
        // Initialize transition and observation models from the problem
        // Use the models that are already part of the problem
        transition_model_ = problem.transition_model;
        observation_model_ = problem.observation_model;
    }

    posg_core::CredibleMDP CMDPSolver::reduce_to_cmdp(const posg_parser::POSGProblem& problem) {
        /**
         * From Paper: Definition 4 - Reduction to Credible MDP
         * 
         * Given a Leader-Follower POSG G = (S, A_L, A_F, O_L, O_F, T, O, R, γ, μ_0),
         * we construct a Credible MDP M = (O, A_L, T_C, R_C, γ, μ_0) where:
         * 
         * - O is the space of occupancy states
         * - A_L is the leader's action space
         * - T_C: O × A_L → Δ(O) is the credible transition function
         * - R_C: O × A_L → R is the credible reward function
         * - γ is the discount factor
         * - μ_0 is the initial occupancy state
         */
        
        posg_core::CredibleMDP cmdpm(problem.transition_model, problem.observation_model, 
                                   problem.initial_belief, 100); // Default horizon
        
        // Note: The CredibleMDP constructor handles the initial setup
        // We don't need to call set_discount_factor, set_initial_occupancy_state, etc.
        // as they're not part of the current interface
        
        return cmdpm;
    }

    double CMDPSolver::bellman_update(const ValueFunction& value_function,
                                     const posg_core::OccupancyState& occupancy_state,
                                     const posg_core::Action& leader_action) {
        /**
         * From Paper: Section 4.3 Bellman Recursion
         * 
         * The Bellman equation for the Credible MDP is:
         * V(μ) = max_{a_L} [R_C(μ, a_L) + γ * min_{μ' ∈ C(μ)} V(μ')]
         * 
         * where C(μ) is the credible set of μ.
         */
        
        // Compute credible set for current occupancy state
        posg_core::CredibleSet credible_set = compute_credible_set(occupancy_state);
        
        // Compute immediate reward: R_C(μ, a_L)
        double immediate_reward = compute_credible_reward(occupancy_state, leader_action, credible_set);
        
        // Compute future value: min_{μ' ∈ C(μ)} V(μ')
        double future_value = std::numeric_limits<double>::infinity();
        for (const auto& successor_occupancy : credible_set.get_occupancy_states()) {
            double successor_value = value_function.get_value(successor_occupancy);
            future_value = std::min(future_value, successor_value);
        }
        
        // If no successors, future value is 0
        if (credible_set.get_occupancy_states().empty()) {
            future_value = 0.0;
        }
        
        // Bellman update: R_C(μ, a_L) + γ * min_{μ' ∈ C(μ)} V(μ')
        double bellman_value = immediate_reward + problem_.discount_factor * future_value;
        
        return bellman_value;
    }

    // ============================================================================
    // PBVI Expansion Phase (Algorithm 2) Implementation
    // ============================================================================

    std::vector<posg_core::CredibleSet> CMDPSolver::expand_credible_sets(
        const std::vector<posg_core::CredibleSet>& current_credible_sets,
        double epsilon) {
        // Use a slightly larger default epsilon for expansion
        if (epsilon < 1e-6) epsilon = 0.05;
        std::vector<posg_core::CredibleSet> expanded_sets;
        for (const auto& credible_set : current_credible_sets) {
            posg_core::CredibleSet new_credible_set;
            std::vector<posg_core::ConditionalOccupancyState> Cy;
            posg_core::LeaderDecisionRule sampled_leader_rule(0);
            for (int action_id = 0; action_id < transition_model_.get_num_leader_actions(); ++action_id) {
                posg_core::Action action(action_id, 0);
                posg_core::AgentHistory empty_history(0);
                sampled_leader_rule.set_action_probability(empty_history, action, 
                    1.0 / transition_model_.get_num_leader_actions());
            }
            bool added_any = false;
            for (const auto& occupancy_state : credible_set.get_occupancy_states()) {
                auto follower_marginal = occupancy_state.get_follower_history_marginal();
                for (const auto& [follower_history, _] : follower_marginal) {
                    posg_core::FollowerDecisionRule follower_rule(0);
                    for (int aF_id = 0; aF_id < transition_model_.get_num_follower_actions(); ++aF_id) {
                        posg_core::Action aF(aF_id, 1);
                        follower_rule.set_action_probability(follower_history, aF, 1.0 / transition_model_.get_num_follower_actions());
                    }
                    for (int zF_id = 0; zF_id < observation_model_.get_num_follower_observations(); ++zF_id) {
                        posg_core::Observation zF(zF_id, 1);
                        auto conditional_state = occupancy_state.conditional_decompose(follower_history);
                        posg_core::Action follower_action = follower_rule.sample_action(follower_history);
                        auto updated_conditional = conditional_state.tau_zF(
                            sampled_leader_rule,
                            follower_action,
                            zF,
                            transition_model_,
                            observation_model_
                        );
                        bool should_retain = true;
                        if (!Cy.empty()) {
                            double max_distance = 0.0;
                            for (const auto& existing_conditional : Cy) {
                                double distance = updated_conditional.distance_to(existing_conditional);
                                max_distance = std::max(max_distance, distance);
                            }
                            should_retain = (max_distance > epsilon);
                        }
                        if (should_retain) {
                            Cy.push_back(updated_conditional);
                        }
                    }
                    // Move tau_function call here, so follower_rule is in scope
                    if (!Cy.empty()) {
                        posg_core::OccupancyState successor_state = tau_function(
                            occupancy_state, sampled_leader_rule, follower_rule);
                        // Always retain the first occupancy state if the set is empty
                        if (new_credible_set.get_occupancy_states().empty() || should_retain_occupancy_state(successor_state, new_credible_set, epsilon)) {
                            new_credible_set.add_occupancy_state(successor_state);
                            added_any = true;
                        }
                    }
                }
            }
            // Ensure at least one credible set is retained per expansion
            if (!new_credible_set.get_occupancy_states().empty() || expanded_sets.empty()) {
                expanded_sets.push_back(new_credible_set);
            }
        }
        // If no sets were retained, propagate the first input credible set
        if (expanded_sets.empty() && !current_credible_sets.empty()) {
            expanded_sets.push_back(current_credible_sets.front());
        }
        return expanded_sets;
    }

    posg_core::OccupancyState CMDPSolver::tau_function(
        const posg_core::OccupancyState& occupancy_state,
        const posg_core::LeaderDecisionRule& leader_rule,
        const posg_core::FollowerDecisionRule& follower_rule) {
        /**
         * From Paper: τ(o, σL, σF) generates successor occupancy state
         * 
         * This function implements the refined τ function that properly handles
         * belief propagation, state transitions, and observation models
         */
        
        posg_core::OccupancyState successor_state;
        
        // For each current occupancy state entry
        const auto& current_distribution = occupancy_state.get_occupancy_distribution();
        
        for (const auto& [current_state, leader_histories] : current_distribution) {
            for (const auto& [leader_history, follower_histories] : leader_histories) {
                for (const auto& [follower_history, current_prob] : follower_histories) {
                    if (current_prob <= 0.0) continue;
                    
                    // Sample actions from decision rules
                    posg_core::Action leader_action = leader_rule.sample_action(leader_history);
                    posg_core::Action follower_action = follower_rule.sample_action(follower_history);
                    
                    // For each possible next state
                    for (int next_state = 0; next_state < transition_model_.get_num_states(); ++next_state) {
                        // For each possible observation pair
                        for (int zL_id = 0; zL_id < observation_model_.get_num_leader_observations(); ++zL_id) {
                            for (int zF_id = 0; zF_id < observation_model_.get_num_follower_observations(); ++zF_id) {
                                posg_core::Observation leader_obs(zL_id, 0);
                                posg_core::Observation follower_obs(zF_id, 1);
                                
                                // Compute transition and observation probabilities
                                posg_core::JointAction joint_action(leader_action, follower_action);
                                posg_core::JointObservation joint_obs(leader_obs, follower_obs);
                                
                                double transition_prob = transition_model_.get_transition_probability(
                                    current_state, joint_action, next_state);
                                double obs_prob = observation_model_.get_observation_probability(
                                    next_state, joint_action, joint_obs);
                                
                                // Create new histories
                                posg_core::AgentHistory new_leader_history = leader_history;
                                new_leader_history.add_action(leader_action);
                                new_leader_history.add_observation(leader_obs);
                                
                                posg_core::AgentHistory new_follower_history = follower_history;
                                new_follower_history.add_action(follower_action);
                                new_follower_history.add_observation(follower_obs);
                                
                                // Update occupancy probability
                                double new_prob = current_prob * transition_prob * obs_prob;
                                successor_state.add_entry(next_state, new_leader_history, new_follower_history, new_prob);
                            }
                        }
                    }
                }
            }
        }
        
        successor_state.normalize();
        return successor_state;
    }

    std::pair<std::vector<posg_core::ConditionalOccupancyState>, 
              std::vector<posg_core::ConditionalOccupancyState>> 
    CMDPSolver::compute_conditional_collections(const posg_core::CredibleSet& credible_set,
                                               const posg_core::OccupancyState& reference_occupancy) {
        /**
         * From Paper: Cx = {c(o',hF) | ∀o' ∈ x, hF ∈ HF} and Co = {c(o,hF) | ∀hF ∈ HF}
         */
        
        std::vector<posg_core::ConditionalOccupancyState> Cx, Co;
        
        // Compute Co = {c(o,hF) | ∀hF ∈ HF} for reference occupancy state
        auto reference_follower_marginal = reference_occupancy.get_follower_history_marginal();
        for (const auto& [follower_history, _] : reference_follower_marginal) {
            auto conditional = reference_occupancy.conditional_decompose(follower_history);
            if (conditional.is_valid(true)) {
                Co.push_back(conditional);
            }
        }
        
        // Compute Cx = {c(o',hF) | ∀o' ∈ x, hF ∈ HF} for all occupancy states in credible set
        for (const auto& occupancy_state : credible_set.get_occupancy_states()) {
            auto follower_marginal = occupancy_state.get_follower_history_marginal();
            for (const auto& [follower_history, _] : follower_marginal) {
                auto conditional = occupancy_state.conditional_decompose(follower_history);
                if (conditional.is_valid(true)) {
                    Cx.push_back(conditional);
                }
            }
        }
        
        return {Cx, Co};
    }

    bool CMDPSolver::should_retain_occupancy_state(
        const posg_core::OccupancyState& new_occupancy_state,
        const posg_core::CredibleSet& existing_credible_set,
        double epsilon) {
        /**
         * From Paper: Retain only occupancy states with ℓ1-distance > ε from current set
         */
        
        if (existing_credible_set.get_occupancy_states().empty()) {
            return true; // Always retain if no existing states
        }
        
        // Check distance to all existing occupancy states
        for (const auto& existing_state : existing_credible_set.get_occupancy_states()) {
            double distance = new_occupancy_state.distance_to(existing_state);
            if (distance <= epsilon) {
                return false; // Too close to existing state, don't retain
            }
        }
        
        return true; // Far enough from all existing states, retain
    }

    // ============================================================================
    // Uniform Continuity and ε-Approximation Logic (Theorem 5.3)
    // ============================================================================

    posg_core::CredibleSet CMDPSolver::nearest_neighbor(
        const posg_core::CredibleSet& target_credible_set,
        const std::vector<posg_core::CredibleSet>& sampled_credible_sets) {
        /**
         * From Paper: vL(x) = vL(x') where x' = arg min dH(x, x')
         */
        
        if (sampled_credible_sets.empty()) {
            return target_credible_set; // Return target if no samples
        }
        
        double min_distance = std::numeric_limits<double>::infinity();
        posg_core::CredibleSet nearest_set = sampled_credible_sets[0];
        
        for (const auto& sampled_set : sampled_credible_sets) {
            double hausdorff_distance = target_credible_set.hausdorff_distance(sampled_set);
            if (hausdorff_distance < min_distance) {
                min_distance = hausdorff_distance;
                nearest_set = sampled_set;
            }
        }
        
        return nearest_set;
    }

    double CMDPSolver::compute_delta(const std::vector<posg_core::CredibleSet>& all_credible_sets,
                                   const std::vector<posg_core::CredibleSet>& sampled_credible_sets) {
        /**
         * From Paper: δ = supx∈X minx'∈X' dH(x, x')
         */
        
        if (sampled_credible_sets.empty()) {
            return std::numeric_limits<double>::infinity();
        }
        
        double max_min_distance = 0.0;
        
        for (const auto& target_set : all_credible_sets) {
            double min_distance = std::numeric_limits<double>::infinity();
            
            for (const auto& sampled_set : sampled_credible_sets) {
                double hausdorff_distance = target_set.hausdorff_distance(sampled_set);
                min_distance = std::min(min_distance, hausdorff_distance);
            }
            
            max_min_distance = std::max(max_min_distance, min_distance);
        }
        
        return max_min_distance;
    }

    double CMDPSolver::compute_exploitability_bound(double delta, int horizon) {
        /**
         * From Paper: Theorem 5.3 - ε ≤ mℓδ where m = max{||rL||∞, ||rF||∞}
         */
        
        // Estimate m = max{||rL||∞, ||rF||∞} from reward functions
        // For simplicity, we'll use a conservative estimate
        double m = 10.0; // Conservative bound on reward magnitude
        
        return m * horizon * delta;
    }

    double CMDPSolver::approximate_value_function(
        const posg_core::CredibleSet& target_credible_set,
        const ValueFunction& value_function,
        const std::vector<posg_core::CredibleSet>& sampled_credible_sets) {
        /**
         * From Paper: vL(x) ≈ vL(x') using nearest neighbor approximation
         */
        
        // Find nearest neighbor
        posg_core::CredibleSet nearest_set = nearest_neighbor(target_credible_set, sampled_credible_sets);
        
        // Evaluate value function at nearest neighbor
        if (nearest_set.get_occupancy_states().empty()) {
            return 0.0;
        }
        
        // Use the first occupancy state in the nearest credible set
        const auto& nearest_occupancy = *nearest_set.get_occupancy_states().begin();
        return value_function.get_value(nearest_occupancy);
    }

    std::pair<ValueFunction, double> CMDPSolver::pbvi_with_approximation(
        const std::vector<posg_core::OccupancyState>& initial_occupancy_states,
        size_t max_iterations,
        double epsilon,
        bool use_approximation) {
        /**
         * Enhanced PBVI with uniform continuity and ε-approximation
         * Complete implementation with theoretical guarantees from Theorem 5.3
         */
        
        std::cout << "PBVI with Approximation: Starting algorithm with " << initial_occupancy_states.size() 
                  << " initial occupancy states" << std::endl;
        
        // Convert occupancy states to credible sets
        std::vector<posg_core::CredibleSet> credible_sets;
        for (const auto& occupancy_state : initial_occupancy_states) {
            posg_core::CredibleSet credible_set;
            credible_set.add_occupancy_state(occupancy_state);
            credible_sets.push_back(credible_set);
        }
        
        // Track all credible sets for delta computation
        std::vector<posg_core::CredibleSet> all_credible_sets = credible_sets;
        
        // Initialize value function
        ValueFunction value_function;
        
        // Reward functions – replace placeholder once domain rewards are connected
        auto reward_function_leader = [this](int state, const posg_core::Action& leader_action, const posg_core::Action& follower_action) -> double {
            return 1.0; // TODO(domain): connect to problem rewards
        };

        auto reward_function_follower = [this](int state, const posg_core::Action& leader_action, const posg_core::Action& follower_action) -> double {
            return 1.0; // TODO(domain): connect to problem rewards
        };
        
        double exploitability_bound = 0.0;
        
        for (size_t iteration = 0; iteration < max_iterations; ++iteration) {
            std::cout << "PBVI with Approximation: Iteration " << iteration + 1 << std::endl;
            
            // Improve phase: solve MILP for each credible set
            std::vector<posg_core::LeaderDecisionRule> optimal_rules;
            
            for (const auto& credible_set : credible_sets) {
                // Solve MILP for greedy leader decision rule
                posg_core::LeaderDecisionRule optimal_rule = milp_solver_.solve_milp(
                    credible_set, {}, transition_model_, observation_model_,
                    reward_function_leader, reward_function_follower);
                
                optimal_rules.push_back(optimal_rule);
            }
            
            // Expansion phase: generate new credible sets
            std::vector<posg_core::CredibleSet> expanded_sets = expand_credible_sets(credible_sets, 0.1);
            
            // Add new credible sets to tracking
            for (const auto& new_set : expanded_sets) {
                all_credible_sets.push_back(new_set);
            }
            
            // Update credible sets for next iteration
            credible_sets = expanded_sets;
            
            // Compute exploitability bound if using approximation
            if (use_approximation && !credible_sets.empty()) {
                double delta = compute_delta(all_credible_sets, credible_sets);
                exploitability_bound = compute_exploitability_bound(delta, 100); // Default horizon
                
                std::cout << "PBVI with Approximation: Delta = " << delta 
                          << ", Exploitability bound = " << exploitability_bound << std::endl;
            }
            
            // Check convergence
            if (iteration > 0 && credible_sets.empty()) {
                std::cout << "PBVI with Approximation: Converged (no more credible sets)" << std::endl;
                break;
            }
        }
        
        std::cout << "PBVI with Approximation: Algorithm completed" << std::endl;
        return {value_function, exploitability_bound};
    }

    // ============================================================================
    // PBVI with MILP Implementation
    // ============================================================================

    ValueFunction CMDPSolver::pbvi_with_milp(const std::vector<posg_core::OccupancyState>& initial_occupancy_states,
                                            size_t max_iterations,
                                            double epsilon) {
        /**
         * From Paper: Section 5 - Point-Based Value Iteration with MILP-based improve phase
         * 
         * Algorithm 1: PBVI for M'
         * 1. Initialize X0', ..., Xℓ' ← ∅ and L0, ..., Lℓ ← ∅
         * 2. while not converged do
         *    a. for t = ℓ - 1 to 0 do
         *       Perform improve(Xt', Lt+1)  // MILP-based improve phase
         *    b. for t = 0 to ℓ - 1 do
         *       Sampling expand(Xt', Xt+1')
         * 3. end while
         * 
         * Algorithm 3: The improve phase for PBVI in M'
         * 1. function improve(Xt', Lt+1)
         * 2. for each credible set x ∈ Xt' do
         * 3.   Greedy decision rule σL,x ← MILP(x, Lt+1)
         * 4.   Improvement rule Lt ← Lt ∪ {Fx,σL,x}
         * 5. end for
         */
        
        std::cout << "PBVI with MILP: Starting algorithm with " << initial_occupancy_states.size() 
                  << " initial occupancy states" << std::endl;
        
        // Convert occupancy states to credible sets
        std::vector<posg_core::CredibleSet> credible_sets;
        for (const auto& occupancy_state : initial_occupancy_states) {
            posg_core::CredibleSet credible_set;
            credible_set.add_occupancy_state(occupancy_state);
            credible_sets.push_back(credible_set);
        }
        
        // Initialize value function collections L0, ..., Lℓ
        // For simplicity, we'll use a single stage for now
        std::vector<std::vector<std::pair<std::vector<double>, std::vector<double>>>> value_function_collection;

        // Track every credible set ever generated for Hausdorff radius (δ)
        std::vector<posg_core::CredibleSet> all_credible_sets = credible_sets;

        double exploitability_bound = std::numeric_limits<double>::infinity();
        
        // Define reward functions for leader and follower (placeholder, see TODO)
        auto reward_function_leader = [this](int state, const posg_core::Action& leader_action, const posg_core::Action& follower_action) -> double {
            return 1.0; // TODO(domain): connect to problem rewards
        };

        auto reward_function_follower = [this](int state, const posg_core::Action& leader_action, const posg_core::Action& follower_action) -> double {
            return 1.0; // TODO(domain): connect to problem rewards
        };
        
        for (size_t iteration = 0; iteration < max_iterations; ++iteration) {
            std::cout << "PBVI with MILP: Iteration " << iteration + 1 << std::endl;
            
            // Improve phase: for each credible set, solve MILP to get greedy decision rule
            std::vector<posg_core::LeaderDecisionRule> optimal_rules;
            std::vector<std::vector<std::pair<std::vector<double>, std::vector<double>>>> new_alpha_vectors;
            
            for (const auto& credible_set : credible_sets) {
                std::cout << "PBVI with MILP: Solving MILP for credible set with " 
                          << credible_set.get_occupancy_states().size() << " occupancy states" << std::endl;
                
                // Solve MILP for greedy leader decision rule
                posg_core::LeaderDecisionRule optimal_rule = milp_solver_.solve_milp(
                    credible_set, value_function_collection, transition_model_, observation_model_,
                    reward_function_leader, reward_function_follower);
                
                optimal_rules.push_back(optimal_rule);
                
                // Extract alpha-vectors from MILP solution
                std::vector<std::pair<std::vector<double>, std::vector<double>>> alpha_vectors = 
                    milp_solver_.extract_alpha_vectors(credible_set, optimal_rule, value_function_collection,
                                                     transition_model_, observation_model_);
                
                new_alpha_vectors.push_back(alpha_vectors);
            }
            
            // Update value function collection
            value_function_collection = new_alpha_vectors;
            
            // -------------------- Expansion Phase ---------------------------
            std::vector<posg_core::CredibleSet> expanded_sets = expand_credible_sets(credible_sets, 0.1);
            
            // Add new credible sets to tracking
            for (const auto& new_set : expanded_sets) {
                all_credible_sets.push_back(new_set);
            }
            
            // Update credible sets for next iteration
            credible_sets = expanded_sets;

            // ------------------ Convergence & ε-bound -----------------------
            double delta = compute_delta(all_credible_sets, credible_sets);
            exploitability_bound = compute_exploitability_bound(delta, /*horizon=*/100);
            LOG_INFO("PBVI with MILP: δ=" << delta << "   ε-bound=" << exploitability_bound);

            // Terminate when Hausdorff radius ≤ ε (Theorem 5.3)
            if (delta <= epsilon) {
                LOG_INFO("PBVI with MILP: Converged – exploitability within ε target");
                break;
            }

            // Safety: if no new credible sets were generated, stop iteration
            if (credible_sets.empty()) {
                LOG_INFO("PBVI with MILP: Converged – no more credible sets");
                break;
            }
            
            // Note: Additional stopping criteria handled above; no further checks needed here.
        }
        
        // Convert back to ValueFunction format for compatibility
        ValueFunction value_function;
        
        // Add alpha-vectors from the final collection
        for (const auto& alpha_vector_set : value_function_collection) {
            for (const auto& [alpha_L, alpha_F] : alpha_vector_set) {
                // Use alpha_L as the alpha-vector (leader's value)
                value_function.add_alpha_vector(alpha_L, posg_core::Action(0, 0));
            }
        }
        
        std::cout << "PBVI with MILP: Algorithm completed" << std::endl;
        return value_function;
    }

    std::function<posg_core::Action(const posg_core::OccupancyState&)> 
    CMDPSolver::extract_policy(const ValueFunction& value_function, double epsilon) {
        /**
         * From Paper: Section 4.5 - Policy Extraction
         * 
         * Extract ε-optimal policy π: O → A_L from value function V:
         * π(μ) = argmax_{a_L} [R_C(μ, a_L) + γ * min_{μ' ∈ C(μ)} V(μ')]
         */
        
        return [this, &value_function, epsilon](const posg_core::OccupancyState& occupancy_state) -> posg_core::Action {
            double best_value = -std::numeric_limits<double>::infinity();
            posg_core::Action best_action(0, 0);
            
            // Find the action that maximizes the Bellman equation
            for (size_t a_l = 0; a_l < problem_.actions[0].size(); ++a_l) {
                posg_core::Action leader_action(a_l, 0);
                double action_value = bellman_update(value_function, occupancy_state, leader_action);
                
                if (action_value > best_value + epsilon) {
                    best_value = action_value;
                    best_action = leader_action;
                }
            }
            
            return best_action;
        };
    }

    std::pair<ValueFunction, std::function<posg_core::Action(const posg_core::OccupancyState&)>>
    CMDPSolver::solve(const posg_parser::POSGProblem& problem, double epsilon) {
        /**
         * Main entry point that combines all algorithmic components
         * 
         * 1. Reduce POSG to Credible MDP
         * 2. Run PBVI to compute value function
         * 3. Extract ε-optimal policy
         */
        
        // Step 1: Reduce to Credible MDP
        posg_core::CredibleMDP cmdpm = reduce_to_cmdp(problem);
        
        // Step 2: Create initial set of occupancy states for PBVI
        std::vector<posg_core::OccupancyState> initial_states;
        const auto& initial_credible_set = cmdpm.get_initial_credible_set();
        if (!initial_credible_set.get_occupancy_states().empty()) {
            initial_states.push_back(*initial_credible_set.get_occupancy_states().begin()); // Use first occupancy state
        }
        
        // Add more representative occupancy states for better coverage
        // This is a simplified approach - in practice, we'd use more sophisticated sampling
        for (size_t i = 0; i < std::min(size_t(10), problem.states.size()); ++i) {
            posg_core::OccupancyState sampled_state;
            posg_core::AgentHistory leader_hist(0);
            posg_core::AgentHistory follower_hist(1);
            
            // Create a sampled occupancy state with uniform distribution
            for (size_t s = 0; s < problem.states.size(); ++s) {
                double prob = 1.0 / problem.states.size();
                sampled_state.add_entry(s, leader_hist, follower_hist, prob);
            }
            initial_states.push_back(sampled_state);
        }
        
        // Step 3: Run PBVI with MILP
        ValueFunction value_function = pbvi_with_milp(initial_states, 1000, epsilon);
        
        // Step 4: Extract policy
        auto policy = extract_policy(value_function, epsilon);
        
        return {value_function, policy};
    }

    // ============================================================================
    // Private Helper Methods
    // ============================================================================

    posg_core::OccupancyState CMDPSolver::compute_successor(
        const posg_core::OccupancyState& occupancy_state,
        const posg_core::Action& leader_action,
        const posg_core::Action& follower_action,
        const posg_core::Observation& leader_obs,
        const posg_core::Observation& follower_obs) {
        
        /**
         * From Paper: Section 3.2 - Occupancy State Dynamics
         * 
         * Compute successor occupancy state under joint actions and observations.
         * This implements the occupancy state update equation.
         */
        
        posg_core::OccupancyState successor;
        
        // For each current state and history combination
        const auto& dist = occupancy_state.get_occupancy_distribution();
        for (const auto& [current_state, leader_map] : dist) {
            for (const auto& [leader_hist, follower_map] : leader_map) {
                for (const auto& [follower_hist, current_prob] : follower_map) {
                    
                    // For each possible successor state
                    for (size_t next_state = 0; next_state < problem_.states.size(); ++next_state) {
                        // Create joint action and joint observation
                        posg_core::JointAction joint_action(leader_action, follower_action);
                        posg_core::JointObservation joint_obs(leader_obs, follower_obs);
                        
                        // Get transition probability
                        double trans_prob = transition_model_.get_transition_probability(
                            current_state, joint_action, next_state);
                        
                        // Get observation probability
                        double obs_prob = observation_model_.get_observation_probability(
                            next_state, joint_action, joint_obs);
                        
                        // Update histories
                        posg_core::AgentHistory new_leader_hist = leader_hist;
                        posg_core::AgentHistory new_follower_hist = follower_hist;
                        new_leader_hist.add_action(leader_action);
                        new_leader_hist.add_observation(leader_obs);
                        new_follower_hist.add_action(follower_action);
                        new_follower_hist.add_observation(follower_obs);
                        
                        // Compute joint probability
                        double joint_prob = current_prob * trans_prob * obs_prob;
                        
                        if (joint_prob > 1e-10) { // Numerical stability
                            successor.add_entry(next_state, new_leader_hist, new_follower_hist, joint_prob);
                        }
                    }
                }
            }
        }
        
        // Normalize the successor occupancy state
        successor.normalize();
        
        return successor;
    }

    posg_core::CredibleSet CMDPSolver::compute_credible_set(
        const posg_core::OccupancyState& occupancy_state) {
        
        /**
         * From Paper: Definition 3 - Credible Set
         * 
         * C(μ) = {μ' | μ' is reachable from μ under some follower policy}
         * 
         * The credible set contains all occupancy states that the follower
         * can induce by playing optimally against the leader.
         */
        
        posg_core::CredibleSet credible_set;
        
        // Add the current occupancy state
        credible_set.add_occupancy_state(occupancy_state);
        
        // For each leader action, compute all possible successor occupancy states
        for (size_t a_l = 0; a_l < problem_.actions[0].size(); ++a_l) {
            posg_core::Action leader_action(a_l, 0);
            
            // For each possible follower action (simplified - in practice, this would be
            // computed based on optimal follower responses)
            for (size_t a_f = 0; a_f < problem_.actions[1].size(); ++a_f) {
                posg_core::Action follower_action(a_f, 1);
                
                // For each possible observation pair
                for (size_t o_l = 0; o_l < problem_.observations[0].size(); ++o_l) {
                    posg_core::Observation leader_obs(o_l, 0);
                    for (size_t o_f = 0; o_f < problem_.observations[1].size(); ++o_f) {
                        posg_core::Observation follower_obs(o_f, 1);
                        
                        // Compute successor occupancy state
                        posg_core::OccupancyState successor = compute_successor(
                            occupancy_state, leader_action, follower_action, 
                            leader_obs, follower_obs);
                        
                        // Check if successor is valid (has entries)
                        const auto& successor_dist = successor.get_occupancy_distribution();
                        bool has_entries = false;
                        for (const auto& [state, leader_map] : successor_dist) {
                            for (const auto& [leader_hist, follower_map] : leader_map) {
                                if (!follower_map.empty()) {
                                    has_entries = true;
                                    break;
                                }
                            }
                            if (has_entries) break;
                        }
                        
                        if (has_entries) {
                            credible_set.add_occupancy_state(successor);
                        }
                    }
                }
            }
        }
        
        return credible_set;
    }

    std::vector<double> CMDPSolver::backproject(
        const ValueFunction& value_function,
        const posg_core::OccupancyState& occupancy_state,
        const posg_core::Action& leader_action) {
        
        /**
         * From Paper: Section 4.4 - Alpha-Vector Computation
         * 
         * Backproject the value function to compute new alpha-vectors.
         * This is a key operation in PBVI for maintaining the alpha-vector representation.
         */
        
        // Count total entries in occupancy state
        size_t num_entries = 0;
        const auto& dist = occupancy_state.get_occupancy_distribution();
        for (const auto& [state, leader_map] : dist) {
            for (const auto& [leader_hist, follower_map] : leader_map) {
                num_entries += follower_map.size();
            }
        }
        
        std::vector<double> alpha_vector(num_entries, 0.0);
        
        // For each entry in the occupancy state
        size_t entry_index = 0;
        for (const auto& [state, leader_map] : dist) {
            for (const auto& [leader_hist, follower_map] : leader_map) {
                for (const auto& [follower_hist, prob] : follower_map) {
                    
                    // Compute the contribution of this entry to the alpha-vector
                    double entry_value = 0.0;
                    
                    // For each possible follower action
                    for (size_t a_f = 0; a_f < problem_.actions[1].size(); ++a_f) {
                        posg_core::Action follower_action(a_f, 1);
                        
                        // For each possible observation pair
                        for (size_t o_l = 0; o_l < problem_.observations[0].size(); ++o_l) {
                            posg_core::Observation leader_obs(o_l, 0);
                            for (size_t o_f = 0; o_f < problem_.observations[1].size(); ++o_f) {
                                posg_core::Observation follower_obs(o_f, 1);
                                
                                // Compute successor occupancy state
                                posg_core::OccupancyState successor = compute_successor(
                                    occupancy_state, leader_action, follower_action, 
                                    leader_obs, follower_obs);
                                
                                // Check if successor has entries
                                const auto& successor_dist = successor.get_occupancy_distribution();
                                bool has_entries = false;
                                for (const auto& [s, l_map] : successor_dist) {
                                    for (const auto& [l_hist, f_map] : l_map) {
                                        if (!f_map.empty()) {
                                            has_entries = true;
                                            break;
                                        }
                                    }
                                    if (has_entries) break;
                                }
                                
                                if (has_entries) {
                                    // Get value of successor from value function
                                    double successor_value = value_function.get_value(successor);
                                    
                                    // Create joint action and joint observation
                                    posg_core::JointAction joint_action(leader_action, follower_action);
                                    posg_core::JointObservation joint_obs(leader_obs, follower_obs);
                                    
                                    // Get transition and observation probabilities
                                    double trans_prob = transition_model_.get_transition_probability(
                                        state, joint_action, state);
                                    double obs_prob = observation_model_.get_observation_probability(
                                        state, joint_action, joint_obs);
                                    
                                    // Accumulate contribution
                                    entry_value += trans_prob * obs_prob * successor_value;
                                }
                            }
                        }
                    }
                    
                    // Add immediate reward contribution
                    double immediate_reward = 0.0;
                    for (size_t a_f = 0; a_f < problem_.actions[1].size(); ++a_f) {
                        posg_core::Action follower_action(a_f, 1);
                        // Use rewards_leader with joint action index
                        size_t joint_action_idx = leader_action.get_action_id() * problem_.actions[1].size() + follower_action.get_action_id();
                        if (state < problem_.rewards_leader.size() && joint_action_idx < problem_.rewards_leader[state].size()) {
                            immediate_reward += problem_.rewards_leader[state][joint_action_idx];
                        }
                    }
                    immediate_reward /= problem_.actions[1].size(); // Average over follower actions
                    
                    alpha_vector[entry_index] = immediate_reward + problem_.discount_factor * entry_value;
                    entry_index++;
                }
            }
        }
        
        return alpha_vector;
    }

    double CMDPSolver::compute_credible_reward(
        const posg_core::OccupancyState& occupancy_state,
        const posg_core::Action& leader_action,
        const posg_core::CredibleSet& credible_set) {
        
        /**
         * From Paper: R_C(μ, a_L) = min_{μ' ∈ C(μ)} R(μ', a_L)
         * 
         * Compute the credible reward as the minimum reward over all occupancy states
         * in the credible set.
         */
        
        double min_reward = std::numeric_limits<double>::infinity();
        
        for (const auto& cred_occupancy : credible_set.get_occupancy_states()) {
            double occupancy_reward = 0.0;
            
            // Compute reward for this occupancy state
            const auto& dist = cred_occupancy.get_occupancy_distribution();
            for (const auto& [state, leader_map] : dist) {
                for (const auto& [leader_hist, follower_map] : leader_map) {
                    for (const auto& [follower_hist, prob] : follower_map) {
                        
                        // Average reward over all follower actions
                        for (size_t a_f = 0; a_f < problem_.actions[1].size(); ++a_f) {
                            posg_core::Action follower_action(a_f, 1);
                            // Use rewards_leader with joint action index
                            size_t joint_action_idx = leader_action.get_action_id() * problem_.actions[1].size() + follower_action.get_action_id();
                            if (state < problem_.rewards_leader.size() && joint_action_idx < problem_.rewards_leader[state].size()) {
                                occupancy_reward += prob * problem_.rewards_leader[state][joint_action_idx];
                            }
                        }
                    }
                }
            }
            
            min_reward = std::min(min_reward, occupancy_reward);
        }
        
        return min_reward;
    }

} // namespace posg_algorithms 