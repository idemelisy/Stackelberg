#pragma once

#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>

// Core includes from Phase 1
#include "../../core/include/occupancy_state.hpp"
#include "../../core/include/conditional_occupancy_state.hpp"
#include "../../core/include/credible_set.hpp"
#include "../../core/include/credible_mdp.hpp"
#include "../../core/include/transition_model.hpp"
#include "../../core/include/observation_model.hpp"
#include "../../parser/include/posg_parser.hpp"

// MILP solver include
#include "milp_solver.hpp"

namespace posg_algorithms {

    /**
     * @brief Value function representation over occupancy states
     * 
     * Represents V: O -> R where O is the space of occupancy states.
     * Uses alpha-vector representation for efficient computation.
     * 
     * From Paper: Section 4.2 - Value Function Representation
     */
    class ValueFunction {
    public:
        ValueFunction() = default;
        
        /**
         * @brief Add an alpha-vector to the value function
         * @param alpha_vector The alpha-vector coefficients
         * @param action The associated leader action
         */
        void add_alpha_vector(const std::vector<double>& alpha_vector, 
                             const posg_core::Action& action);
        
        /**
         * @brief Get the value at a given occupancy state
         * @param occupancy_state The occupancy state to evaluate
         * @return The maximum value over all alpha-vectors
         */
        double get_value(const posg_core::OccupancyState& occupancy_state) const;
        
        /**
         * @brief Get the best action for a given occupancy state
         * @param occupancy_state The occupancy state to evaluate
         * @return The optimal leader action
         */
        posg_core::Action get_best_action(const posg_core::OccupancyState& occupancy_state) const;
        
        /**
         * @brief Get the number of alpha-vectors
         * @return Number of alpha-vectors in the value function
         */
        size_t size() const { return alpha_vectors_.size(); }
        
        /**
         * @brief Check if the value function is empty
         * @return true if no alpha-vectors
         */
        bool empty() const { return alpha_vectors_.empty(); }

    private:
        std::vector<std::vector<double>> alpha_vectors_;  // Alpha-vector coefficients
        std::vector<posg_core::Action> actions_;          // Associated leader actions
    };

    /**
     * @brief Credible MDP Solver implementing the reduction from Leader-Follower POSG
     * 
     * Implements the algorithmic components from the AAAI 2026 paper:
     * - Reduction to Credible MDP (Definition 4)
     * - Bellman recursion over credible sets
     * - Point-Based Value Iteration (PBVI)
     * - ε-optimal policy extraction
     */
    class CMDPSolver {
    public:
        CMDPSolver() = default;
        
        /**
         * @brief Constructor with POSG problem
         * @param problem The parsed POSG problem
         */
        explicit CMDPSolver(const posg_parser::POSGProblem& problem);
        
        /**
         * @brief Reduce Leader-Follower POSG to Credible MDP
         * 
         * From Paper: Definition 4 - Reduction to Credible MDP
         * 
         * @param problem The original POSG problem
         * @return The constructed Credible MDP
         */
        posg_core::CredibleMDP reduce_to_cmdp(const posg_parser::POSGProblem& problem);
        
        /**
         * @brief Bellman update for value function over credible sets
         * 
         * From Paper: Section 4.3 Bellman Recursion
         * 
         * @param value_function Current value function
         * @param occupancy_state Current occupancy state
         * @param leader_action Leader action to evaluate
         * @return Updated value for the state-action pair
         */
        double bellman_update(const ValueFunction& value_function,
                             const posg_core::OccupancyState& occupancy_state,
                             const posg_core::Action& leader_action);
        
        /**
         * @brief Point-Based Value Iteration (PBVI) algorithm with MILP-based improve phase
         * 
         * From Paper: Section 5 - Point-Based Value Iteration
         * 
         * @param initial_occupancy_states Set of initial occupancy states
         * @param max_iterations Maximum number of iterations
         * @param epsilon Convergence threshold
         * @return Converged value function
         */
        ValueFunction pbvi_with_milp(const std::vector<posg_core::OccupancyState>& initial_occupancy_states,
                                   size_t max_iterations = 1000,
                                   double epsilon = 1e-6);
        
        /**
         * @brief Extract ε-optimal policy from value function
         * 
         * From Paper: Section 4.5 - Policy Extraction
         * 
         * @param value_function The converged value function
         * @param epsilon Optimality threshold
         * @return Policy function mapping occupancy states to actions
         */
        std::function<posg_core::Action(const posg_core::OccupancyState&)> 
        extract_policy(const ValueFunction& value_function, double epsilon = 1e-6);
        
        /**
         * @brief Solve the complete Leader-Follower POSG
         * 
         * Main entry point that combines all algorithmic components
         * 
         * @param problem The POSG problem to solve
         * @param epsilon Optimality threshold
         * @return The optimal value function and policy
         */
        std::pair<ValueFunction, std::function<posg_core::Action(const posg_core::OccupancyState&)>>
        solve(const posg_parser::POSGProblem& problem, double epsilon = 1e-6);

    private:
        posg_parser::POSGProblem problem_;
        posg_core::TransitionModel transition_model_;
        posg_core::ObservationModel observation_model_;
        
        // MILP solver for improve phase
        MILPSolver milp_solver_;
        
        /**
         * @brief Compute successor occupancy states under joint actions
         * 
         * From Paper: Section 3.2 - Occupancy State Dynamics
         * 
         * @param occupancy_state Current occupancy state
         * @param leader_action Leader action
         * @param follower_action Follower action
         * @param leader_obs Leader observation
         * @param follower_obs Follower observation
         * @return Successor occupancy state
         */
        posg_core::OccupancyState compute_successor(
            const posg_core::OccupancyState& occupancy_state,
            const posg_core::Action& leader_action,
            const posg_core::Action& follower_action,
            const posg_core::Observation& leader_obs,
            const posg_core::Observation& follower_obs);
        
        /**
         * @brief Compute credible set for given occupancy state
         * 
         * From Paper: Definition 3 - Credible Set
         * 
         * @param occupancy_state The occupancy state
         * @return The credible set of reachable occupancy states
         */
        posg_core::CredibleSet compute_credible_set(
            const posg_core::OccupancyState& occupancy_state);
        
        /**
         * @brief Backproject value function to compute alpha-vectors
         * 
         * From Paper: Section 4.4 - Alpha-Vector Computation
         * 
         * @param value_function Current value function
         * @param occupancy_state Current occupancy state
         * @param leader_action Leader action
         * @return New alpha-vector for the state-action pair
         */
        std::vector<double> backproject(const ValueFunction& value_function,
                                       const posg_core::OccupancyState& occupancy_state,
                                       const posg_core::Action& leader_action);
        
        /**
         * @brief Compute credible reward for occupancy state and action
         * 
         * From Paper: R_C(μ, a_L) = min_{μ' ∈ C(μ)} R(μ', a_L)
         * 
         * @param occupancy_state The occupancy state
         * @param leader_action The leader action
         * @param credible_set The credible set
         * @return The credible reward
         */
        double compute_credible_reward(const posg_core::OccupancyState& occupancy_state,
                                      const posg_core::Action& leader_action,
                                      const posg_core::CredibleSet& credible_set);

    public:
        // ============================================================================
        // PBVI Expansion Phase (Algorithm 2) Implementation
        // ============================================================================
        
        /**
         * @brief Complete expansion phase for PBVI (Algorithm 2)
         * 
         * From Paper: Algorithm 2 - The expand phase for PBVI in M'
         * Samples finite set X' ⊂ X of representative credible sets through forward simulations
         * 
         * @param current_credible_sets Current set of credible sets Xt'
         * @param epsilon Distance threshold for filtering
         * @return Expanded set of credible sets Xt+1'
         */
        std::vector<posg_core::CredibleSet> expand_credible_sets(
            const std::vector<posg_core::CredibleSet>& current_credible_sets,
            double epsilon = 0.1);
        
        /**
         * @brief Refined τ(o, σL, σF) function to generate successor occupancy states
         * 
         * From Paper: τ(o, σL, σF) generates successor occupancy state under leader-follower policies
         * 
         * @param occupancy_state Current occupancy state o
         * @param leader_rule Leader decision rule σL
         * @param follower_rule Follower decision rule σF
         * @return Successor occupancy state
         */
        posg_core::OccupancyState tau_function(
            const posg_core::OccupancyState& occupancy_state,
            const posg_core::LeaderDecisionRule& leader_rule,
            const posg_core::FollowerDecisionRule& follower_rule);
        
        /**
         * @brief Compute conditional occupancy state collections Cx and Co
         * 
         * From Paper: Cx = {c(o',hF) | ∀o' ∈ x, hF ∈ HF} and Co = {c(o,hF) | ∀hF ∈ HF}
         * 
         * @param credible_set The credible set x
         * @param reference_occupancy Reference occupancy state o (for Co)
         * @return Pair of conditional occupancy state collections (Cx, Co)
         */
        std::pair<std::vector<posg_core::ConditionalOccupancyState>, 
                  std::vector<posg_core::ConditionalOccupancyState>> 
        compute_conditional_collections(const posg_core::CredibleSet& credible_set,
                                       const posg_core::OccupancyState& reference_occupancy);
        
        /**
         * @brief Apply ℓ1-distance filtering to avoid redundant credible sets
         * 
         * From Paper: Retain only occupancy states with ℓ1-distance > ε from current set
         * 
         * @param new_occupancy_state New occupancy state to check
         * @param existing_credible_set Existing credible set for comparison
         * @param epsilon Distance threshold
         * @return true if state should be retained (distance > ε)
         */
        bool should_retain_occupancy_state(
            const posg_core::OccupancyState& new_occupancy_state,
            const posg_core::CredibleSet& existing_credible_set,
            double epsilon);

        // ============================================================================
        // Uniform Continuity and ε-Approximation Logic (Theorem 5.3)
        // ============================================================================
        
        /**
         * @brief Find nearest neighbor credible set using Hausdorff distance
         * 
         * From Paper: vL(x) = vL(x') where x' = arg min dH(x, x')
         * 
         * @param target_credible_set Target credible set x
         * @param sampled_credible_sets Set of sampled credible sets X'
         * @return Nearest neighbor credible set x'
         */
        posg_core::CredibleSet nearest_neighbor(
            const posg_core::CredibleSet& target_credible_set,
            const std::vector<posg_core::CredibleSet>& sampled_credible_sets);
        
        /**
         * @brief Compute Hausdorff covering radius δ
         * 
         * From Paper: δ = supx∈X minx'∈X' dH(x, x')
         * 
         * @param all_credible_sets Complete set of credible sets X
         * @param sampled_credible_sets Sampled set of credible sets X'
         * @return Hausdorff covering radius δ
         */
        double compute_delta(const std::vector<posg_core::CredibleSet>& all_credible_sets,
                           const std::vector<posg_core::CredibleSet>& sampled_credible_sets);
        
        /**
         * @brief Compute exploitability bound ε ≤ m·ℓ·δ
         * 
         * From Paper: Theorem 5.3 - ε ≤ mℓδ where m = max{||rL||∞, ||rF||∞}
         * 
         * @param delta Hausdorff covering radius
         * @param horizon Planning horizon ℓ
         * @return Exploitability bound ε
         */
        double compute_exploitability_bound(double delta, int horizon);
        
        /**
         * @brief Approximate value function evaluation for unseen credible sets
         * 
         * From Paper: vL(x) ≈ vL(x') using nearest neighbor approximation
         * 
         * @param target_credible_set Target credible set x
         * @param value_function Current value function
         * @param sampled_credible_sets Sampled credible sets X'
         * @return Approximate value vL(x)
         */
        double approximate_value_function(
            const posg_core::CredibleSet& target_credible_set,
            const ValueFunction& value_function,
            const std::vector<posg_core::CredibleSet>& sampled_credible_sets);
        
        /**
         * @brief Enhanced PBVI with uniform continuity and ε-approximation
         * 
         * Complete implementation with theoretical guarantees from Theorem 5.3
         * 
         * @param initial_occupancy_states Initial occupancy states
         * @param max_iterations Maximum iterations
         * @param epsilon Convergence threshold
         * @param use_approximation Whether to use nearest-neighbor approximation
         * @return Value function and exploitability bound
         */
        std::pair<ValueFunction, double> pbvi_with_approximation(
            const std::vector<posg_core::OccupancyState>& initial_occupancy_states,
            size_t max_iterations = 1000,
            double epsilon = 1e-6,
            bool use_approximation = true);

    private:
    };

} // namespace posg_algorithms 