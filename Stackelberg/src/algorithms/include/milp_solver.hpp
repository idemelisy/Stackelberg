#pragma once

#include "../../core/include/credible_set.hpp"
#include "../../core/include/credible_mdp.hpp"
#include "../../core/include/conditional_occupancy_state.hpp"
#include <vector>
#include <functional>
#include <memory>

// CPLEX includes
#include <ilcplex/ilocplex.h>

#include "../../common/logging.hpp"

namespace posg_algorithms {

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/**
 * @brief Big-M penalty constant used in MILP constraints (constraints 5–8).
 * @note Paper Appendix H recommends any value ≥ max |α|. 1e4 is sufficient for
 *       the benchmark instances shipped with the repository.
 */
constexpr double BIG_M_CONSTANT = 1e4;


    /**
     * @brief MILP Solver using CPLEX for the improve phase of PBVI
     * 
     * Implements the MILP formulation from the AAAI 2026 paper for computing
     * optimal leader decision rules in the improve phase of Point-Based Value Iteration.
     * 
     * From Paper: Section 5.2 - MILP-based Improve Phase
     * The MILP formulation computes σ_L* = argmax_{σ_L} min_{μ' ∈ C(μ)} V(μ')
     */
    class MILPSolver {
    private:
        // CPLEX environment and model
        std::unique_ptr<IloEnv> env_;
        std::unique_ptr<IloModel> model_;
        std::unique_ptr<IloCplex> cplex_;
        
        // MILP variables and constraints
        IloNumVarArray leader_action_vars_;      // σ_L(a_L|h_L) variables
        IloNumVarArray follower_action_vars_;    // σ_F(a_F|h_F) variables  
        IloNumVarArray occupancy_vars_;          // μ'(s, h_L, h_F) variables
        IloNumVarArray value_vars_;              // V(μ') variables
        
        // MILP parameters
        IloNumArray state_rewards_;              // R(s, a_L, a_F) rewards
        IloNumArray transition_probs_;           // P(s'|s, a_L, a_F) transition probabilities
        IloNumArray observation_probs_;          // P(z_L, z_F|s', a_L, a_F) observation probabilities
        
        double milp_time_limit_ = 10.0;
        
        /**
         * @brief Setup the MILP problem formulation
         * 
         * From Paper: The MILP formulation includes:
         * - Objective: max σ_L min μ' V(μ')
         * - Constraints: μ' ∈ C(μ) (credible set constraints)
         * - Constraints: σ_L is a valid decision rule
         * - Constraints: Value function constraints V(μ') = max_α α^T μ'
         */
        void setup_milp_problem(const posg_core::CredibleSet& credible_set,
                               const std::vector<std::vector<std::pair<std::vector<double>, std::vector<double>>>>& value_function_collection,
                               const posg_core::TransitionModel& transition_model,
                               const posg_core::ObservationModel& observation_model,
                               const std::function<double(int, const posg_core::Action&, const posg_core::Action&)>& reward_function_leader,
                               const std::function<double(int, const posg_core::Action&, const posg_core::Action&)>& reward_function_follower);
        
        /**
         * @brief Add credible set constraints
         * 
         * Ensures that the computed occupancy states belong to the credible set C(μ)
         */
        void add_credible_set_constraints(const posg_core::CredibleSet& credible_set);
        
        /**
         * @brief Add decision rule constraints
         * 
         * Ensures that σ_L and σ_F are valid probability distributions
         */
        void add_decision_rule_constraints();
        
        /**
         * @brief Add value function constraints
         * 
         * Links the value variables V(μ') to the alpha-vectors from the value function
         */
        void add_value_function_constraints(const std::vector<std::vector<std::pair<std::vector<double>, std::vector<double>>>>& value_function_collection);
        
        /**
         * @brief Add transition and observation constraints
         * 
         * Ensures consistency between occupancy states and the underlying POSG dynamics
         */
        void add_dynamics_constraints(const posg_core::TransitionModel& transition_model,
                                    const posg_core::ObservationModel& observation_model);
        
        /**
         * @brief Parse CPLEX solution into a leader decision rule
         */
        posg_core::LeaderDecisionRule parse_milp_solution(const void* milp_solution);
        
        /**
         * @brief Compute conditional occupancy state collections
         * 
         * Helper method to compute Co and Cf collections for the MILP formulation
         */
        std::pair<std::vector<posg_core::ConditionalOccupancyState>, 
                  std::vector<posg_core::ConditionalOccupancyState>> 
        compute_conditional_collections(const posg_core::CredibleSet& credible_set);

        /**
         * @brief Evaluate alpha-vector on a conditional occupancy state
         */
        double evaluate_alpha_vector(const std::vector<double>& alpha, 
                                     const posg_core::ConditionalOccupancyState& c);

        /**
         * @brief Compute immediate expected reward for a conditional occupancy state
         */
        double compute_immediate_reward(const posg_core::ConditionalOccupancyState& c, 
                                        const std::function<double(int, const posg_core::Action&, const posg_core::Action&)>& reward_func);

    public:
        MILPSolver(double milp_time_limit = 10.0);
        ~MILPSolver();
        
        /**
         * @brief Extract alpha-vectors from the MILP solution
         * 
         * From Paper: The alpha-vectors are computed from the dual variables
         * of the value function constraints V(μ') = max_α α^T μ'
         */
        std::vector<std::pair<std::vector<double>, std::vector<double>>> extract_alpha_vectors(
            const posg_core::CredibleSet& credible_set,
            const posg_core::LeaderDecisionRule& optimal_decision_rule,
            const std::vector<std::vector<std::pair<std::vector<double>, std::vector<double>>>>& value_function_collection,
            const posg_core::TransitionModel& transition_model,
            const posg_core::ObservationModel& observation_model);
        
        /**
         * @brief Solve the MILP for optimal leader decision rule
         * 
         * From Paper: σ_L* = argmax_{σ_L} min_{μ' ∈ C(μ)} V(μ')
         * 
         * @param credible_set The current credible set
         * @param value_function_collection Collection of alpha-vectors
         * @param transition_model Transition model for dynamics
         * @param observation_model Observation model for dynamics
         * @param reward_function_leader Leader reward function
         * @param reward_function_follower Follower reward function
         * @return Optimal leader decision rule
         */
        posg_core::LeaderDecisionRule solve_milp(
            const posg_core::CredibleSet& credible_set,
            const std::vector<std::vector<std::pair<std::vector<double>, std::vector<double>>>>& value_function_collection,
            const posg_core::TransitionModel& transition_model,
            const posg_core::ObservationModel& observation_model,
            const std::function<double(int, const posg_core::Action&, const posg_core::Action&)>& reward_function_leader,
            const std::function<double(int, const posg_core::Action&, const posg_core::Action&)>& reward_function_follower);
        
        /**
         * @brief Check if CPLEX is available and working
         */
        bool is_cplex_available() const;
        
        /**
         * @brief Get CPLEX version information
         */
        std::string get_cplex_version() const;
    };

} // namespace posg_algorithms 