// milp_solver.cpp
// --------------
// Implements the MILP Solver using CPLEX for the improve phase of PBVI.
// What: Solves MILP formulations to compute optimal leader decision rules
// Why: Enables efficient computation of ε-optimal policies in leader-follower POSGs
// Fit: Used in the improve phase of Point-Based Value Iteration algorithm

/*
 * =============================================================================
 *  ε-Optimal Solutions for Leader–Follower POSGs (AAAI-26)
 *  ---------------------------------------------------------------------------
 *  Component   : MILPSolver – Improve Phase of Point-Based Value Iteration
 *  Paper Refs  :
 *      • Algorithm 3   – Improve phase for PBVI in M′
 *      • Theorem 5.1   – MILP characterisation of greedy σ*_L
 *      • Appendix H    – Dual-based α-vector extraction
 *
 *  High-level Overview
 *  -------------------
 *  Given a credible set x_t and the current collection L_{t+1} of value-vector
 *  sets, this module formulates the MILP in Eq. (5.1) using IBM CPLEX and
 *  returns the greedy leader decision rule σ*_L as well as (optionally) the
 *  next-stage α-vectors derived from the dual variables.
 *
 *  Outstanding Work (tracked with TODO tags)
 *  -----------------------------------------
 *  TODO(milp/immediate-reward):
 *      Populate ν_i(o) terms using `reward_function_[LF]` once the reward
 *      interface is deemed complete (see Lemma 4.2).
 *  TODO(milp/alpha-vectors):
 *      Implement `extract_alpha_vectors()` by querying the dual values of the
 *      max-constraints that tie V(μ′) to α^⊤ μ′ (Appendix H).
 *  TODO(pbvi/convergence):
 *      Ensure `CMDPSolver::pbvi_with_milp` halts when δ ≤ ε, propagating the
 *      exploitability bound of Theorem 5.3.
 *  TODO(cleanup/debug):
 *      Replace ad-hoc DEBUG_ILOARRAY_ACCESS with a proper logging facility or
 *      guard behind `#ifdef MILP_DEBUG`.
 *
 *  Edge-case Guarantees
 *  --------------------
 *  • Empty credible set ⇒ throws `std::runtime_error`.
 *  • Infeasible MILP   ⇒ returns uniform fallback `LeaderDecisionRule` keeping
 *                         downstream PBVI iterations alive.
 *
 * =============================================================================
 */
#include "../include/milp_solver.hpp"
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <map>
#include <tuple>
#include <vector>
#include <unordered_map>
#include "../../common/logging.hpp"

// Custom hash for std::tuple
namespace std {
    template <typename... Ts>
    struct hash<std::tuple<Ts...>> {
        size_t operator()(const std::tuple<Ts...>& t) const {
            size_t seed = 0;
            std::apply([&](const auto&... args) {
                ((seed ^= hash<Ts>{}(args) + 0x9e3779b9 + (seed << 6) + (seed >> 2)), ...);
            }, t);
            return seed;
        }
    };
}

// Add debug macros for array access
#ifdef MILP_DEBUG
#  define DEBUG_ILOARRAY_ACCESS(arr, idx)                                                         \
    std::cout << "[MILP_DEBUG] Accessing " #arr " at index " << (idx)                           \
              << " (size=" << (arr).getSize() << ")" << std::endl;
#else
#  define DEBUG_ILOARRAY_ACCESS(arr, idx) /* no-op */
#endif

namespace posg_algorithms {

    MILPSolver::MILPSolver() {
        try {
            // Initialize CPLEX environment
            env_ = std::make_unique<IloEnv>();
            model_ = std::make_unique<IloModel>(*env_);
            cplex_ = std::make_unique<IloCplex>(*model_);
            
            // Set CPLEX parameters for MILP solving
            cplex_->setParam(IloCplex::Param::Threads, 1);  // Single thread for reproducibility
            cplex_->setParam(IloCplex::Param::MIP::Tolerances::MIPGap, 1e-6);  // High precision
            cplex_->setParam(IloCplex::Param::TimeLimit, 300);  // 5 minute time limit
            
            // Initialize variable arrays
            leader_action_vars_ = IloNumVarArray(*env_);
            follower_action_vars_ = IloNumVarArray(*env_);
            occupancy_vars_ = IloNumVarArray(*env_);
            value_vars_ = IloNumVarArray(*env_);
            
            // Initialize parameter arrays
            state_rewards_ = IloNumArray(*env_);
            transition_probs_ = IloNumArray(*env_);
            observation_probs_ = IloNumArray(*env_);
            
        } catch (const IloException& e) {
            std::cerr << "CPLEX initialization error: " << e.getMessage() << std::endl;
            throw std::runtime_error("Failed to initialize CPLEX environment");
        }
    }

    MILPSolver::~MILPSolver() {
        // CPLEX environment will be automatically cleaned up by unique_ptr
    }

    posg_core::LeaderDecisionRule MILPSolver::solve_milp(
        const posg_core::CredibleSet& credible_set,
        const std::vector<std::vector<std::pair<std::vector<double>, std::vector<double>>>>& value_function_collection,
        const posg_core::TransitionModel& transition_model,
        const posg_core::ObservationModel& observation_model,
        const std::function<double(int, const posg_core::Action&, const posg_core::Action&)>& reward_function_leader,
        const std::function<double(int, const posg_core::Action&, const posg_core::Action&)>& reward_function_follower) {
        
        try {
            // Setup the MILP problem
            setup_milp_problem(credible_set, value_function_collection, 
                             transition_model, observation_model,
                             reward_function_leader, reward_function_follower);
            
            // Solve the MILP
            if (!cplex_->solve()) {
                std::cerr << "CPLEX failed to solve MILP: " << cplex_->getStatus() << std::endl;
                // Return a default decision rule if solving fails
                return posg_core::LeaderDecisionRule(credible_set.get_timestep());
            }
            
            // Parse the solution
            return parse_milp_solution(nullptr);  // CPLEX solution is stored in the model
            
        } catch (const IloException& e) {
            std::cerr << "CPLEX solving error: " << e.getMessage() << std::endl;
            // Return a default decision rule on error
            return posg_core::LeaderDecisionRule(credible_set.get_timestep());
        }
    }

    void MILPSolver::setup_milp_problem(
        const posg_core::CredibleSet& credible_set,
        const std::vector<std::vector<std::pair<std::vector<double>, std::vector<double>>>>& value_function_collection,
        const posg_core::TransitionModel& transition_model,
        const posg_core::ObservationModel& observation_model,
        const std::function<double(int, const posg_core::Action&, const posg_core::Action&)>& reward_function_leader,
        const std::function<double(int, const posg_core::Action&, const posg_core::Action&)>& reward_function_follower) {
        std::cout << "[MILP DEBUG] Beginning MILP setup..." << std::endl;
        std::cout << "[MILP DEBUG] Occupancy size: " << credible_set.get_occupancy_states().size() << std::endl;
        std::cout << "[MILP DEBUG] Num states: " << transition_model.get_num_states() << std::endl;
        std::cout << "[MILP DEBUG] Num leader actions: " << transition_model.get_num_leader_actions() << std::endl;
        std::cout << "[MILP DEBUG] Num follower actions: " << transition_model.get_num_follower_actions() << std::endl;
        
        // Clear previous variables
        leader_action_vars_.clear();
        follower_action_vars_.clear();
        occupancy_vars_.clear();
        value_vars_.clear();
        
        // Create a new model to avoid multiple objectives
        model_ = std::make_unique<IloModel>(*env_);
        cplex_ = std::make_unique<IloCplex>(*model_);
        
        // Set CPLEX parameters for MILP solving
        cplex_->setParam(IloCplex::Param::Threads, 1);  // Single thread for reproducibility
        cplex_->setParam(IloCplex::Param::MIP::Tolerances::MIPGap, 1e-6);  // High precision
        cplex_->setParam(IloCplex::Param::TimeLimit, 300);  // 5 minute time limit
        
        // From Paper: MILP formulation for greedy leader decision rule
        // Variables: σL(aL|hL), qL^F(o,σL), qF^F(o,σL), wF^F(c,zF,α)
        
        // Get the first occupancy state as the reference state o
        const auto& occupancy_states = credible_set.get_occupancy_states();
        if (occupancy_states.empty()) {
            throw std::runtime_error("Credible set is empty");
        }
        const auto& reference_occupancy = *occupancy_states.begin();
        
        // Compute collections Cx and Co as defined in the paper
        auto [Cx, Co] = compute_conditional_collections(credible_set);
        
        // Create variables for leader decision rule σL(aL|hL)
        // These are probability variables: 0 ≤ σL(aL|hL) ≤ 1
        int var_index = 0;
        std::map<std::pair<posg_core::AgentHistory, int>, int> sigma_L_vars;
        
        for (const auto& occupancy_state : occupancy_states) {
            auto leader_histories = occupancy_state.get_leader_history_marginal();
            for (const auto& [leader_history, _] : leader_histories) {
                for (int action_id = 0; action_id < transition_model.get_num_leader_actions(); ++action_id) {
                    std::string var_name = "sigma_L_" + std::to_string(var_index);
                    leader_action_vars_.add(IloNumVar(*env_, 0.0, 1.0, ILOFLOAT, var_name.c_str()));
                    DEBUG_ILOARRAY_ACCESS(leader_action_vars_, var_index);
                    sigma_L_vars[{leader_history, action_id}] = var_index++;
                }
            }
        }
        
        // Create variables for leader action-value function qL^F(o,σL)
        std::map<posg_core::OccupancyState, int> qL_vars;
        var_index = 0;
        for (const auto& occupancy_state : occupancy_states) {
            std::string var_name = "qL_" + std::to_string(var_index);
            value_vars_.add(IloNumVar(*env_, -1000.0, 1000.0, ILOFLOAT, var_name.c_str()));
            DEBUG_ILOARRAY_ACCESS(value_vars_, var_index);
            qL_vars[occupancy_state] = var_index++;
        }
        
        // Create variables for follower action-value function qF^F(o,σL)
        std::map<posg_core::OccupancyState, int> qF_vars;
        for (const auto& occupancy_state : occupancy_states) {
            std::string var_name = "qF_" + std::to_string(var_index);
            value_vars_.add(IloNumVar(*env_, -1000.0, 1000.0, ILOFLOAT, var_name.c_str()));
            DEBUG_ILOARRAY_ACCESS(value_vars_, var_index);
            qF_vars[occupancy_state] = var_index++;
        }
        
        // Create variables for follower choice indicators wF^F(c,zF,α)
        // These are binary variables indicating which alpha-vector the follower chooses
        std::unordered_map<std::tuple<int, int, size_t>, IloNumVar> wF_vars; // (conditional_idx, zF, alpha_idx) -> var
        int conditional_idx = 0;
        for (const auto& conditional_state : Cx) {
            for (int zF = 0; zF < observation_model.get_num_follower_observations(); ++zF) {
                for (size_t alpha_idx = 0; alpha_idx < value_function_collection.size(); ++alpha_idx) {
                    std::string var_name = "wF_" + std::to_string(var_index);
                    IloNumVar var = IloNumVar(*env_, 0.0, 1.0, ILOBOOL, var_name.c_str());
                    follower_action_vars_.add(var);
                    wF_vars[{conditional_idx, zF, alpha_idx}] = var;
                    var_index++;
                }
            }
            conditional_idx++;
        }
        
        // Create variables for qL^F(c,σL,zF) and qF^F(c,σL,zF) for conditional states
        std::unordered_map<std::tuple<int, int>, IloNumVar> qL_cond_vars, qF_cond_vars; // (conditional_idx, zF) -> var
        
        conditional_idx = 0;
        for (const auto& conditional_state : Cx) {
            for (int zF = 0; zF < observation_model.get_num_follower_observations(); ++zF) {
                std::string var_name = "qF_cond_" + std::to_string(var_index);
                IloNumVar var = IloNumVar(*env_, -1000.0, 1000.0, ILOFLOAT, var_name.c_str());
                qF_cond_vars[{conditional_idx, zF}] = var;
                var_index++;
            }
            conditional_idx++;
        }
        
        conditional_idx = 0;
        for (const auto& conditional_state : Co) {
            for (int zF = 0; zF < observation_model.get_num_follower_observations(); ++zF) {
                std::string var_name = "qL_cond_" + std::to_string(var_index);
                IloNumVar var = IloNumVar(*env_, -1000.0, 1000.0, ILOFLOAT, var_name.c_str());
                qL_cond_vars[{conditional_idx, zF}] = var;
                var_index++;
            }
            conditional_idx++;
        }
        
        // Objective function: maximize leader's expected value
        // From Paper: max Σ_s α(s) · μ(s) subject to constraints
        IloExpr qL_expr(*env_);
        
        // Get the first occupancy state to determine number of states
        const auto& states_in_set = credible_set.get_occupancy_states();
        if (!states_in_set.empty()) {
            const auto& first_occupancy = *states_in_set.begin();
            auto state_marginal = first_occupancy.get_state_marginal();
            for (const auto& [state, prob] : state_marginal) {
                auto key = std::make_tuple(0, state);
                if (qL_cond_vars.find(key) != qL_cond_vars.end()) {
                    qL_expr += qL_cond_vars.at(key) * prob;
                }
            }
        }
        
        // ---------------------------------------------------------------------
        // Immediate reward term  ν_L(o)  (Lemma 4.2 / Eq. 16)
        // ---------------------------------------------------------------------
        // We cache per-occupancy reward because the same state can appear in
        // several constraint instances.
        static std::unordered_map<const posg_core::OccupancyState*, double> occ_reward_cache_L;
        const posg_core::OccupancyState* occ_ptr = &reference_occupancy;
        double immediate_reward_L = 0.0;
        if (occ_reward_cache_L.find(occ_ptr) != occ_reward_cache_L.end()) {
            immediate_reward_L = occ_reward_cache_L.at(occ_ptr);
        } else {
            auto follower_marginal_tmp = reference_occupancy.get_follower_history_marginal();
            for (const auto& [follower_history, probFH] : follower_marginal_tmp) {
                auto conditional_tmp = reference_occupancy.conditional_decompose(follower_history);
                if (!conditional_tmp.is_valid(true)) { continue; }
                immediate_reward_L += probFH * compute_immediate_reward(conditional_tmp, reward_function_leader);
            }
            occ_reward_cache_L[occ_ptr] = immediate_reward_L;
        }
        // The master objective subtracts ν_L(o) to satisfy the equality form
        // qL(o,σ) - ν_L(o) - Σ Pr·qL(c,zF) = 0  (Eq. 17)
        qL_expr -= immediate_reward_L;
        
        model_->add(IloMaximize(*env_, qL_expr));
        qL_expr.end();
        
        // Add constraints from the paper
        
        // Constraint 1: qF^F(o,σL) ≥ qF^F(o',σL), ∀o' ∈ x (follower rationality)
        for (const auto& [occupancy_state, qF_idx] : qF_vars) {
            if (occupancy_state != reference_occupancy) {
                model_->add(value_vars_[qF_vars[reference_occupancy]] >= value_vars_[qF_idx]);
            }
        }
        
        // Constraint 2: qL^F(o,σL) = νL(o) + Σ_hF,zF Pr(hF|o) · qL^F(c(o,hF),σL,zF)
        IloExpr qL_expr_constraint2(*env_);
        qL_expr_constraint2 += value_vars_[qL_vars[reference_occupancy]];
        
        // Add immediate reward term  ν_L(o)
        qL_expr_constraint2 -= immediate_reward_L;
        
        // Add discounted future value term
        auto follower_marginal = reference_occupancy.get_follower_history_marginal();
        int cond_idx = 0;
        for (const auto& [follower_history, prob] : follower_marginal) {
            auto conditional = reference_occupancy.conditional_decompose(follower_history);
            if (conditional.is_valid(true)) {
                for (int zF = 0; zF < observation_model.get_num_follower_observations(); ++zF) {
                    double obs_prob = observation_model.get_follower_observation_probability(0, posg_core::Action(0, 1), posg_core::Observation(zF, 1));
                    auto key = std::make_tuple(cond_idx, zF);
                    if (qL_cond_vars.find(key) != qL_cond_vars.end()) {
                        qL_expr_constraint2 -= prob * obs_prob * qL_cond_vars.at(key);
                    }
                }
                cond_idx++;
            }
        }
        model_->add(qL_expr_constraint2 == 0.0);
        qL_expr_constraint2.end();
        
        // Constraint 3: qF^F(o',σL) = νF(o') + Σ_hF,zF Pr(hF|o') · qF^F(c(o',hF),σL,zF), ∀o' ∈ x
        for (const auto& [occupancy_state, qF_idx] : qF_vars) {
            IloExpr qF_expr(*env_);
            qF_expr += value_vars_[qF_idx];
            
            // ------------------------------------------------------------------
            // Immediate reward term  ν_F(o')
            // ------------------------------------------------------------------
            static std::unordered_map<const posg_core::OccupancyState*, double> occ_reward_cache_F;
            double immediate_reward_F = 0.0;
            const posg_core::OccupancyState* occ_ptr_F = &occupancy_state;
            if (occ_reward_cache_F.find(occ_ptr_F) != occ_reward_cache_F.end()) {
                immediate_reward_F = occ_reward_cache_F.at(occ_ptr_F);
            } else {
                auto follower_marginal_F = occupancy_state.get_follower_history_marginal();
                for (const auto& [fh_F, probFH_F] : follower_marginal_F) {
                    auto conditional_F = occupancy_state.conditional_decompose(fh_F);
                    if (!conditional_F.is_valid(true)) { continue; }
                    immediate_reward_F += probFH_F * compute_immediate_reward(conditional_F, reward_function_follower);
                }
                occ_reward_cache_F[occ_ptr_F] = immediate_reward_F;
            }
            qF_expr -= immediate_reward_F;
            
            // Add discounted future value term
            auto follower_marginal = occupancy_state.get_follower_history_marginal();
            int cond_idx = 0;
            for (const auto& [follower_history, prob] : follower_marginal) {
                auto conditional = occupancy_state.conditional_decompose(follower_history);
                if (conditional.is_valid(true)) {
                    for (int zF = 0; zF < observation_model.get_num_follower_observations(); ++zF) {
                        double obs_prob = observation_model.get_follower_observation_probability(0, posg_core::Action(0, 1), posg_core::Observation(zF, 1));
                        auto key = std::make_tuple(cond_idx, zF);
                        if (qF_cond_vars.find(key) != qF_cond_vars.end()) {
                            qF_expr -= prob * obs_prob * qF_cond_vars.at(key);
                        }
                    }
                    cond_idx++;
                }
            }
            model_->add(qF_expr == 0.0);
            qF_expr.end();
        }
        
        // Constraint 4: Σ_α∈F wF^F(c,zF,α) = 1, ∀c ∈ Cx, zF ∈ ZF
        conditional_idx = 0;
        for (const auto& conditional_state : Cx) {
            for (int zF = 0; zF < observation_model.get_num_follower_observations(); ++zF) {
                IloExpr sum_expr(*env_);
                for (size_t alpha_idx = 0; alpha_idx < value_function_collection.size(); ++alpha_idx) {
                    if (wF_vars.find({conditional_idx, zF, alpha_idx}) != wF_vars.end()) {
                        sum_expr += wF_vars.at({conditional_idx, zF, alpha_idx});
                    }
                }
                model_->add(sum_expr == 1.0);
                sum_expr.end();
            }
            conditional_idx++;
        }
        
        // Constraint 5: qF^F(c,σL,zF) ≥ αF(τ̃zF(c,σL,aF)), ∀c ∈ Cx, aF ∈ AF, zF ∈ ZF, α ∈ F
        // Constraint 6: qF^F(c,σL,zF) ≤ αF(τ̃zF(c,σL,aF)) + M·(1-wF^F(c,zF,α)), ∀c ∈ Cx, aF ∈ AF, zF ∈ ZF, α ∈ F
        constexpr double M = BIG_M_CONSTANT; // Use global constant
        
        conditional_idx = 0;
        for (const auto& conditional_state : Cx) {
            for (int zF = 0; zF < observation_model.get_num_follower_observations(); ++zF) {
                for (int aF = 0; aF < transition_model.get_num_follower_actions(); ++aF) {
                    for (size_t alpha_idx = 0; alpha_idx < value_function_collection.size(); ++alpha_idx) {
                        // Compute αF(τ̃zF(c,σL,aF)) - simplified for now
                        // From Paper: αF(τ̃zF(c,σL,aF)) = Σ_s τ̃zF(c,σL,aF)(s) · αF(s)
                        // For now, use the first alpha-vector pair in the set
                        if (!value_function_collection.empty() && !value_function_collection[alpha_idx].empty()) {
                            const auto& alpha_pair = value_function_collection[alpha_idx][0];
                            double alpha_value = evaluate_alpha_vector(alpha_pair.second, conditional_state); // Follower alpha
                        
                            auto qF_key = std::make_tuple(conditional_idx, zF);
                            auto wF_key = std::make_tuple(conditional_idx, zF, alpha_idx);
                            if (qF_cond_vars.find(qF_key) != qF_cond_vars.end() &&
                                wF_vars.find(wF_key) != wF_vars.end()) {
                                
                                // Constraint 5: qF^F(c,σL,zF) ≥ αF(τ̃zF(c,σL,aF))
                                model_->add(qF_cond_vars.at(qF_key) >= alpha_value);
                                
                                // Constraint 6: qF^F(c,σL,zF) ≤ αF(τ̃zF(c,σL,aF)) + M·(1-wF^F(c,zF,α))
                                model_->add(qF_cond_vars.at(qF_key) <= 
                                          alpha_value + M * (1.0 - wF_vars.at(wF_key)));
                            }
                        }
                    }
                }
            }
            conditional_idx++;
        }
        
        // Constraint 7: qL^F(c,σL,zF) ≥ αL(τ̃zF(c,σL,aF)) - M·(1-wF^F(c,zF,α)), ∀c ∈ Co, aF ∈ AF, zF ∈ ZF, α ∈ F
        // Constraint 8: qL^F(c,σL,zF) ≤ αL(τ̃zF(c,σL,aF)) + M·(1-wF^F(c,zF,α)), ∀c ∈ Co, aF ∈ AF, zF ∈ ZF, α ∈ F
        conditional_idx = 0;
        for (const auto& conditional_state : Co) {
            for (int zF = 0; zF < observation_model.get_num_follower_observations(); ++zF) {
                for (int aF = 0; aF < transition_model.get_num_follower_actions(); ++aF) {
                    for (size_t alpha_idx = 0; alpha_idx < value_function_collection.size(); ++alpha_idx) {
                        // Compute αL(τ̃zF(c,σL,aF)) - simplified for now
                        // From Paper: αL(τ̃zF(c,σL,aF)) = Σ_s τ̃zF(c,σL,aF)(s) · αL(s)
                        // For now, use the first alpha-vector pair in the set
                        if (!value_function_collection.empty() && !value_function_collection[alpha_idx].empty()) {
                            const auto& alpha_pair = value_function_collection[alpha_idx][0];
                            double alpha_value = evaluate_alpha_vector(alpha_pair.first, conditional_state); // Leader alpha
                        
                        auto qL_key = std::make_tuple(conditional_idx, zF);
                        auto wF_key = std::make_tuple(conditional_idx, zF, alpha_idx);
                        if (qL_cond_vars.find(qL_key) != qL_cond_vars.end() &&
                            wF_vars.find(wF_key) != wF_vars.end()) {
                            
                            // Constraint 7: qL^F(c,σL,zF) ≥ αL(τ̃zF(c,σL,aF)) - M·(1-wF^F(c,zF,α))
                            model_->add(qL_cond_vars.at(qL_key) >= 
                                      alpha_value - M * (1.0 - wF_vars.at(wF_key)));
                            
                            // Constraint 8: qL^F(c,σL,zF) ≤ αL(τ̃zF(c,σL,aF)) + M·(1-wF^F(c,zF,α))
                            model_->add(qL_cond_vars.at(qL_key) <= 
                                      alpha_value + M * (1.0 - wF_vars.at(wF_key)));
                        }
                        }
                    }
                }
            }
            conditional_idx++;
        }
        
        // Add probability constraints for leader decision rule
        // Σ_aL σL(aL|hL) = 1, ∀hL
        std::map<posg_core::AgentHistory, std::vector<int>> history_to_vars;
        for (const auto& [key, var_idx] : sigma_L_vars) {
            history_to_vars[key.first].push_back(var_idx);
        }
        
        for (const auto& [history, var_indices] : history_to_vars) {
            IloExpr prob_sum(*env_);
            for (int var_idx : var_indices) {
                prob_sum += leader_action_vars_[var_idx];
            }
            model_->add(prob_sum == 1.0);
            prob_sum.end();
        }
        LOG_DEBUG("qL_cond_vars size: " << qL_cond_vars.size());
        LOG_DEBUG("qF_cond_vars size: " << qF_cond_vars.size());
        // After all constraints are added (right before solving)
        std::cout << "[MILP DEBUG] Number of constraints added: " << cplex_->getNrows() << std::endl;
        // Optionally, write the LP file for CPLEX inspection
        cplex_->exportModel("milp_debug.lp");
    }

    // These methods are now implemented directly in setup_milp_problem
    void MILPSolver::add_credible_set_constraints(const posg_core::CredibleSet& credible_set) {
        // Implemented directly in setup_milp_problem
    }

    void MILPSolver::add_decision_rule_constraints() {
        // Implemented directly in setup_milp_problem
    }

    void MILPSolver::add_value_function_constraints(
        const std::vector<std::vector<std::pair<std::vector<double>, std::vector<double>>>>& value_function_collection) {
        // Implemented directly in setup_milp_problem
    }

    void MILPSolver::add_dynamics_constraints(
        const posg_core::TransitionModel& transition_model,
        const posg_core::ObservationModel& observation_model) {
        // Implemented directly in setup_milp_problem
    }

    posg_core::LeaderDecisionRule MILPSolver::parse_milp_solution(const void* milp_solution) {
        posg_core::LeaderDecisionRule decision_rule(0);  // Default timestep
        
        try {
            // TODO: Parse CPLEX solution values into leader decision rule
        // From Paper: Extract σ_L(a_L|h_L) values from the MILP solution
            // Extract σ_L(a_L|h_L) values from leader_action_vars_
            
            // For now, create a uniform decision rule
            int var_index = 0;
            for (int i = 0; i < leader_action_vars_.getSize(); i += 2) {
                if (i + 1 < leader_action_vars_.getSize()) {
                    DEBUG_ILOARRAY_ACCESS(leader_action_vars_, i);
                    double prob1 = cplex_->getValue(leader_action_vars_[i]);
                    DEBUG_ILOARRAY_ACCESS(leader_action_vars_, i + 1);
                    double prob2 = cplex_->getValue(leader_action_vars_[i + 1]);
                    
                    // Create dummy histories and actions for now
                    posg_core::AgentHistory dummy_history(0);
                    posg_core::Action action1(var_index, 0);
                    posg_core::Action action2(var_index + 1, 0);
                    
                    decision_rule.set_action_probability(dummy_history, action1, prob1);
                    decision_rule.set_action_probability(dummy_history, action2, prob2);
                    
                    var_index += 2;
                }
            }
            
        } catch (const IloException& e) {
            std::cerr << "Error parsing CPLEX solution: " << e.getMessage() << std::endl;
        }
        
        return decision_rule;
    }

    std::vector<std::pair<std::vector<double>, std::vector<double>>> MILPSolver::extract_alpha_vectors(
        const posg_core::CredibleSet& credible_set,
        const posg_core::LeaderDecisionRule& optimal_decision_rule,
        const std::vector<std::vector<std::pair<std::vector<double>, std::vector<double>>>>& value_function_collection,
        const posg_core::TransitionModel& transition_model,
        const posg_core::ObservationModel& observation_model) {
        
        // TODO: Extract alpha-vectors from dual variables of value function constraints
        // From Paper: The alpha-vectors are computed from the dual variables
        // of the value function constraints V(μ') = max_α α^T μ'
        // From Paper: The alpha-vectors are computed from the dual variables
        // of the value function constraints V(μ') = max_α α^T μ'
        
        // For now, return empty alpha-vectors
        return {};
    }

    std::pair<std::vector<posg_core::ConditionalOccupancyState>, 
              std::vector<posg_core::ConditionalOccupancyState>> 
    MILPSolver::compute_conditional_collections(const posg_core::CredibleSet& credible_set) {
        std::vector<posg_core::ConditionalOccupancyState> Co, Cf;
        
        // Find the first occupancy state in the set (if any)
        const auto& occupancy_states = credible_set.get_occupancy_states();
        if (occupancy_states.empty()) {
            // TODO: Handle empty credible set case robustly
            // From Paper: Empty credible sets require special handling in the MILP formulation
            return { {}, {} };
        }
        const auto& first_occupancy = *occupancy_states.begin();
        
        // Compute Co: conditional states for the first occupancy state and all follower histories
        // (This is a simplification - in practice, you might need more sophisticated logic)
        auto follower_marginal = first_occupancy.get_follower_history_marginal();
        for (const auto& [follower_history, _] : follower_marginal) {
            auto conditional = first_occupancy.conditional_decompose(follower_history);
            if (conditional.is_valid(true) && !conditional.get_state_marginal().empty()) {
                Co.push_back(conditional);
            }
        }
        
        // Compute Cf: conditional states for all occupancy states and the first follower history
        // (This is also a simplification)
        auto first_follower_history = first_occupancy.get_follower_history_marginal().begin()->first;
        for (const auto& occupancy_state : occupancy_states) {
            auto conditional = occupancy_state.conditional_decompose(first_follower_history);
            if (conditional.is_valid(true) && !conditional.get_state_marginal().empty()) {
                Cf.push_back(conditional);
            }
        }
        
        return {Co, Cf};
    }

    bool MILPSolver::is_cplex_available() const {
        return env_ != nullptr && cplex_ != nullptr;
    }

    std::string MILPSolver::get_cplex_version() const {
        try {
            if (env_) {
                return "CPLEX " + std::string(env_->getVersion());
            }
        } catch (const IloException& e) {
            return "CPLEX version unknown: " + std::string(e.getMessage());
        }
        return "CPLEX not initialized";
    }
    
    double MILPSolver::evaluate_alpha_vector(const std::vector<double>& alpha, const posg_core::ConditionalOccupancyState& c) {
        double alpha_value = 0.0;
        for (const auto& [state, prob] : c.get_state_marginal()) {
            if (state < alpha.size()) {
                alpha_value += prob * alpha[state];
            }
        }
        return alpha_value;
    }

    double MILPSolver::compute_immediate_reward(const posg_core::ConditionalOccupancyState& c, const std::function<double(int, const posg_core::Action&, const posg_core::Action&)>& reward_func) {
        double reward = 0.0;
        for (const auto& [state, prob] : c.get_state_marginal()) {
            reward += prob * reward_func(0, posg_core::Action(0, 1), posg_core::Action(0, 1)); // Placeholder for aF
        }
        return reward;
    }
    
} // namespace posg_algorithms 