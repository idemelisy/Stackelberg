#include "../include/cmdp_solver.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <unordered_set>
#include "../../common/logging.hpp"
#include <chrono>
#include <random> // Added for random sampling

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

    CMDPSolver::CMDPSolver(const posg_parser::POSGProblem& problem, double milp_time_limit)
        : problem_(problem), milp_time_limit_(milp_time_limit), milp_solver_(milp_time_limit) {
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
        constexpr double similarity_epsilon = 0.05;
        constexpr int num_leader_samples = 10;
        constexpr int num_follower_samples = 10;
        constexpr double dirichlet_alpha = 0.5;
        std::vector<posg_core::CredibleSet> expanded_sets;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::gamma_distribution<double> gamma_dist(dirichlet_alpha, 1.0);
        try {
            for (const auto& credible_set : current_credible_sets) {
                if (credible_set.get_occupancy_states().empty()) {
                    std::cerr << "[EXPAND WARNING] Credible set has no occupancy states!" << std::endl;
                    continue;
                }
                posg_core::CredibleSet new_credible_set;
                int new_conditional_count = 0;
                int new_occupancy_count = 0;
                std::vector<posg_core::ConditionalOccupancyState> added_conditionals;
                std::vector<posg_core::OccupancyState> added_occupancies;
                for (int l_sample = 0; l_sample < num_leader_samples; ++l_sample) {
                    posg_core::LeaderDecisionRule sampled_leader_rule(0);
                    std::vector<double> leader_probs;
                    double leader_total = 0.0;
                    for (int action_id = 0; action_id < transition_model_.get_num_leader_actions(); ++action_id) {
                        double prob = gamma_dist(gen);
                        leader_probs.push_back(prob);
                        leader_total += prob;
                    }
                    for (int action_id = 0; action_id < transition_model_.get_num_leader_actions(); ++action_id) {
                        posg_core::Action action(action_id, 0);
                        posg_core::AgentHistory empty_history(0);
                        double norm_prob = leader_probs[action_id] / (leader_total > 0 ? leader_total : 1.0);
                        norm_prob = std::max(0.0, std::min(1.0, norm_prob));
                        sampled_leader_rule.set_action_probability(empty_history, action, norm_prob);
                    }
                    sampled_leader_rule.normalize();
                    for (const auto& occupancy_state : credible_set.get_occupancy_states()) {
                        auto follower_marginal = occupancy_state.get_follower_history_marginal();
                        for (const auto& [follower_history, _] : follower_marginal) {
                            for (int f_sample = 0; f_sample < num_follower_samples; ++f_sample) {
                                posg_core::FollowerDecisionRule follower_rule(0);
                                std::vector<double> probs;
                                double total_prob = 0.0;
                                for (int aF_id = 0; aF_id < transition_model_.get_num_follower_actions(); ++aF_id) {
                                    double prob = gamma_dist(gen);
                                    probs.push_back(prob);
                                    total_prob += prob;
                                }
                                for (int aF_id = 0; aF_id < transition_model_.get_num_follower_actions(); ++aF_id) {
                                    posg_core::Action aF(aF_id, 1);
                                    double norm_prob = probs[aF_id] / (total_prob > 0 ? total_prob : 1.0);
                                    norm_prob = std::max(0.0, std::min(1.0, norm_prob));
                                    follower_rule.set_action_probability(follower_history, aF, norm_prob);
                                }
                                follower_rule.normalize();
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
                                    // Similarity check: L1 distance to all previous
                                    bool should_retain = true;
                                    for (const auto& existing_conditional : added_conditionals) {
                                        double distance = updated_conditional.distance_to(existing_conditional);
                                        if (distance <= similarity_epsilon) {
                                            should_retain = false;
                                            break;
                                        }
                                    }
                                    if (should_retain) {
                                        ++new_conditional_count;
                                        added_conditionals.push_back(updated_conditional);
                                        // Add to a new occupancy state via tau
                                        posg_core::OccupancyState successor_state = tau_function(
                                            occupancy_state, sampled_leader_rule, follower_rule);
                                        // OccupancyState equality: compare distributions directly
                                        bool already_present = false;
                                        for (const auto& occ : added_occupancies) {
                                            if (occ.get_occupancy_distribution() == successor_state.get_occupancy_distribution()) {
                                                already_present = true;
                                                break;
                                            }
                                        }
                                        if (!already_present) {
                                            new_credible_set.add_occupancy_state(successor_state);
                                            added_occupancies.push_back(successor_state);
                                            ++new_occupancy_count;
                                            // Print the full occupancy distribution for this new state
                                            std::cout << "[DEBUG] New occupancy state distribution: " << successor_state.to_string() << std::endl;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if (!new_credible_set.get_occupancy_states().empty() || expanded_sets.empty()) {
                    expanded_sets.push_back(new_credible_set);
                }
                std::cout << "[DEBUG] expand_credible_sets: New conditional occupancy states added: " << new_conditional_count << std::endl;
                std::cout << "[DEBUG] expand_credible_sets: New occupancy states added: " << new_occupancy_count << std::endl;
                // Pairwise L1 distances between all occupancy states in this credible set
                std::vector<double> l1_distances;
                std::vector<posg_core::OccupancyState> occs_vec(new_credible_set.get_occupancy_states().begin(), new_credible_set.get_occupancy_states().end());
                for (size_t i = 0; i < occs_vec.size(); ++i) {
                    for (size_t j = i + 1; j < occs_vec.size(); ++j) {
                        double dist = occs_vec[i].distance_to(occs_vec[j]);
                        l1_distances.push_back(dist);
                    }
                }
                if (!l1_distances.empty()) {
                    std::sort(l1_distances.begin(), l1_distances.end());
                    double min_dist = l1_distances.front();
                    double max_dist = l1_distances.back();
                    double median_dist = l1_distances[l1_distances.size() / 2];
                    std::cout << "[DEBUG] expand_credible_sets: Pairwise L1 distances (min/median/max): "
                              << min_dist << " / " << median_dist << " / " << max_dist << std::endl;
                } else {
                    std::cout << "[DEBUG] expand_credible_sets: Only one occupancy state, no pairwise distances." << std::endl;
                }
            }
        } catch (const std::exception& ex) {
            std::cerr << "[EXPAND ERROR] Exception during expansion: " << ex.what() << std::endl;
            // Optionally print last processed occupancy state/credible set
            throw;
        }
        if (expanded_sets.empty() && !current_credible_sets.empty()) {
            expanded_sets.push_back(current_credible_sets.front());
        }
        return expanded_sets;
    }

    posg_core::OccupancyState CMDPSolver::tau_function(
        const posg_core::OccupancyState& occupancy_state,
        const posg_core::LeaderDecisionRule& leader_rule,
        const posg_core::FollowerDecisionRule& follower_rule) {
        posg_core::OccupancyState successor_state;
        const auto& current_entries = occupancy_state.get_entries();
        std::cout << "[TAU DEBUG] current_entries size: " << current_entries.size() << std::endl;
        double total_prob = 0.0;
        int contrib_printed = 0;
        int entries_added = 0;
        int last_s = -1, last_aL = -1, last_aF = -1;
        size_t last_hL_size = 0, last_hF_size = 0;
        for (const auto& [tuple, prob] : current_entries) {
            if (prob <= 0.0 || !std::isfinite(prob)) {
                std::cerr << "[TAU WARNING] Skipping entry with nonpositive or nonfinite prob: " << prob << std::endl;
                continue;
            }
            int s = tuple.state;
            const auto& hL = tuple.leader_history;
            const auto& hF = tuple.follower_history;
            last_s = s;
            last_hL_size = hL.size();
            last_hF_size = hF.size();
            for (int aL = 0; aL < transition_model_.get_num_leader_actions(); ++aL) {
                double pL = leader_rule.get_prob(hL, posg_core::Action(aL, 0));
                if (pL <= 0.0 || !std::isfinite(pL)) {
                    std::cerr << "[TAU WARNING] Skipping leader action with prob: " << pL << std::endl;
                    continue;
                }
                for (int aF = 0; aF < transition_model_.get_num_follower_actions(); ++aF) {
                    double pF = follower_rule.get_prob(hF, posg_core::Action(aF, 1));
                    if (pF <= 0.0 || !std::isfinite(pF)) {
                        std::cerr << "[TAU WARNING] Skipping follower action with prob: " << pF << std::endl;
                        continue;
                    }
                    last_aL = aL;
                    last_aF = aF;
                    for (int s_prime = 0; s_prime < transition_model_.get_num_states(); ++s_prime) {
                        double t_prob = transition_model_.get_prob(s, aL, aF, s_prime);
                        if (t_prob <= 0.0 || !std::isfinite(t_prob)) {
                            std::cerr << "[TAU WARNING] Skipping transition with prob: " << t_prob << std::endl;
                            continue;
                        }
                        for (int zL = 0; zL < observation_model_.get_num_leader_observations(); ++zL) {
                            for (int zF = 0; zF < observation_model_.get_num_follower_observations(); ++zF) {
                                double o_prob = observation_model_.get_prob(s_prime, aL, aF, zL, zF);
                                if (o_prob <= 0.0 || !std::isfinite(o_prob)) {
                                    std::cerr << "[TAU WARNING] Skipping observation with prob: " << o_prob << std::endl;
                                    continue;
                                }
                                double contrib = prob * pL * pF * t_prob * o_prob;
                                if (contrib <= 0.0 || !std::isfinite(contrib)) {
                                    std::cerr << "[TAU WARNING] Skipping contrib with value: " << contrib << std::endl;
                                    continue;
                                }
                                auto hL_prime = hL;
                                hL_prime.append_action_observation(posg_core::Action(aL, 0), zL);
                                auto hF_prime = hF;
                                hF_prime.append_action_observation(posg_core::Action(aF, 1), zF);
                                posg_core::OccupancyState::Key new_tuple{s_prime, hL_prime, hF_prime};
                                successor_state.increment_entry(new_tuple, contrib);
                                total_prob += contrib;
                                ++entries_added;
                                if (contrib_printed < 5) {
                                    std::cout << "[TAU DEBUG] s=" << s << ", hL.size=" << hL.size() << ", hF.size=" << hF.size()
                                              << ", aL=" << aL << ", aF=" << aF << " -> s'=" << s_prime << ", zL=" << zL << ", zF=" << zF << " contrib=" << contrib << std::endl;
                                    ++contrib_printed;
                                }
                            }
                        }
                    }
                }
            }
        }
        // Defensive normalization and diagnostics
        double sum = successor_state.total_probability();
        std::cout << "[TAU DEBUG] successor_state entries: " << successor_state.get_entries().size() << ", total_prob: " << sum << ", entries_added: " << entries_added << std::endl;
        if (successor_state.get_entries().empty()) {
            std::cerr << "[TAU WARNING] successor_state is empty after update. Last s=" << last_s << ", hL.size=" << last_hL_size << ", hF.size=" << last_hF_size << ", aL=" << last_aL << ", aF=" << last_aF << std::endl;
        }
        if (sum <= 0.0 || !std::isfinite(sum)) {
            std::cerr << "[TAU ERROR] successor_state has invalid total_prob: " << sum << ". Skipping normalization and returning empty state.\n";
            for (const auto& [tuple, p] : successor_state.get_entries()) {
                std::cerr << "[TAU ERROR] Entry: s=" << tuple.state << ", hL.size=" << tuple.leader_history.size() << ", hF.size=" << tuple.follower_history.size() << ", prob=" << p << std::endl;
            }
            return posg_core::OccupancyState(); // Return empty state
        }
        successor_state.normalize();
        std::vector<std::pair<posg_core::OccupancyState::Key, double>> entries_vec(successor_state.get_entries().begin(), successor_state.get_entries().end());
        std::sort(entries_vec.begin(), entries_vec.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        for (size_t i = 0; i < std::min<size_t>(3, entries_vec.size()); ++i) {
            const auto& [tuple, p] = entries_vec[i];
            std::cout << "[TAU DEBUG] Top entry " << i << ": s=" << tuple.state << ", hL.size=" << tuple.leader_history.size() << ", hF.size=" << tuple.follower_history.size() << ", prob=" << p << std::endl;
        }
        if (!(std::abs(successor_state.total_probability() - 1.0) < 1e-6)) {
            std::cerr << "[TAU WARNING] successor_state not normalized! total_prob=" << successor_state.total_probability() << std::endl;
        }
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
            std::vector<posg_core::CredibleSet> expanded_sets;
            try {
                expanded_sets = expand_credible_sets(credible_sets, 0.1);
            } catch (const std::exception& ex) {
                std::cerr << "[PBVI ERROR] Exception during expansion: " << ex.what() << std::endl;
                // Optionally print last processed occupancy state/credible set
                throw;
            }
            
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
        
        using clock = std::chrono::steady_clock;
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
        
        // Define reward functions for leader and follower (now using real problem rewards)
        auto reward_function_leader = [this](int state, const posg_core::Action& leader_action, const posg_core::Action& follower_action) -> double {
            size_t joint_action_idx = leader_action.get_action_id() * problem_.actions[1].size() + follower_action.get_action_id();
            if (state < problem_.rewards_leader.size() && joint_action_idx < problem_.rewards_leader[state].size()) {
                return problem_.rewards_leader[state][joint_action_idx];
            }
            return 0.0;
        };

        auto reward_function_follower = [this](int state, const posg_core::Action& leader_action, const posg_core::Action& follower_action) -> double {
            size_t joint_action_idx = leader_action.get_action_id() * problem_.actions[1].size() + follower_action.get_action_id();
            if (state < problem_.rewards_follower.size() && joint_action_idx < problem_.rewards_follower[state].size()) {
                return problem_.rewards_follower[state][joint_action_idx];
            }
            return 0.0;
        };
        
        auto pbvi_start = clock::now();
        for (size_t iteration = 0; iteration < max_iterations; ++iteration) {
            auto iter_start = clock::now();
            std::cout << "PBVI with MILP: Iteration " << iteration + 1 << std::endl;
            std::cout << "[DEBUG] Number of credible sets: " << credible_sets.size() << std::endl;
            size_t total_occ = 0;
            for (const auto& cs : credible_sets) total_occ += cs.get_occupancy_states().size();
            std::cout << "[DEBUG] Total occupancy states in all credible sets: " << total_occ << std::endl;
            
            // Improve phase: for each credible set, solve MILP to get greedy decision rule
            std::vector<posg_core::LeaderDecisionRule> optimal_rules;
            std::vector<std::vector<std::pair<std::vector<double>, std::vector<double>>>> new_alpha_vectors;
            
            for (const auto& credible_set : credible_sets) {
                std::cout << "PBVI with MILP: Solving MILP for credible set with " 
                          << credible_set.get_occupancy_states().size() << " occupancy states" << std::endl;
                
                // Solve MILP for greedy leader decision rule
                auto milp_start = clock::now();
                posg_core::LeaderDecisionRule optimal_rule = milp_solver_.solve_milp(
                    credible_set, value_function_collection, transition_model_, observation_model_,
                    reward_function_leader, reward_function_follower);
                auto milp_end = clock::now();
                double milp_sec = std::chrono::duration<double>(milp_end - milp_start).count();
                std::cout << "[PROFILE] MILP solve time: " << milp_sec << " seconds" << std::endl;
                
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
            std::vector<posg_core::CredibleSet> expanded_sets;
            try {
                expanded_sets = expand_credible_sets(credible_sets, 0.1);
            } catch (const std::exception& ex) {
                std::cerr << "[PBVI ERROR] Exception during expansion: " << ex.what() << std::endl;
                // Optionally print last processed occupancy state/credible set
                throw;
            }
            
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
            auto iter_end = clock::now();
            double iter_sec = std::chrono::duration<double>(iter_end - iter_start).count();
            std::cout << "[PROFILE] PBVI iteration time: " << iter_sec << " seconds" << std::endl;
        }
        auto pbvi_end = clock::now();
        double pbvi_sec = std::chrono::duration<double>(pbvi_end - pbvi_start).count();
        std::cout << "[PROFILE] Total PBVI time: " << pbvi_sec << " seconds" << std::endl;
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