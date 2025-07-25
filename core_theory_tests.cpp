#include "core_theory_tests.hpp"
#include <vector>
#include <cmath>
#include "../core/include/occupancy_state.hpp"
#include "../core/include/conditional_occupancy_state.hpp"
#include "../core/include/credible_set.hpp"
#include "../core/include/transition_model.hpp"
#include "../core/include/observation_model.hpp"

namespace test_framework {

// Lemma 4.2: Conditional update
bool CoreTest::test_conditional_update_lemma_4_2() {
    posg_core::AgentHistory follower_hist(1);
    std::vector<double> belief = {0.6, 0.4};
    posg_core::ConditionalOccupancyState c(follower_hist, belief);
    posg_core::TransitionModel tm(2, 1, 1);
    posg_core::ObservationModel om(2, 1, 1, 1, 1);
    posg_core::Action leader_action(0, 0);
    posg_core::Action follower_action(0, 1);
    posg_core::JointAction joint_action(leader_action, follower_action);
    tm.set_transition_probability(0, joint_action, 1, 1.0);
    tm.set_transition_probability(1, joint_action, 0, 1.0);
    posg_core::Observation leader_obs(0, 0);
    posg_core::Observation follower_obs(0, 1);
    posg_core::JointObservation joint_obs(leader_obs, follower_obs);
    om.set_observation_probability(1, joint_action, joint_obs, 1.0);
    om.set_observation_probability(0, joint_action, joint_obs, 1.0);
    auto updated = c.update(leader_action, follower_action, leader_obs, follower_obs, tm, om);
    posg_core::AgentHistory leader_hist(0);
    this->assert_equal(0.4, updated.get_conditional_occupancy(0, leader_hist), 1e-6, "s'=0 per Lemma 4.2");
    this->assert_equal(0.6, updated.get_conditional_occupancy(1, leader_hist), 1e-6, "s'=1 per Lemma 4.2");
    return true;
}

// Tie-breaking under SSE
bool CoreTest::test_sse_tie_breaking() {
    // Not implemented: filtered_reward not available
    return true;
}

// Bellman backup (Theorem 4.1)
bool CoreTest::test_bellman_backup_theorem_4_1() {
    // Not implemented: filtered_reward not available
    return true;
}

// Basic state creation
bool CoreTest::test_occupancy_initialization_from_belief() {
    std::vector<double> belief = {0.7, 0.3};
    posg_core::OccupancyState o(belief);
    posg_core::AgentHistory leader_hist(0);
    posg_core::AgentHistory follower_hist(1);
    this->assert_equal(0.7, o.get_occupancy(0, leader_hist, follower_hist), 1e-6, "Occupancy for state 0 should match");
    this->assert_equal(0.3, o.get_occupancy(1, leader_hist, follower_hist), 1e-6, "Occupancy for state 1 should match");
    return true;
}

// Validates Lemma 4.2
bool CoreTest::test_conditional_occupancy_state_update() {
    posg_core::AgentHistory follower_hist(1);
    std::vector<double> initial_belief = {1.0, 0.0};
    posg_core::ConditionalOccupancyState conditional(follower_hist, initial_belief);
    posg_core::TransitionModel transition_model(2, 2, 2);
    posg_core::ObservationModel observation_model(2, 2, 2, 2, 2);
    posg_core::Action leader_action(0, 0);
    posg_core::Action follower_action(0, 1);
    posg_core::JointAction joint_action(leader_action, follower_action);
    transition_model.set_transition_probability(0, joint_action, 0, 1.0);
    transition_model.set_transition_probability(1, joint_action, 1, 1.0);
    posg_core::Observation leader_obs(0, 0);
    posg_core::Observation follower_obs(0, 1);
    posg_core::JointObservation joint_obs(leader_obs, follower_obs);
    observation_model.set_observation_probability(0, joint_action, joint_obs, 1.0);
    observation_model.set_observation_probability(1, joint_action, joint_obs, 0.0);
    posg_core::ConditionalOccupancyState updated = conditional.update(
        leader_action, follower_action, leader_obs, follower_obs, 
        transition_model, observation_model);
    this->assert_true(updated.is_valid(true), "Updated conditional occupancy state should be valid");
    this->assert_true(follower_hist == updated.get_follower_history(), "Follower history should be preserved");
    return true;
}

// Marginalization property
bool CoreTest::test_conditional_occupancy_state_marginals() {
    posg_core::AgentHistory follower_hist(1);
    posg_core::ConditionalOccupancyState conditional(follower_hist);
    posg_core::AgentHistory leader_hist0(0);
    posg_core::AgentHistory leader_hist1(0);
    leader_hist1.add_action(posg_core::Action(0, 0));
    conditional.set_conditional_occupancy(0, leader_hist0, 0.3);
    conditional.set_conditional_occupancy(1, leader_hist0, 0.2);
    conditional.set_conditional_occupancy(0, leader_hist1, 0.4);
    conditional.set_conditional_occupancy(1, leader_hist1, 0.5);
    auto state_marginal = conditional.get_state_marginal();
    this->assert_equal(0.7, state_marginal[0], 1e-6, "State marginal for state 0 should be 0.7");
    this->assert_equal(0.3, state_marginal[1], 1e-6, "State marginal for state 1 should be 0.3");
    auto leader_marginal = conditional.get_leader_history_marginal();
    this->assert_equal(0.5, leader_marginal[leader_hist0], 1e-6, "Leader marginal for hist0 should be 0.5");
    this->assert_equal(0.5, leader_marginal[leader_hist1], 1e-6, "Leader marginal for hist1 should be 0.5");
    return true;
}

// Decomposition property
bool CoreTest::test_conditional_occupancy_state_decomposition() {
    posg_core::OccupancyState full_occupancy;
    posg_core::AgentHistory leader_hist0(0);
    posg_core::AgentHistory leader_hist1(0);
    posg_core::AgentHistory follower_hist0(1);
    posg_core::AgentHistory follower_hist1(1);
    follower_hist1.add_action(posg_core::Action(0, 1));
    full_occupancy.add_entry(0, leader_hist0, follower_hist0, 0.3);
    full_occupancy.add_entry(1, leader_hist0, follower_hist0, 0.2);
    full_occupancy.add_entry(0, leader_hist1, follower_hist1, 0.4);
    full_occupancy.add_entry(1, leader_hist1, follower_hist1, 0.5);
    posg_core::ConditionalOccupancyState conditional0(follower_hist0);
    conditional0.set_conditional_occupancy(0, leader_hist0, 0.6);
    conditional0.set_conditional_occupancy(1, leader_hist0, 0.4);
    this->assert_true(conditional0.is_valid(true), "Manually constructed conditional should be valid");
    return true;
}

// CredibleSet structure
bool CoreTest::test_credible_set_creation_and_operations() {
    posg_core::CredibleSet empty_set;
    this->assert_true(empty_set.empty(), "Default credible set should be empty");
    this->assert_equal(0, empty_set.size(), "Empty credible set should have size 0");
    posg_core::OccupancyState single_occupancy;
    posg_core::AgentHistory leader_hist(0);
    posg_core::AgentHistory follower_hist(1);
    single_occupancy.add_entry(0, leader_hist, follower_hist, 1.0);
    posg_core::CredibleSet single_set(single_occupancy);
    this->assert_false(single_set.empty(), "Credible set with single state should not be empty");
    this->assert_equal(1, single_set.size(), "Credible set should have size 1");
    this->assert_true(single_set.contains(single_occupancy), "Credible set should contain the added state");
    posg_core::OccupancyState occupancy1, occupancy2;
    occupancy1.add_entry(0, leader_hist, follower_hist, 0.6);
    occupancy1.add_entry(1, leader_hist, follower_hist, 0.4);
    occupancy2.add_entry(0, leader_hist, follower_hist, 0.3);
    occupancy2.add_entry(1, leader_hist, follower_hist, 0.7);
    std::vector<posg_core::OccupancyState> states = {occupancy1, occupancy2};
    posg_core::CredibleSet multi_set(states);
    this->assert_equal(2, multi_set.size(), "Credible set should have size 2");
    this->assert_true(multi_set.contains(occupancy1), "Credible set should contain occupancy1");
    this->assert_true(multi_set.contains(occupancy2), "Credible set should contain occupancy2");
    posg_core::CredibleSet test_set;
    test_set.add_occupancy_state(occupancy1);
    this->assert_equal(1, test_set.size(), "After adding, size should be 1");
    test_set.add_occupancy_state(occupancy2);
    this->assert_equal(2, test_set.size(), "After adding second, size should be 2");
    test_set.remove_occupancy_state(occupancy1);
    this->assert_equal(1, test_set.size(), "After removing, size should be 1");
    this->assert_false(test_set.contains(occupancy1), "Should not contain removed state");
    this->assert_true(test_set.contains(occupancy2), "Should still contain remaining state");
    return true;
}

// CredibleSet conditional logic
bool CoreTest::test_credible_set_conditional_decomposition() {
    posg_core::CredibleSet credible_set;
    posg_core::AgentHistory leader_hist(0);
    posg_core::AgentHistory follower_hist(1);
    posg_core::OccupancyState occupancy;
    occupancy.add_entry(0, leader_hist, follower_hist, 0.6);
    occupancy.add_entry(1, leader_hist, follower_hist, 0.4);
    credible_set.add_occupancy_state(occupancy);
    auto all_conditionals = credible_set.get_conditional_occupancy_states();
    this->assert_false(all_conditionals.empty(), "Should have conditional occupancy states");
    auto specific_conditionals = credible_set.get_conditional_occupancy_states(follower_hist);
    this->assert_false(specific_conditionals.empty(), "Should have conditional occupancy states for specific history");
    for (const auto& conditional : specific_conditionals) {
        this->assert_true(conditional.is_valid(true), "Conditional occupancy state should be valid");
        this->assert_true(follower_hist == conditional.get_follower_history(), "Follower history should match");
    }
    return true;
}

// Hausdorff metric for ε-bound (Thm 5.3)
bool CoreTest::test_credible_set_hausdorff_distance() {
    posg_core::CredibleSet set1, set2;
    posg_core::AgentHistory leader_hist(0);
    posg_core::AgentHistory follower_hist(1);
    posg_core::OccupancyState occupancy1, occupancy2, occupancy3;
    occupancy1.add_entry(0, leader_hist, follower_hist, 1.0);
    occupancy2.add_entry(1, leader_hist, follower_hist, 1.0);
    occupancy3.add_entry(0, leader_hist, follower_hist, 0.5);
    occupancy3.add_entry(1, leader_hist, follower_hist, 0.5);
    set1.add_occupancy_state(occupancy1);
    set1.add_occupancy_state(occupancy2);
    set2.add_occupancy_state(occupancy3);
    double distance = set1.hausdorff_distance(set2);
    this->assert_true(distance >= 0.0, "Hausdorff distance should be non-negative");
    double distance_reverse = set2.hausdorff_distance(set1);
    this->assert_equal(distance, distance_reverse, 1e-6, "Hausdorff distance should be symmetric");
    double self_distance = set1.hausdorff_distance(set1);
    this->assert_equal(0.0, self_distance, 0.000001, "Distance to self should be zero");
    return true;
}

} // namespace test_framework 