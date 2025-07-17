#pragma once

#include "test_framework.hpp"
#include "../core/include/common.hpp"
#include "../core/include/transition_model.hpp"
#include "../core/include/observation_model.hpp"

namespace test_framework {

    /**
     * @brief Test suite for core POSG classes
     * 
     * Tests the fundamental data structures and models used in the POSG framework:
     * - Action, Observation, JointAction, JointObservation
     * - TransitionModel for state dynamics
     * - ObservationModel for observation generation
     */
    class CoreTest : public TestBase {
    private:
        // Test methods for common classes
        bool test_action_creation_and_operations();
        bool test_observation_creation_and_operations();
        bool test_joint_action_creation_and_operations();
        bool test_joint_observation_creation_and_operations();
        bool test_action_hash_functions();
        bool test_observation_hash_functions();
        
        // Test methods for transition model
        bool test_transition_model_creation();
        bool test_transition_probability_setting();
        bool test_transition_probability_retrieval();
        bool test_transition_model_validation();
        bool test_transition_model_normalization();
        bool test_transition_sampling();
        bool test_transition_model_integration();
        
        // Test methods for observation model
        bool test_observation_model_creation();
        bool test_observation_probability_setting();
        bool test_observation_probability_retrieval();
        bool test_observation_model_validation();
        bool test_observation_model_normalization();
        bool test_observation_sampling();
        bool test_observation_model_integration();
        
        // Integration tests
        bool test_belief_update_integration();
        bool test_tiger_problem_integration();
        bool test_centipede_problem_integration();

        // Core POSG class tests
        bool test_empty_occupancy_state();
        bool test_malformed_occupancy_distribution();
        bool test_conditional_decomposition();
        bool test_occupancy_propagation();
        bool test_credible_set_transition();
        bool test_reward_filtering();
        bool test_convex_decomposition();
        // New OccupancyState tests
        bool test_occupancy_initialization_from_belief();
        bool test_occupancy_entry_addition_removal();
        bool test_occupancy_normalization_and_validation();
        bool test_occupancy_entropy();
        bool test_occupancy_distance();
        bool test_occupancy_accessors();

        // Phase 1 Test Finalization - New Tests
        // ConditionalOccupancyState tests
        bool test_conditional_occupancy_state_creation();
        bool test_conditional_occupancy_state_update();
        bool test_conditional_occupancy_state_marginals();
        bool test_conditional_occupancy_state_decomposition();
        bool test_conditional_occupancy_state_entropy_and_distance();
        
        // CredibleSet tests
        bool test_credible_set_creation_and_operations();
        bool test_credible_set_transition_under_policy();
        bool test_credible_set_filtered_reward();
        bool test_credible_set_conditional_decomposition();
        bool test_credible_set_hausdorff_distance();
        
        // Missing OccupancyState utilities
        bool test_occupancy_state_convex_decomposition();
        bool test_occupancy_state_propagation();
        bool test_occupancy_state_conditional_decomposition();

    public:
        CoreTest() : TestBase("Core POSG Classes") {}
        
        bool run() override {
            bool all_passed = true;
            
            // Test common classes
            all_passed &= test_action_creation_and_operations();
            all_passed &= test_observation_creation_and_operations();
            all_passed &= test_joint_action_creation_and_operations();
            all_passed &= test_joint_observation_creation_and_operations();
            all_passed &= test_action_hash_functions();
            all_passed &= test_observation_hash_functions();
            
            // Test transition model
            all_passed &= test_transition_model_creation();
            all_passed &= test_transition_probability_setting();
            all_passed &= test_transition_probability_retrieval();
            all_passed &= test_transition_model_validation();
            all_passed &= test_transition_model_normalization();
            all_passed &= test_transition_sampling();
            all_passed &= test_transition_model_integration();
            
            // Test observation model
            all_passed &= test_observation_model_creation();
            all_passed &= test_observation_probability_setting();
            all_passed &= test_observation_probability_retrieval();
            all_passed &= test_observation_model_validation();
            all_passed &= test_observation_model_normalization();
            all_passed &= test_observation_sampling();
            all_passed &= test_observation_model_integration();
            
            // Integration tests
            all_passed &= test_tiger_problem_integration();
            all_passed &= test_centipede_problem_integration();

            // Core POSG class tests
            all_passed &= test_empty_occupancy_state();
            all_passed &= test_malformed_occupancy_distribution();
            all_passed &= test_conditional_decomposition();
            all_passed &= test_occupancy_propagation();
            all_passed &= test_credible_set_transition();
            all_passed &= test_reward_filtering();
            all_passed &= test_convex_decomposition();
            // New OccupancyState tests
            all_passed &= test_occupancy_initialization_from_belief();
            all_passed &= test_occupancy_entry_addition_removal();
            all_passed &= test_occupancy_normalization_and_validation();
            all_passed &= test_occupancy_entropy();
            all_passed &= test_occupancy_distance();
            all_passed &= test_occupancy_accessors();
            
            // Phase 1 Test Finalization - New Tests
            // ConditionalOccupancyState tests
            all_passed &= test_conditional_occupancy_state_creation();
            all_passed &= test_conditional_occupancy_state_update();
            all_passed &= test_conditional_occupancy_state_marginals();
            all_passed &= test_conditional_occupancy_state_decomposition();
            all_passed &= test_conditional_occupancy_state_entropy_and_distance();
            
            // CredibleSet tests
            all_passed &= test_credible_set_creation_and_operations();
            all_passed &= test_credible_set_transition_under_policy();
            all_passed &= test_credible_set_filtered_reward();
            all_passed &= test_credible_set_conditional_decomposition();
            all_passed &= test_credible_set_hausdorff_distance();
            
            // Missing OccupancyState utilities
            all_passed &= test_occupancy_state_convex_decomposition();
            all_passed &= test_occupancy_state_propagation();
            all_passed &= test_occupancy_state_conditional_decomposition();

            return all_passed;
        }
    };

} // namespace test_framework 