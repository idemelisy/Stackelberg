#pragma once
#include "test_framework.hpp"

namespace test_framework {

class CoreTest : public TestBase {
public:
    CoreTest() : TestBase("CoreTest") {}
    bool run() override {
        bool all_passed = true;
        all_passed &= test_conditional_update_lemma_4_2();
        all_passed &= test_occupancy_initialization_from_belief();
        all_passed &= test_conditional_occupancy_state_update();
        all_passed &= test_conditional_occupancy_state_marginals();
        all_passed &= test_conditional_occupancy_state_decomposition();
        all_passed &= test_credible_set_creation_and_operations();
        all_passed &= test_credible_set_conditional_decomposition();
        all_passed &= test_credible_set_hausdorff_distance();
        return all_passed;
    }
    bool test_conditional_update_lemma_4_2();
    bool test_occupancy_initialization_from_belief();
    bool test_conditional_occupancy_state_update();
    bool test_conditional_occupancy_state_marginals();
    bool test_conditional_occupancy_state_decomposition();
    bool test_credible_set_creation_and_operations();
    bool test_credible_set_conditional_decomposition();
    bool test_credible_set_hausdorff_distance();
};

} // namespace test_framework 