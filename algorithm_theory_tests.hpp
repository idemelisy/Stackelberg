#pragma once
#include "test_framework.hpp"

namespace test_framework {

class AlgorithmTest : public TestBase {
public:
    AlgorithmTest() : TestBase("AlgorithmTest") {}
    bool run() override {
        bool all_passed = true;
        all_passed &= test_bellman_update();
        all_passed &= test_policy_extraction();
        all_passed &= test_milp_value_matches_hand_solution();
        all_passed &= test_epsilon_bound_matches_theorem_5_3();
        all_passed &= test_centipede_game_matches_known_sse();
        return all_passed;
    }
    bool test_bellman_update();
    bool test_policy_extraction();
    bool test_milp_value_matches_hand_solution();
    bool test_epsilon_bound_matches_theorem_5_3();
    bool test_centipede_game_matches_known_sse();
};

} // namespace test_framework 