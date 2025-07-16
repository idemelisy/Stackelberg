#pragma once

#include "test_framework.hpp"
#include "../parser/include/posg_parser.hpp"
#include <string>

namespace test_framework {

    /**
     * @brief Test class for POSG parser functionality
     */
    class ParserTest : public TestBase {
    private:
        // Individual test methods
        bool test_parser_construction();
        bool test_tiger_problem_parsing();
        bool test_centipede_problem_parsing();
        bool test_conitzer_problem_parsing();
        bool test_mabc_problem_parsing();
        bool test_patrolling_problem_parsing();
        bool test_invalid_file_handling();
        bool test_problem_validation();
        bool test_problem_properties();

    public:
        ParserTest();
        
        /**
         * @brief Run all parser tests
         */
        bool run() override;
    };

} // namespace test_framework 