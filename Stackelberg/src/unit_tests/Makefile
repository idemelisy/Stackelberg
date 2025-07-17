# Makefile for POSG Parser Unit Tests

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -g -O0
INCLUDES = -I. -I../parser/include -I../core/include

# Source files
TEST_FRAMEWORK_SRC = test_framework.cpp
PARSER_TEST_SRC = parser_test.cpp
CORE_TEST_SRC = core_test.cpp
TEST_RUNNER_SRC = test_runner.cpp
PARSER_SRC = ../parser/src/posg_parser.cpp
CORE_SRC = ../core/src/common.cpp ../core/src/transition_model.cpp ../core/src/observation_model.cpp ../core/src/occupancy_state.cpp ../core/src/conditional_occupancy_state.cpp ../core/src/credible_set.cpp ../core/src/credible_mdp.cpp

# Object files
TEST_FRAMEWORK_OBJ = test_framework.o
PARSER_TEST_OBJ = parser_test.o
CORE_TEST_OBJ = core_test.o
TEST_RUNNER_OBJ = test_runner.o
PARSER_OBJ = ../parser/src/posg_parser.o
CORE_OBJS = ../core/src/common.o ../core/src/transition_model.o ../core/src/observation_model.o ../core/src/occupancy_state.o ../core/src/conditional_occupancy_state.o ../core/src/credible_set.o ../core/src/credible_mdp.o

# Target executable
TARGET = run_tests

# Default target
all: $(TARGET)

# Build the test executable
$(TARGET): $(TEST_FRAMEWORK_OBJ) $(PARSER_TEST_OBJ) $(CORE_TEST_OBJ) $(TEST_RUNNER_OBJ) $(PARSER_OBJ) $(CORE_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile test framework
$(TEST_FRAMEWORK_OBJ): $(TEST_FRAMEWORK_SRC) test_framework.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile parser tests
$(PARSER_TEST_OBJ): $(PARSER_TEST_SRC) parser_test.hpp test_framework.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile core tests
$(CORE_TEST_OBJ): $(CORE_TEST_SRC) core_test.hpp test_framework.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile test runner
$(TEST_RUNNER_OBJ): $(TEST_RUNNER_SRC) test_framework.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile parser source
$(PARSER_OBJ): $(PARSER_SRC) ../parser/include/posg_parser.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile core source files
../core/src/common.o: ../core/src/common.cpp ../core/include/common.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

../core/src/transition_model.o: ../core/src/transition_model.cpp ../core/include/transition_model.hpp ../core/include/common.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

../core/src/observation_model.o: ../core/src/observation_model.cpp ../core/include/observation_model.hpp ../core/include/common.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

../core/src/occupancy_state.o: ../core/src/occupancy_state.cpp ../core/include/occupancy_state.hpp ../core/include/common.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

../core/src/conditional_occupancy_state.o: ../core/src/conditional_occupancy_state.cpp ../core/include/conditional_occupancy_state.hpp ../core/include/occupancy_state.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

../core/src/credible_set.o: ../core/src/credible_set.cpp ../core/include/credible_set.hpp ../core/include/occupancy_state.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

../core/src/credible_mdp.o: ../core/src/credible_mdp.cpp ../core/include/credible_mdp.hpp ../core/include/credible_set.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Run tests
test: $(TARGET)
	./$(TARGET)

# Clean build artifacts
clean:
	rm -f *.o ../parser/src/*.o ../core/src/*.o $(TARGET) debug_tiger

# Show help
help:
	@echo "Available targets:"
	@echo "  all     - Build the test executable (default)"
	@echo "  test    - Build and run all tests"
	@echo "  clean   - Remove build artifacts"
	@echo "  help    - Show this help message"

.PHONY: all test clean help 