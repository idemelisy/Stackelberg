# Minimal, paper-aligned test suite Makefile

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -g -O0 -I. -I../parser/include -I../core/include -I../algorithms/include

CPLEX_ROOT = /home/ide/Desktop/cplex_ibm
CPLEX_INCLUDE = -I$(CPLEX_ROOT)/cplex/include -I$(CPLEX_ROOT)/concert/include
CPLEX_LIB = -L$(CPLEX_ROOT)/cplex/lib/x86-64_linux/static_pic -L$(CPLEX_ROOT)/concert/lib/x86-64_linux/static_pic
CPLEX_LIBS = -lilocplex -lconcert -lcplex -lm -lpthread -ldl

CXXFLAGS += $(CPLEX_INCLUDE)

# Test sources
TEST_SOURCES = test_framework.cpp core_theory_tests.cpp algorithm_theory_tests.cpp test_runner.cpp

# Core sources
CORE_SOURCES = ../core/src/common.cpp \
               ../core/src/transition_model.cpp \
               ../core/src/observation_model.cpp \
               ../core/src/occupancy_state.cpp \
               ../core/src/conditional_occupancy_state.cpp \
               ../core/src/credible_set.cpp \
               ../core/src/credible_mdp.cpp

# Parser sources
PARSER_SOURCES = ../parser/src/posg_parser.cpp

# Algorithm sources
ALGORITHM_SOURCES = ../algorithms/src/cmdp_solver.cpp \
                    ../algorithms/src/milp_solver.cpp

SOURCES = $(TEST_SOURCES) $(CORE_SOURCES) $(PARSER_SOURCES) $(ALGORITHM_SOURCES)
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = run_tests

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(TARGET) $(CPLEX_LIB) $(CPLEX_LIBS)

# Compile test sources
core_theory_tests.o: core_theory_tests.cpp test_framework.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

algorithm_theory_tests.o: algorithm_theory_tests.cpp test_framework.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

test_framework.o: test_framework.cpp test_framework.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

test_runner.o: test_runner.cpp test_framework.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile core source files
../core/src/common.o: ../core/src/common.cpp ../core/include/common.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

../core/src/transition_model.o: ../core/src/transition_model.cpp ../core/include/transition_model.hpp ../core/include/common.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

../core/src/observation_model.o: ../core/src/observation_model.cpp ../core/include/observation_model.hpp ../core/include/common.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

../core/src/occupancy_state.o: ../core/src/occupancy_state.cpp ../core/include/occupancy_state.hpp ../core/include/common.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

../core/src/conditional_occupancy_state.o: ../core/src/conditional_occupancy_state.cpp ../core/include/conditional_occupancy_state.hpp ../core/include/occupancy_state.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

../core/src/credible_set.o: ../core/src/credible_set.cpp ../core/include/credible_set.hpp ../core/include/occupancy_state.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

../core/src/credible_mdp.o: ../core/src/credible_mdp.cpp ../core/include/credible_mdp.hpp ../core/include/credible_set.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile parser source
../parser/src/posg_parser.o: ../parser/src/posg_parser.cpp ../parser/include/posg_parser.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile algorithm source files
../algorithms/src/cmdp_solver.o: ../algorithms/src/cmdp_solver.cpp ../algorithms/include/cmdp_solver.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

../algorithms/src/milp_solver.o: ../algorithms/src/milp_solver.cpp ../algorithms/include/milp_solver.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Run tests
test: $(TARGET)
	./$(TARGET)

clean:
	rm -f *.o ../parser/src/*.o ../core/src/*.o ../algorithms/src/*.o $(TARGET) 