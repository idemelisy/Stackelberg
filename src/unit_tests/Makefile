# Unit Tests Makefile
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -g -O0
INCLUDES = -I. -I../parser/include -I.
LIBS = 

# Source files
TEST_FRAMEWORK_SOURCES = test_framework.cpp
PARSER_TEST_SOURCES = parser_test.cpp
PARSER_SOURCES = ../parser/src/posg_parser.cpp
TEST_RUNNER_SOURCES = test_runner.cpp
DEBUG_TIGER_SOURCES = debug_tiger.cpp

# Object files
TEST_FRAMEWORK_OBJS = $(TEST_FRAMEWORK_SOURCES:.cpp=.o)
PARSER_TEST_OBJS = $(PARSER_TEST_SOURCES:.cpp=.o)
PARSER_OBJS = $(PARSER_SOURCES:.cpp=.o)
TEST_RUNNER_OBJS = $(TEST_RUNNER_SOURCES:.cpp=.o)
DEBUG_TIGER_OBJS = $(DEBUG_TIGER_SOURCES:.cpp=.o)

# Target executables
TEST_EXECUTABLE = run_tests
DEBUG_TIGER_EXECUTABLE = debug_tiger

# Default target
all: $(TEST_EXECUTABLE)

# Build the test executable
$(TEST_EXECUTABLE): $(TEST_FRAMEWORK_OBJS) $(PARSER_TEST_OBJS) $(PARSER_OBJS) $(TEST_RUNNER_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Build the debug tiger executable
$(DEBUG_TIGER_EXECUTABLE): $(DEBUG_TIGER_OBJS) $(PARSER_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Compile test framework
test_framework.o: test_framework.cpp test_framework.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile parser test
parser_test.o: parser_test.cpp parser_test.hpp test_framework.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile parser source
../parser/src/posg_parser.o: ../parser/src/posg_parser.cpp ../parser/include/posg_parser.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile test runner
test_runner.o: test_runner.cpp test_framework.hpp parser_test.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile debug tiger
debug_tiger.o: debug_tiger.cpp ../parser/include/posg_parser.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Run tests
test: $(TEST_EXECUTABLE)
	./$(TEST_EXECUTABLE)

# Run debug tiger
debug-tiger: $(DEBUG_TIGER_EXECUTABLE)
	./$(DEBUG_TIGER_EXECUTABLE)

# Clean build artifacts
clean:
	rm -f *.o ../parser/src/*.o $(TEST_EXECUTABLE) $(DEBUG_TIGER_EXECUTABLE)

# Clean and rebuild
rebuild: clean all

# Show help
help:
	@echo "Available targets:"
	@echo "  all         - Build the test executable"
	@echo "  test        - Build and run all tests"
	@echo "  debug-tiger - Build and run tiger parser debug"
	@echo "  clean       - Remove build artifacts"
	@echo "  rebuild     - Clean and rebuild"
	@echo "  help        - Show this help message"

.PHONY: all test debug-tiger clean rebuild help 