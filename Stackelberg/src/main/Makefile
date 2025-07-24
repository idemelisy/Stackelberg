CORE_SRCS := $(wildcard ../core/src/*.cpp)
ALGO_SRCS := $(wildcard ../algorithms/src/*.cpp)
PARSER_SRCS := $(wildcard ../parser/src/*.cpp)

CPLEX_INC := -I/home/ide/Desktop/cplex_ibm/cplex/include -I/home/ide/Desktop/cplex_ibm/concert/include
CPLEX_LIB := -L/home/ide/Desktop/cplex_ibm/cplex/lib/x86-64_linux/static_pic -L/home/ide/Desktop/cplex_ibm/concert/lib/x86-64_linux/static_pic -lilocplex -lcplex -lconcert -lm -lpthread

run_solver: main.cpp $(CORE_SRCS) $(ALGO_SRCS) $(PARSER_SRCS)
	g++ -std=c++17 -O2 \
		-I../core/include \
		-I../algorithms/include \
		-I../parser/include \
		$(CPLEX_INC) \
		main.cpp $(CORE_SRCS) $(ALGO_SRCS) $(PARSER_SRCS) \
		-o run_solver \
		$(CPLEX_LIB) 