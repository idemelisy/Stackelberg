


## Directory Structure
```
LFPOSG-Stackelberg/
├── Stackelberg/src/
│   ├── core/         # Core math objects (occupancy, transitions, etc.)
│   ├── algorithms/   # PBVI and MILP solvers
│   ├── parser/       # Problem file parsing
│   ├── main/         # Command-line interface
│   └── unit_tests/   # Unit and algorithm tests
└── problem_examples/ # Benchmark problems (.stackelberg)
```

## Prerequisites
- **C++17** compiler (GCC 7+ or Clang 5+)
- **IBM CPLEX 22.1+** (with C++ Concert API)
- **Make** build system

## Installation & Build
1. **Install CPLEX** and set environment variables for `CPLEX` and `CONCERT` include/lib paths.
2. **Clone this repo** and enter the main directory.
3. **Build the solver:**
   ```bash
   cd Stackelberg/src/main
   make
   ```

## Running the Solver
Run on a benchmark (e.g., Tiger):
```bash
./run_solver --problem ../../problem_examples/tiger.stackelberg --epsilon 0.1 --maxIter 100
```

**Key options:**
- `--problem <file>`: Path to .stackelberg problem file
- `--epsilon <val>`: Convergence threshold (higher = faster, lower = more accurate)
- `--maxIter <n>`: Maximum PBVI iterations
- `--milpTimeLimit <sec>`: Max seconds per MILP solve (default: 10)

**Example for quick test:**
```bash
./run_solver --problem ../../problem_examples/tiger.stackelberg --epsilon 0.2 --maxIter 2 --milpTimeLimit 2
```

## Running Tests
```bash
cd Stackelberg/src/unit_tests
make clean && make
./run_tests
```

## Troubleshooting
- **Segmentation fault?**
  - Check for empty or invalid occupancy states in debug output.
  - Try running with higher epsilon or fewer iterations for debugging.
- **CPLEX errors?**
  - Ensure CPLEX libraries and include paths are set correctly.
  - Check your CPLEX license.
- **Slow runs?**
  - Increase `--epsilon`, decrease `--maxIter`, or use a smaller benchmark.

---
