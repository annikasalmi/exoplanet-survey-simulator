# Unit Tests for mdwarf-habitability

This directory contains unit tests for the mdwarf-habitability project, focusing on physics calculations and exozodi constraint functionality.

## Test Structure

- `test_physics.py`: Tests for physics calculations (blackbody, Planck integration, exozodi temperature)
- `test_exozodi_constraint.py`: Tests for exozodi constraint functionality in HWO data analysis

## Running Tests

### Local Testing

1. **Using the test runner script:**
   ```bash
   python run_tests.py
   ```

2. **Using pytest:**
   ```bash
   pytest tests/ -v
   ```

3. **Running specific test files:**
   ```bash
   pytest tests/test_physics.py -v
   pytest tests/test_exozodi_constraint.py -v
   ```

4. **Running with coverage:**
   ```bash
   pytest tests/ -v --cov=. --cov-report=html
   ```

### GitHub Actions

Tests are automatically run on:
- Every push to main/master branch
- Every pull request

The workflow tests against Python versions 3.8, 3.9, 3.10, and 3.11.

## Test Categories

### Physics Tests (`test_physics.py`)

- **TestBlackbodyPhysics**: Tests for blackbody radiation calculations
  - Planck integration over full wavelength range
  - Planck integration for HWO wavelength band
  - Temperature dependence of Planck function
  - Wien's displacement law verification
  - Numerical stability

- **TestExozodiTemperature**: Tests for exozodi temperature calculations
  - Radiative equilibrium behavior
  - Stellar property dependence
  - Temperature bounds validation
  - Temperature profile calculations

- **TestExozodiFlux**: Tests for exozodi flux calculations
  - Distance dependence
  - Exozodi level scaling
  - Flux ratio calculations

- **TestHWODataPhysics**: Tests for HWO data physics calculations
  - Blackbody flux calculations
  - Planet flux calculations
  - Flux ratio calculations
  - Photon rate calculations
  - Exozodi constraint calculations

- **TestPhysicsConstants**: Tests for physics constants
  - Planck constant
  - Speed of light
  - Boltzmann constant
  - Stefan-Boltzmann constant
  - Earth and solar radii

### Exozodi Constraint Tests (`test_exozodi_constraint.py`)

- **TestExozodiConstraint**: Tests for exozodi constraint functionality
  - Detection without exozodi constraint
  - Detection with exozodi constraint
  - Impact of exozodi constraint on detection rates
  - Different exozodi scenarios (baseline, pessimistic, optimistic)
  - Flux ratio validation
  - Pass condition validation
  - Simplified constraint calculations
  - Exozodi level scaling

## Test Dependencies

The tests require the following packages:
- numpy
- pandas
- scipy
- matplotlib
- pytest
- pytest-cov

## Adding New Tests

1. Create a new test file following the naming convention `test_*.py`
2. Inherit from `unittest.TestCase`
3. Use descriptive test method names starting with `test_`
4. Add docstrings explaining what each test validates
5. Use appropriate assertions to validate expected behavior

## Continuous Integration

The GitHub Actions workflow (`.github/workflows/tests.yml`) automatically:
1. Sets up Python environments
2. Installs dependencies
3. Runs all tests
4. Generates coverage reports
5. Uploads coverage to Codecov (if configured)

## Coverage

To view test coverage locally:
```bash
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html
```

This will show which parts of the code are tested and which need additional test coverage. 