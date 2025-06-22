# Makefile for mdwarf-habitability project

.PHONY: test test-verbose test-coverage test-physics test-exozodi clean help

# Default target
help:
	@echo "Available commands:"
	@echo "  test          - Run all tests"
	@echo "  test-verbose  - Run tests with verbose output"
	@echo "  test-coverage - Run tests with coverage report"
	@echo "  test-physics  - Run only physics tests"
	@echo "  test-exozodi  - Run only exozodi constraint tests"
	@echo "  clean         - Clean up generated files"
	@echo "  install       - Install dependencies"

# Run all tests
test:
	python run_tests.py

# Run tests with verbose output
test-verbose:
	python -m pytest tests/ -v

# Run tests with coverage
test-coverage:
	python -m pytest tests/ -v --cov=. --cov-report=html --cov-report=term

# Run only physics tests
test-physics:
	python -m pytest tests/test_physics.py -v

# Run only exozodi constraint tests
test-exozodi:
	python -m pytest tests/test_exozodi_constraint.py -v

# Run basic tests
test-basic:
	python -m pytest tests/test_basic.py -v

# Install dependencies
install:
	pip install -r requirements.txt

# Clean up generated files
clean:
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	rm -rf tests/__pycache__/
	rm -rf *.pyc
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Run tests with specific Python version (for CI)
test-ci:
	python -m pytest tests/ -v --tb=short 