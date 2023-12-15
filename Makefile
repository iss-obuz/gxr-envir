.PHONY: help clean clean-pyc clean-build test cov-run cov-report coverage list-deps build

help:
	@echo "clean-build - remove build artifacts"
	@echo "clean-misc - remove various Python file artifacts"
	@echo "test - run unit tests"
	@echo "cov-run - run tests and calculate coverage statistics"
	@echo "cov-report - display test coverage statistics"
	@echo "coverage - run tests and display coverage statistics"
	@echo "list-deps - list explicit dependencies of the project"

clean: clean-build clean-misc

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr *.egg-info

clean-misc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '.benchmarks' -exec rm -rf {} +
	find . -name '.pytest-cache' -exec rm -rf {} +
	find . -name '.pytest_cache' -exec rm -rf {} +
	find . -name '__pycache__' -exec rm -rf {} +

test:
	pytest

cov-run:
	coverage run
cov-report:
	coverage report

coverage: cov-run cov-report

list-deps:
	find nodes/data tests -name "*.py" -exec cat {} + | grep "^import \| import " | grep -v "\"\|'" | grep -v "\(import\|from\) \+\." | sed 's/^\s*import\s*//g' | sed 's/\s*import.*//g' | sed 's/\s*from\s*//g' | grep -v "\..*" | sort | uniq

build:
	python -m build
	twine check dist/*
