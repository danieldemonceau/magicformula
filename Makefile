.PHONY: help install install-dev install-pre-commit format lint lint-fix type-check test test-cov check fix clean all conda-install conda-install-dev run-analysis

.DEFAULT_GOAL := help

PACKAGE := src

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies with pip
	pip install -r requirements.txt
	pip install -e .

install-dev: ## Install development dependencies with pip
	pip install -e ".[dev]"
	pip install pre-commit pytest-cov

install-pre-commit: install-dev ## Install pre-commit hooks
	pre-commit install

conda-install: ## Create/update conda environment (production)
	@echo "Using conda from: $$(conda info --base)"
	@echo "Environment will be created in first writable envs_dir"
	conda env update -f environment.yml --prune
	pip install -r requirements.txt

conda-install-dev: ## Create/update conda environment (development)
	@echo "Using conda from: $$(conda info --base)"
	@echo "Environment will be created in first writable envs_dir"
	@echo "Current envs_dirs: $$(conda config --show envs_dirs 2>/dev/null || echo 'Not configured')"
	conda env update -f environment-dev.yml --prune
	pip install -r requirements-dev.txt

conda-remove-dev: ## Remove development conda environment
	@echo "Removing magicformula-env-dev environment..."
	conda env remove -n magicformula-env-dev || echo "Environment not found or already removed"

conda-setup-miniconda: ## Configure conda to prioritize current conda base (reorders existing envs_dirs)
	@echo "Reordering envs_dirs to prioritize current conda base..."
	@CONDA_BASE=$$(conda info --base 2>/dev/null); \
	if [ -z "$$CONDA_BASE" ]; then \
		echo "Error: Could not determine conda base. Is conda installed and activated?"; \
		exit 1; \
	fi; \
	CONDA_ENVS_DIR="$$CONDA_BASE/envs"; \
	echo "Detected conda base: $$CONDA_BASE"; \
	echo "Will reorder existing envs_dirs to put $$CONDA_ENVS_DIR first"; \
	echo ""; \
	echo "Saving current envs_dirs..."; \
	EXISTING_DIRS=$$(conda config --show envs_dirs 2>/dev/null | grep -E "^\s+-" | sed 's/^[[:space:]]*-[[:space:]]*//' | sed 's/^"//' | sed 's/"$$//' | tr '\n' '|'); \
	echo "Removing existing entries..."; \
	for dir in $$(echo "$$EXISTING_DIRS" | tr '|' '\n'); do \
		[ -n "$$dir" ] && conda config --remove envs_dirs "$$dir" 2>/dev/null || true; \
	done; \
	echo "Adding current conda envs directory first..."; \
	conda config --prepend envs_dirs "$$CONDA_ENVS_DIR" 2>/dev/null || conda config --add envs_dirs "$$CONDA_ENVS_DIR" 2>/dev/null || true; \
	echo "Restoring other directories..."; \
	for dir in $$(echo "$$EXISTING_DIRS" | tr '|' '\n'); do \
		if [ -n "$$dir" ] && [ "$$dir" != "$$CONDA_ENVS_DIR" ]; then \
			conda config --add envs_dirs "$$dir" 2>/dev/null || true; \
		fi; \
	done; \
	echo ""; \
	echo "Current conda is now first in envs_dirs:"; \
	conda config --show envs_dirs

conda-setup-esri: ## Configure conda to prioritize ESRI paths (reorders existing envs_dirs)
	@echo "Reordering envs_dirs to prioritize ESRI paths..."
	@CONDA_BASE=$$(conda info --base 2>/dev/null); \
	if [ -z "$$CONDA_BASE" ]; then \
		echo "Error: Could not determine conda base. Is conda installed and activated?"; \
		exit 1; \
	fi; \
	CONDA_ENVS_DIR="$$CONDA_BASE/envs"; \
	echo "Detected conda base: $$CONDA_BASE"; \
	echo "Will reorder existing envs_dirs to put ESRI paths first"; \
	echo ""; \
	echo "Saving current envs_dirs..."; \
	EXISTING_DIRS=$$(conda config --show envs_dirs 2>/dev/null | grep -E "^\s+-" | sed 's/^[[:space:]]*-[[:space:]]*//' | sed 's/^"//' | sed 's/"$$//' | tr '\n' '|'); \
	ESRI_DIRS=$$(echo "$$EXISTING_DIRS" | tr '|' '\n' | grep -i "esri\|arcgis" | tr '\n' '|'); \
	OTHER_DIRS=$$(echo "$$EXISTING_DIRS" | tr '|' '\n' | grep -v -i "esri\|arcgis" | tr '\n' '|'); \
	echo "Removing existing entries..."; \
	for dir in $$(echo "$$EXISTING_DIRS" | tr '|' '\n'); do \
		[ -n "$$dir" ] && conda config --remove envs_dirs "$$dir" 2>/dev/null || true; \
	done; \
	echo "Adding ESRI paths first..."; \
	for dir in $$(echo "$$ESRI_DIRS" | tr '|' '\n'); do \
		[ -n "$$dir" ] && conda config --prepend envs_dirs "$$dir" 2>/dev/null || true; \
	done; \
	echo "Adding other directories..."; \
	for dir in $$(echo "$$OTHER_DIRS" | tr '|' '\n'); do \
		[ -n "$$dir" ] && conda config --add envs_dirs "$$dir" 2>/dev/null || true; \
	done; \
	echo ""; \
	echo "ESRI paths are now first in envs_dirs:"; \
	conda config --show envs_dirs

format: ## Format code with ruff
	ruff format $(PACKAGE) scripts tests

lint: ## Run linter (ruff)
	ruff check $(PACKAGE) scripts tests

lint-fix: ## Run linter with auto-fix (ruff)
	ruff check $(PACKAGE) scripts tests --fix

type-check: ## Run type checker (mypy)
	mypy $(PACKAGE) scripts

test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ --cov=$(PACKAGE) --cov-report=term-missing --cov-report=html

check: format lint type-check ## Run all checks (format, lint, type-check)
	@echo "All checks passed!"

fix: format lint-fix ## Auto-fix all fixable issues (format + lint with --fix)
	@echo "All auto-fixable issues have been fixed!"

clean: ## Clean up generated files (cross-platform)
	python -c "import shutil; import pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	python -c "import shutil; import pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('*.egg-info')]"
	python -c "import shutil; import pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('.pytest_cache')]"
	python -c "import shutil; import pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('.mypy_cache')]"
	python -c "import shutil; import pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('.ruff_cache')]"
	python -c "import shutil; shutil.rmtree('htmlcov', ignore_errors=True); shutil.rmtree('dist', ignore_errors=True); shutil.rmtree('build', ignore_errors=True)"
	python -c "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"
	python -c "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.pyo')]"
	python -c "import pathlib; p = pathlib.Path('.coverage'); p.unlink() if p.exists() else None"

all: clean install-dev check ## Clean, install dev dependencies, and run all checks

run-analysis: ## Run Magic Formula analysis (example)
	@python scripts/run_analysis.py --strategy magic_formula

run-acquirers: ## Run Acquirer's Multiple analysis
	@python scripts/run_analysis.py --strategy acquirers_multiple

run-dca: ## Run DCA analysis
	@python scripts/run_analysis.py --strategy dca

