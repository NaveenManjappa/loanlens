.PHONY: check test lint run evals smoke

# Format, lint and the unit tests (fake models only). Run before every commit.
check: lint test

lint:
	uv run ruff format .
	uv run ruff check .

test:
	uv run pytest -q

# Start the API locally (http://127.0.0.1:8000/docs).
run:
	uv run uvicorn loanlens.main:app --reload

# Calls the REAL model and costs tokens. Run only when you mean to.
evals:
	uv run python scripts/run_evals.py

# Can LoanLens reach your Azure OpenAI deployment?
smoke:
	uv run python scripts/smoke_test.py
