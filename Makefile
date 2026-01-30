lint:
	ruff check mcphero --fix
	ruff format mcphero

lint-tests:
	ruff check tests --fix
	ruff format tests

typecheck:
	pyright mcphero

test:
	pytest tests
