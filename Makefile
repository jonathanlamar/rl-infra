CMD=bash

.PHONY: clean
clean:
	sudo rm -rf build .mypy_cache .pytest_cache .ruff_cache dist rl_infra.egg-info && find . | grep -E '(__pycache__|\.pyc|\.pyo$|\.pytest_cache|\.tox)' | sudo xargs rm -rf

.PHONY: install-wheel-local
install-wheel-local:
	pip3 install dist/rl_infra-$(VERSION)-py3-none-any.whl

.PHONY: install-sdist-local
install-sdist-local:
	pip3 install dist/rl-infra-$(VERSION).tar.gz

.PHONY: lint
lint:
	ruff check rl_infra/
	mypy rl_infra/

.PHONY: test
test:
	pytest rl_infra

.PHONY: test-watch
test-watch:
	pytest-watch rl_infra/

.PHONY: wheel
wheel:
	python3 -m build
