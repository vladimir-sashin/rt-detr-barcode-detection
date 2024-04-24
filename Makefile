.PHONY: *

# If OS is Windows, path to Python 3.10 executable must be passed in `PYTHON_EXEC` var to `make setup_ws`
# Because poetry fails on `poetry env use pythonX.Y` command on Windows
# https://github.com/python-poetry/poetry/issues/2117
ifndef PYTHON_EXEC
override PYTHON_EXEC = python3.10
endif


# These lines ensure that CTRL+B can be used to jump to definitions in
# code of installed modules on Linux/OSX.
# Explained here: https://github.com/jupyter-lsp/jupyterlab-lsp/blob/39ee7d93f98d22e866bf65a80f1050d67d7cb504/README.md?plain=1#L175
ifeq ($(OS),Windows_NT)
	CREATE_LINK := @echo "Skipped lsp symlink setup for Windows"
else
	CREATE_LINK := ln -s / .lsp_symlink || true  # Create if does not exist.
endif


# ================== LOCAL WORKSPACE SETUP ==================

install_venv:
	@echo "=== Installing project dependencies ==="
	poetry env use $(PYTHON_EXEC)
	poetry install --with CI
	@echo "Virtual environment has been created."
	@echo "Path to virtual environment:"
	poetry env info -p

pre_commit_install:
	@echo "=== Installing pre-commit ==="
	poetry run pre-commit install

setup_ws:
	$(MAKE) install_venv $(PYTHON_EXEC)
	$(MAKE) pre_commit_install
