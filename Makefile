.PHONY: conda format style types black test link check notebooks
.DEFAULT_GOAL = help

PYTHON = python
VERSION = 3.8
NAME = py_name
ROOT = ./
PIP = pip
CONDA = conda
SHELL = bash
PKGROOT = gpjax

help:	## Display this help
		@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)


##@ Formatting
black:  ## Format code in-place using black.
		black ${PKGROOT}/ tests/ -l 88 .

isort:  ## Format imports in-place using isort.
		isort ${PKGROOT}/ tests/

format: ## Code styling - black, isort
		black ${PKGROOT}/ tests/ -l 88 .
		@printf "\033[1;34mBlack passes!\033[0m\n\n"
		isort ${PKGROOT}/ tests/
		@printf "\033[1;34misort passes!\033[0m\n\n"

##@ Testing
test:  ## Test code using pytest.
		@printf "\033[1;34mRunning tests with pytest...\033[0m\n\n"
		pytest -v gpjax tests
		@printf "\033[1;34mPyTest passes!\033[0m\n\n"
