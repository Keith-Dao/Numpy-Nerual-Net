python := python
pip := pip
pip_flags := --require-virtualenv

.PHONY: install clean install-test test
.DEFAULT_GOAL := default
default:
	@ echo "Please provide a command."

.env:
	$(python) -m venv .env

install: 
	$(pip) $(pip_flags) install --upgrade pip
	$(pip) $(pip_flags) install -r requirements.txt
	$(pip) $(pip_flags) install -e .

clean:
	git clean -dfX

install-test: test_requirements.txt 
	$(pip) $(pip_flags) install -r $<

test: install-test
	pytest -v --cov=src --cov-report term-missing
	flake8 src/
	pylint src/