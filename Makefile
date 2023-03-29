python := python
pip := pip

.PHONY: install clean install-test test
.DEFAULT_GOAL := default
default:
	@ echo "Please provide a command."

.env:
	$(python) -m venv .env

install: 
	$(pip) install --upgrade pip
	$(pip) install -r requirements.txt
	$(pip) install -e .

clean:
	git clean -dfX

install-test: test_requirements.txt 
	$(pip) install -r $<

test: install-test
	pytest -v --cov=src --cov-report term-missing
	flake8 src/
	pylint $$(git ls-files 'src/*.py' | grep -v "test")