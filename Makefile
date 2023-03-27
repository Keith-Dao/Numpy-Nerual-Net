.PHONY: install update clean test
.DEFAULT_GOAL := default
default:
	@ echo "Please provide a command."

install-env:
	python -m venv .env

install: 
	pip install --upgrade pip
	pip install -r requirements.txt

clean:
	rm -rf .env

install-test: test_requirements.txt 
	pip install -r $<

test: install-test
	pytest -v --cov=neural_net --cov-report term-missing
	flake8 neural_net/
	pylint $$(git ls-files 'neural_net/*.py' | grep -v "test")