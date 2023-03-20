.PHONY: install update clean test
.DEFAULT_GOAL := default
default:
	@ echo "Please provide a command."

install:
	python -m venv .env

update: 
	pip install --upgrade pip
	pip install -r requirements.txt

clean:
	rm -rf .env

install-test: test_requirements.txt 
	pip install -r $<

test: install-test
	pytest -v