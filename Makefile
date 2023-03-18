.PHONY: install update clean
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