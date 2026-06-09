.PHONY: install test week1 clean

install:
	pip install -e .

test:
	pytest -q

week1:
	bash scripts/run_week1_mre.sh

clean:
	rm -rf outputs .pytest_cache **/__pycache__
