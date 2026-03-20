.PHONY: install data train test lint api clean all

install:
	pip install --upgrade pip && pip install -r requirements.txt

data:
	python -m src.generate_data

train: data
	python -m src.train

test:
	pytest tests/ -v --tb=short

lint:
	flake8 src/ api/ tests/ --max-line-length=120 --statistics

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

clean:
	rm -rf data/raw/* data/processed/* models/* outputs/* __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

all: install data train test
