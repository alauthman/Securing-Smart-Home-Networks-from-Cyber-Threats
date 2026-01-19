.PHONY: install lint format test preprocess train evaluate export docker-build docker-run

install:
	pip install -U pip
	pip install -r requirements-dev.txt
	pip install -e .

lint:
	ruff check .

format:
	black .

test:
	pytest -q

preprocess:
	python scripts/preprocess.py --config configs/example_synth.yaml

train:
	python scripts/train.py --config configs/example_synth.yaml

evaluate:
	python scripts/evaluate.py --config configs/example_synth.yaml

export:
	python scripts/export_model.py --config configs/example_synth.yaml

docker-build:
	docker build -f docker/Dockerfile -t shnet-ml-edge:latest .

docker-run:
	docker run --rm -p 8000:8000 -e SHNET_MODEL_PATH=/app/models/model.joblib -v "$(PWD)/models:/app/models:ro" shnet-ml-edge:latest
