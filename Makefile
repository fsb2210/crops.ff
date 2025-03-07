MAMBA=1
CONDA=0

ifeq ($(MAMBA), 1)
env-manager=mamba
endif

ifeq ($(CONDA), 1)
env-manager=conda
endif

CONFIG_DIR=config
ENV_FNAME=env.yml

env:
	@echo "Using ${env-manager} for environment..."
	@echo
	${env-manager} env create -f ${CONFIG_DIR}/${ENV_FNAME}

lint:
	ruff format src/*.py

run-yolo:
	python src/yolo/run.py --channels=3 --num-classes=29 --mode=training \
		--train-fname=data/plantdoc/train_labels.csv \
		--image-dir=data/plantdoc/images/train \
		--labels-dict=data/plantdoc/classes_ids.json
