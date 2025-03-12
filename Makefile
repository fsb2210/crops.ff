MAMBA=1
CONDA=0

DEBUG=0

ifeq ($(MAMBA), 1)
env-manager=mamba
endif

ifeq ($(CONDA), 1)
env-manager=conda
endif

ifeq ($(DEBUG), 1)
DBG="--verbose"
endif

CONFIG_DIR=config
ENV_FNAME=env.yml

env:
	@echo "Using ${env-manager} for environment..."
	@echo
	${env-manager} env create -f ${CONFIG_DIR}/${ENV_FNAME}

lint:
	ruff format src/*.py

annotations:
	python src/annotations/make_annotations.py --zip-filename=data/plantdoc.zip \
		--root-dataset-dir=data/plantdoc ${DBG}

run-yolo:
	python src/yolo/run.py -C data/plantdoc_dataset.yaml --model models/yolo11s.pt

inference:
	python src/yolo/inference.py -I data/plantdoc/images/val/three-vibrant-leaves-bird-cherry-tree-13905898.jpg --model runs/detect/train/weights/last.pt
