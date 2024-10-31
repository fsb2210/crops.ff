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

test:
	python src/run.py --dataset-directory=data/plants/color --mode=Train
