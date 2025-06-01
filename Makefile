.PHONY: setup clean preprocess train compile-results splitting-test kl-test baseline-test all

# Install project dependencies
setup:
	pip install -r requirements.in

# Remove previous outputs
clean:
	rm -rf data/interim data/processed models plots tables __pycache__

preprocess:
	jupyter nbconvert --to notebook --execute 00-preprocess-data.ipynb

# Train targets
train-synthetic: preprocess
	jupyter nbconvert --to notebook --execute 01-train-on-synthetic-data.ipynb

train-real: preprocess
	jupyter nbconvert --to notebook --execute 02-train-on-real-data.ipynb

train-regularisation: preprocess
	jupyter nbconvert --to notebook --execute 03-train-regularisation-demo.ipynb

# Aggregate training
train: train-synthetic train-real train-regularisation

# Compile results from training notebooks
compile-results-synthetic: train-synthetic
	jupyter nbconvert --to notebook --execute 04-compile-results-synthetic-data.ipynb

compile-results-real: train-real
	jupyter nbconvert --to notebook --execute 05-compile-results-real-data.ipynb

compile-results-regularisation: train-regularisation
	jupyter nbconvert --to notebook --execute 06-compile-results-regularisation-demo.ipynb

# Run all the compilation notebooks
compile-results: compile-results-synthetic compile-results-real compile-results-regularisation

splitting-test: preprocess
	jupyter nbconvert --to notebook --execute 07-dataset-splitting-sensitivity-test.ipynb

kl-test: preprocess train-synthetic
	jupyter nbconvert --to notebook --execute 08-kl-regularisation-sensitivity-test.ipynb

baseline-test: preprocess
	jupyter nbconvert --to notebook --execute 09-baseline-sensitivity-test.ipynb

all: clean preprocess train compile-results splitting-test kl-test baseline-test