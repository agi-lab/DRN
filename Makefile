.PHONY: all setup clean train-synthetic train-real train-regularisation train compile-results

# Default: clean, install deps, train, then compile results
all: clean setup train compile-results

# Install project dependencies
setup:
	pip install -r requirements.in

# Remove previous outputs
clean:
	rm -rf models plots __pycache__

# Train targets
train-synthetic:
	jupyter nbconvert --to notebook --execute 01-train-on-synthetic-data.ipynb

train-real:
	jupyter nbconvert --to notebook --execute 02-train-on-real-data.ipynb

train-regularisation:
	jupyter nbconvert --to notebook --execute 03-train-regularisation-demo.ipynb

# Aggregate training
train: train-synthetic train-real train-regularisation

# Compile results from training notebooks
compile-results-synthetic:
	jupyter nbconvert --to notebook --execute 04-compile-results-synthetic-data.ipynb

compile-results-real:
	jupyter nbconvert --to notebook --execute 05-compile-results-real-data.ipynb

compile-results-regularisation:
	jupyter nbconvert --to notebook --execute 06-compile-results-regularisation-demo.ipynb

# Run all the compilation notebooks
compile-results: compile-results-synthetic compile-results-real compile-results-regularisation