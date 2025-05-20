import pickle
import random
from pathlib import Path

import numpy as np
import skopt
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def hyperparameter_tune(
    objective_fn,
    space,
    result_path: Path,
    seed: int,
    top_n: int = 1,
    gp_minimize_opts: dict = None,
    **objective_kwargs,
):
    """
    Run a gp_minimize over `objective_fn`, save the result, and return the best-N trained models.

    Parameters
    ----------
    objective_fn : callable
        Must have signature
            objective_fn(params, *args, model_store, **kwargs) -> float
        and append the trained model into `model_store` each call.
    space : list
        List of skopt.space.Dimension (Integer, Real, Categorical).
    result_path : Path
        Where to save the OptimizeResult pickle.
    seed : int
        Seed for reproducibility (calls seed_everything(seed)).
    top_n : int, default=1
        How many of the best models to return.
    hp_opts : dict, optional
        Extra kwargs for gp_minimize (e.g. n_calls, n_random_starts).
    **objective_kwargs :
        Passed through to `objective_fn`.

    Returns
    -------
    List
        The top_n models, in ascending order of loss.
    """
    seed_everything(seed)

    # Prepare storage for trained models
    model_store = []

    # Define a wrapper that calls your objective, then afterwards stores
    # the model in the model_store, but then just returns the loss.
    # Also have it catch exceptions and convert the 'params' list which
    # skopt passes to it into a dictionary of keyword arguments.
    tuned_parameters = [dim.name for dim in space]

    def _wrapped_obj(params):
        try:
            params = {k: v for k, v in zip(tuned_parameters, params)}
            loss, model = objective_fn(**params, **objective_kwargs)
        except Exception as e:
            print(f"Objective crashed: {e}")
            loss, model = 1e10, None

        model_store.append(model)
        return loss

    # Run gp_minimize
    gp_minimize_opts = gp_minimize_opts or {}
    res = skopt.gp_minimize(_wrapped_obj, space, **gp_minimize_opts)

    # Save the full result (minus the function pointer)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "wb") as f:
        res.specs["args"].pop("func", None)
        pickle.dump(res, f)

    # Pick out the top-n models by lowest loss
    # res.func_vals is an array of losses in the same order as res.x_iters
    ranked_indices = sorted(range(len(res.func_vals)), key=lambda i: res.func_vals[i])[
        :top_n
    ]
    best_models = [model_store[i] for i in ranked_indices]

    # Print the arguments for the best model and loss
    print(f"Best parameters: {res.x}")
    print(f"Best loss: {res.fun}")

    return best_models
