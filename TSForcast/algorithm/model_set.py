from algorithm import LCA

model_map = {
    "LCA": LCA,
}


def get_model_class(algorithm_name):
    """Return the algorithm class with the given name."""

    if algorithm_name not in model_map:
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return model_map[algorithm_name].Model
