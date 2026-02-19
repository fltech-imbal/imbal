import numpy as np

def interpolate_class_weights(
    start_weights=None,
    end_weights=None,
    steps=9
):
    if end_weights is None:
        end_weights = [0.9, 0.1]
    if start_weights is None:
        start_weights = [0.1, 0.9]

    start_weights = np.array(start_weights)
    start_weights = start_weights / np.sum(start_weights)
    end_weights = np.array(end_weights)
    end_weights = end_weights / np.sum(end_weights)

    if not start_weights.shape == end_weights.shape:
        raise RuntimeError("start_weights and end_weights must have the same shape")

    def lerp(starts, ends, step_array):
        step_array = np.array(step_array).reshape(-1, 1)
        value =  starts * (1-step_array) + ends*step_array
        return value

    step_values = [x / (steps - 1) for x in range(steps)]
    return lerp(start_weights, end_weights, step_values)

print(interpolate_class_weights([1,2,3,4],[4,3,2,1]))