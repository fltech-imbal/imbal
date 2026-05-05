import numpy as np

def interpolate_class_weights(
    start_weights=(0.1, 0.9),
    end_weights=(0.9, 0.1),
    steps=9
):
    """

    Generates an interpolated range of possible weights for class balancing between some
    desired start and end weights.

    Args:
        start_weights: Optional, default :code:`(0.1, 0.9)`. A tuple or array-like
            of class weights, where the value at each index corresponds to the
            starting class weight for samples with a class label equal to the index.
        end_weights: Optional, default :code:`(0.9, 0.1)`. A tuple or array-like
            of class weights, where the value at each index corresponds to the
            ending class weight for samples with a class label equal to the index.
        steps: Optional, default :code:`9`. The number of steps
            used to generate linearly interpolated class weights.

    Returns:
        A 2D NumPy array, where each row contains a list of class weights.

    Example:

    .. code::

        >>> from imbal.classification import interpolate_class_weights

        >>> class_weights = interpolate_class_weights([1, 2, 3, 4], [4, 3, 2, 1], steps=5)

        >>> print(class_weights)

        [[0.1   0.2   0.3   0.4  ]
         [0.175 0.225 0.275 0.325]
         [0.25  0.25  0.25  0.25 ]
         [0.325 0.275 0.225 0.175]
         [0.4   0.3   0.2   0.1  ]]

    """

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