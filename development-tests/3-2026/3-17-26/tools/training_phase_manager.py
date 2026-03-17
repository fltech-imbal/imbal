import tensorflow as tf

class TrainingPhaseManager:
    """
    Manages the training phase flag to switch between training and validation modes.
    This class encapsulates the `is_training` state, making it easier to integrate
    with the custom loss function and callback.
    """

    def __init__(self):
        self.is_training = True

    def set_training(self, is_training: bool) -> None:
        """
        Sets the current phase to training or validation.

        Args:
            is_training (bool): True if training phase, False if validation/testing phase.
        """
        self.is_training = is_training

    def is_training_phase(self) -> bool:
        """
        Returns whether the current phase is training.

        Returns:
            bool: True if in training phase, False otherwise.
        """
        return self.is_training

class IsTraining(tf.keras.callbacks.Callback):
    """
    Custom Keras callback to update the training phase flag in the TrainingPhaseManager object.
    """

    def __init__(self, phase_manager: TrainingPhaseManager):
        """
        Initializes the callback with a reference to the TrainingPhaseManager.

        Args:
            phase_manager (TrainingPhaseManager): The manager that tracks the training phase.
        """
        super().__init__()
        self.phase_manager = phase_manager

    def on_train_batch_begin(self, batch, logs=None) -> None:
        """
        Called at the beginning of each training batch.
        """
        self.phase_manager.set_training(True)

    def on_test_batch_begin(self, batch, logs=None) -> None:
        """
        Called at the beginning of each validation batch.
        """
        self.phase_manager.set_training(False)