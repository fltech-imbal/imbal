# Regression Tutorials

This page provides an overview of all regression tutorials. The tutorials are organized by **training strategy** (columns) and **modeling approach** (rows).

|                           | Regular Training                                                                               | Balanced Training                                                                                | rRT (Regressor Re-Training)                                                |
| ------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |----------------------------------------------------------------------------|
| **Basic**                 | [Regular Training](./Regular/imbal_tutorial_regular_fit_regression_clear_sep.md)               | [Balanced Training](./Regular/imbal_tutorial_balanced_fit_regression_clear_sep.md)               | [rRT](./Regular/imbal_tutorial_decoupled_fit_regression_clear_sep.md)      |
| **With Autoencoder (AE)** | [Regular + AE](./AE/imbal_tutorial_regular_fit_ae_regression_clear_sep.md)                     | [Balanced + AE](./AE/imbal_tutorial_balanced_fit_ae_regression_clear_sep.md)                     | [rRT + AE](./AE/imbal_tutorial_decoupled_fit_ae_regression_clear_sep.md)   |
| **With Validation Set**   | [Regular + Validation](./ValidationSet/imbal_tutorial_regular_fit_val_regression_clear_sep.md) | [Balanced + Validation](./ValidationSet/imbal_tutorial_balanced_fit_val_regression_clear_sep.md) | [rRT + Validation](./ValidationSet/imbal_tutorial_decoupled_fit_val_regression_clear_sep.md) |

## Notes

* **Regular Training**: Standard model training without class balancing.
* **Balanced Training**: Training with techniques to address imbalance.
* **rRT**: Decoupled training where representation is learned first, followed by regressor retraining.
* **Autoencoder (AE)**: Enhances representation learning using an autoencoder.
* **Validation Set**: Helps reduce overfitting and enables better estimation of sample weights.
