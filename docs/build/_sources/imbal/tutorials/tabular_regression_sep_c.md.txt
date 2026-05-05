# Regression Tutorials

This page provides an overview of all regression tutorials. The tutorials are organized by **training strategy** (columns) and **modeling approach** (rows).

|                           | Regular Training                                                                               | Balanced Training                                                                                | rRT (Regressor Re-Training)                                                |
| ------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |----------------------------------------------------------------------------|
| **Basic**                 | [Regular Training](SEP-C/regression/imbal_tutorial_regular_fit_regression_clear_sep.md)               | [Balanced Training](SEP-C/regression/imbal_tutorial_balanced_fit_regression_clear_sep.md)               | [rRT](SEP-C/regression/imbal_tutorial_decoupled_fit_regression_clear_sep.md)      |
| **With Autoencoder (AE)** | [Regular + AE](SEP-C/regression/imbal_tutorial_regular_fit_ae_regression_clear_sep.md)                     | [Balanced + AE](SEP-C/regression/imbal_tutorial_balanced_fit_ae_regression_clear_sep.md)                     | [rRT + AE](SEP-C/regression/imbal_tutorial_decoupled_fit_ae_regression_clear_sep.md)   |
| **With Validation Set**   | [Regular + Validation](SEP-C/regression/imbal_tutorial_regular_fit_val_regression_clear_sep.md) | [Balanced + Validation](SEP-C/regression/imbal_tutorial_balanced_fit_val_regression_clear_sep.md) | [rRT + Validation](SEP-C/regression/imbal_tutorial_decoupled_fit_val_regression_clear_sep.md) |

## Notes

* **Regular Training**: Standard model training without class balancing.
* **Balanced Training**: Training with techniques to address imbalance.
* **rRT**: Decoupled training where representation is learned first, followed by regressor retraining.
* **Autoencoder (AE)**: Enhances representation learning using an autoencoder.
* **Validation Set**: Helps reduce overfitting and enables better estimation of sample weights.

- [Using `imbal.metrics` in model classification](SEP-C/regression/imbal_tutorial_metrics_regression.md)
- [Model representation visualization using t-SNE](SEP-C/regression/imbal_tutorial_tsne_visualization_regression.md)
- [Prediction explanation using LIME](SEP-C/regression/imbal_tutorial_lime_explanation_regression.md)
- [Prediction explanation using SHAP](SEP-C/regression/imbal_tutorial_shap_explanation_regression.md)