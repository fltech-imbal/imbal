# Classification Tutorials

This page provides an overview of all classification tutorials. The tutorials are organized by **training strategy** (columns) and **modeling approach** (rows).

## Row/Column Headers

* **Regular Training**: Standard model training without class balancing.
* **Balanced Training**: Training with techniques to address class imbalance.
* **cRT / rRT**: Decoupled training where the representation is learned first, followed by classifier/regressor retraining.
* **Autoencoder (AE)**: Enhances representation learning using an autoencoder.
* **Validation Set**: Helps reduce overfitting and enables better estimation of class/sample weights.


|                           | Regular Training                                                                                        | Balanced Training                                                                                         | cRT (Classifier Re-Training)                                                                          |
|---------------------------|---------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| **Basic**                 | [Regular Training](SEP-C/classification/imbal_tutorial_regular_fit_classification_clear_sep.md)         | [Balanced Training](SEP-C/classification/imbal_tutorial_balanced_fit_classification_clear_sep.md)         | [cRT](SEP-C/classification/imbal_tutorial_decoupled_fit_classification_clear_sep.md)                  |
| **With Autoencoder (AE)** | [Regular + AE](SEP-C/classification/imbal_tutorial_regular_fit_ae_classification_clear_sep.md)          | [Balanced + AE](SEP-C/classification/imbal_tutorial_balanced_fit_ae_classification_clear_sep.md)          | [cRT + AE](SEP-C/classification/imbal_tutorial_decoupled_fit_ae_classification_clear_sep.md)          |
| **With Validation Set**   | [Regular + Validation](SEP-C/classification/imbal_tutorial_regular_fit_val_classification_clear_sep.md) | [Balanced + Validation](SEP-C/classification/imbal_tutorial_balanced_fit_val_classification_clear_sep.md) | [cRT + Validation](SEP-C/classification/imbal_tutorial_decoupled_fit_val_classification_clear_sep.md) |

- [Using `imbal.metrics` in model classification](SEP-C/classification/imbal_tutorial_metrics_classification.md)
- [Model representation visualization using t-SNE](SEP-C/classification/imbal_tutorial_tsne_visualization_classification.md)
- [Prediction explanation using LIME](SEP-C/classification/imbal_tutorial_lime_explanation_classification.md)
- [Prediction explanation using SHAP](SEP-C/classification/imbal_tutorial_shap_explanation_classification.md)