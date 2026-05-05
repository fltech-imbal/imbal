# Classification Tutorials

This page provides an overview of all classification tutorials. The tutorials are organized by **training strategy** (columns) and **modeling approach** (rows).

|                           | Regular Training                                                                                   | Balanced Training                                                                                    | cRT (Classifier Re-Training)                                                                     |
|---------------------------|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| **Basic**                 | [Regular Training](./Regular/imbal_tutorial_regular_fit_classification_clear_sep.md)               | [Balanced Training](./Regular/imbal_tutorial_balanced_fit_classification_clear_sep.md)               | [cRT](./Regular/imbal_tutorial_decoupled_fit_classification_clear_sep.md)                        |
| **With Autoencoder (AE)** | [Regular + AE](./AE/imbal_tutorial_regular_fit_ae_classification_clear_sep.md)                     | [Balanced + AE](./AE/imbal_tutorial_balanced_fit_ae_classification_clear_sep.md)                     | [cRT + AE](./AE/imbal_tutorial_decoupled_fit_ae_classification_clear_sep.md)                     |
| **With Validation Set**   | [Regular + Validation](./ValidationSet/imbal_tutorial_regular_fit_val_classification_clear_sep.md) | [Balanced + Validation](./ValidationSet/imbal_tutorial_balanced_fit_val_classification_clear_sep.md) | [cRT + Validation](./ValidationSet/imbal_tutorial_decoupled_fit_val_classification_clear_sep.md) |

## Notes

* **Regular Training**: Standard model training without class balancing.
* **Balanced Training**: Training with techniques to address class imbalance.
* **cRT / rRT**: Decoupled training where the representation is learned first, followed by classifier/regressor retraining.
* **Autoencoder (AE)**: Enhances representation learning using an autoencoder.
* **Validation Set**: Helps reduce overfitting and enables better estimation of class/sample weights.
