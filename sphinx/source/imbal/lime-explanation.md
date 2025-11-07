# LIME Model Explanation

Our LIME model explanation implementations are a wrapper around the original *Local Interpretable
Model-Agnostic Explanations* (LIME) implementation, which can be found
[on this GitHub repository](https://github.com/marcotcr/lime). The full LIME documentation
can be found [here](https://lime-ml.readthedocs.io/en/latest/), and the original research
paper can be found [here](https://arxiv.org/abs/1602.04938).
These resources  should be used as reference over our documentation wherever an explanation of the
inner functionalities of LIME are necessary. Our goal is not to provide significant
expansions to LIME's original functionalities, nor provide further insight into LIME's
inner workings, but rather to simply provide a simpler, more streamlined interface
for some of LIME's functionalities.

Below is a list of the functions we have implement which utilize LIME's capabilities:
- [imbal.classification.lime_image_explanation](../imbal/classification/lime_image_explanation.md)
- [imbal.classification.lime_tabular_explanation](../imbal/classification/lime_tabular_explanation.md)
- [imbal.regression.lime_tabular_explanation](../imbal/regression/lime_tabular_explanation.md)