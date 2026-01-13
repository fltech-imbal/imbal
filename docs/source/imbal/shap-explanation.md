# SHAP Model Explanation

Our SHAP model explanation implementations are a wrapper around the original *SHapley Additive
exPlanations* (SHAP) implementation, which can be found
[on this GitHub repository](https://github.com/shap/shap). The full SHAP documentation
can be found [here](https://shap.readthedocs.io/en/latest/index.html), and the original research
paper can be found [here](https://arxiv.org/abs/1705.07874).
These resources should be used as reference over our documentation wherever an explanation of the
inner functionalities of SHAP are necessary. Our goal is not to provide significant
expansions to SHAP's original functionalities, nor provide further insight into SHAP's
inner workings, but rather to simply provide a simpler, more streamlined interface
for some of SHAP's functionalities.

One difference that separates SHAP from [LIME](lime-explanation.md) is that SHAP includes
the ability to perform explanations across an entire dataset, along with per-sample explanations,
whereas LIME only contains the capability to perform per-sample explanations out of the box.

At the moment, our package does not implement wrappers for SHAP's text classification
capabilities. The main goal of this package is to be used for space research applications,
and we felt that this field would have little use for this capability, though we may
add these wrappers later on.

Additionally, at this time we do not support explanations for image-based regression models, as
SHAP does not inherently support this type of explanation by default. We may implement
this capability at a later date.

Below is a list of the functions we have implement which utilize SHAP's capabilities:
- [imbal.classification.shap_explain_image_sample](../imbal/classification/shap_explain_image_sample.md)
- [imbal.classification.shap_explain_tabular_sample](../imbal/classification/shap_explain_tabular_sample.md)
- [imbal.classification.shap_explain_tabular_dataset](../imbal/classification/shap_explain_tabular_dataset.md)
- [imbal.regression.shap_explain_tabular_sample](../imbal/regression/shap_explain_tabular_sample.md)
- [imbal.regression.shap_explain_tabular_dataset](../imbal/regression/shap_explain_tabular_dataset.md)