# Grad-CAM Prediction Explanation

Our Grad-CAM prediction explanation implementations are based on the Keras tutorial found 
[here](https://keras.io/examples/vision/grad_cam/). Additional tweaks to this implementation 
were made to support explanations for classification problems, following 
the original Grad-CAM paper which can be found [here](https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf). 
Also, we extended the idea from the Grad-CAM paper to binary classification and regression.
Our goal is not to provide significant expansions to Grad-CAM's original functionalities, nor 
provide further insight into Grad-CAM's inner workings, but rather to simply provide a simpler, 
more streamlined interface for some of Grad-CAM's functionalities.

Our implementation of Grad-CAM covers three cases:

- Binary classification with one output unit: positive contribution for class 1 and 0.
- Classification with at least 2 output units: positive contribution for each class. A softmax layer is expected.
- Regression: both positive and negative contribution.

All 3 cases look for the last convolutional layer as the feature map, unless provided the name of another convolutional layer.



Below is a list of the functions we have implemented which utilize Grad-CAM's capabilities:
- [imbal.classification.gradcam_explain_image_sample](../imbal/classification/gradcam_explain_image_sample.md)
- [imbal.regression.gradcam_explain_image_sample](../imbal/regression/gradcam_explain_image_sample.md)