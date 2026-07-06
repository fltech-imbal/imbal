# Model

```{eval-rst}
.. autoclass:: imbal.regression.Model
   :exclude-members: add_loss,add_metric,add_variable,add_weight,build,build_from_config,call,compile_from_config,compiled_loss,compute_loss,compute_mask,compute_metrics,compute_output_shape,compute_output_spec,count_params,evaluate,from_config,_generate_decoder,get_weights,load_own_variables,load_weights,loss,make_predict_function,make_test_function,make_train_function,predict,predict_on_batch,predict_step,quantize,quantized_build,quantized_call,rematerialized_call,reset_metrics,save,save_own_variables,save_weights,symbolic_call,test_on_batch,weights,variables,variable_dtype,trainable_weights,trainable,trainable_variables,train_step,supports_masking,run_eagerly,quantization_mode,path,output,non_trainable_weights,non_trainable_variables,metrics_variables,metrics_names,metrics,losses,layers,jit_compile,input_spec,input_dtype,input,dtype_policy,dtype,distribute_strategy,distribute_reduction_method,compute_dtype,compiled_metrics,train_on_batch,to_json,test_step,summary,stateless_compute_loss,stateless_call,set_weights,set_state_tree,get_state_tree,get_metrics_result,get_layer,get_config,get_compile_config,get_build_config,export
```

## Using Multiple Weight Candidates

When a 2D list of sample weights is provided to any fit function in this class, a
special fit will be performed to determine the weights which produce the best results.

In order to compare the performance of the weight candidates, some baseline metric must be used. By default,
weighted `keras.metrics.MeanAbsoluteError` will be used, however, metrics specified in `Model.compile` override this default, with
the following precedence:
1. The first weighted metric specified in the `weighted_metrics` list in `Model.compile`
2. The first metric specified in the `metrics` list in `Model.compile`
3. Default (weighted `keras.metrics.MeanAbsoluteError`)

Note that if a weighted metric is being used, `candidate_evaluation_sample_weight` will be used to compute the specified metric.

The process of performing a fit on multiple weight candidates is described below:

1. The initial weights of the model
2. The model is fit on the first list of sample weights
3. Using the validation set (or training set, if no validation set is specified), the model is evaluated using the first metric passed to `Model.compile` (defaults to `keras.metrics.MeanAbsoluteError` if no metric is specified)
4. The sample weights are recorded, along with the model's weights
5. The initial weights of the model are restored, and this process repeats from step 2 using the next list of sample weights. If a set of sample weights outperforms the previous best, the record of the sample weights and model weights is updated
6. After all fits have been performed, the model weights of the best performing fit are restored, and the best performing sample weights are saved in `Model.best_sample_weights`

## A Note on Validation Data

Using validation data is a good strategy for reducing overfitting (by means of using a callback such as `keras.callbacks.EarlyStopping`),
however, one downside of using validation data is that your model does not have access to the entire set of possible
data for training, as the withheld validation data is not used to update the model's weights.

As a remedy for this, whenever validation data is specified for training, we include an additional fit at the
end of the training process, combining the validation data with the training data, and training for a number
of epochs that produces the best results when the validation data was held out. This is also compatible with
using multiple weight candidates, where the candidate that produced the best results on the validation data
will be used for the final model fit.

## Comparison of Fit Methods on Tabular Data
- A comparison of the performance of the different fit options in this class on tabular data can be found [here](comparison_of_fit_methods_tabular.md).
- A comparison of the performance of the different fit options in this class on tabular data with autoencoder generation enabled can be found [here](comparison_of_ae_methods_tabular.md).

## Comparison of Fit Methods on Image Data
- A comparison of the performance of the different fit options in this class on image data can be found [here](comparison_of_fit_methods_image.md).