# Model

```{eval-rst}
.. autoclass:: imbal.classification.Model
   :exclude-members: add_loss,add_metric,add_variable,add_weight,build,build_from_config,call,compile_from_config,compiled_loss,compute_loss,compute_mask,compute_metrics,compute_output_shape,compute_output_spec,count_params,evaluate,from_config,get_weights,load_own_variables,load_weights,loss,make_predict_function,make_test_function,make_train_function,predict,predict_on_batch,predict_step,quantize,quantized_build,quantized_call,rematerialized_call,reset_metrics,save,save_own_variables,save_weights,symbolic_call,test_on_batch,weights,variables,variable_dtype,trainable_weights,trainable,trainable_variables,train_step,supports_masking,run_eagerly,quantization_mode,path,output,non_trainable_weights,non_trainable_variables,metrics_variables,metrics_names,metrics,losses,layers,jit_compile,input_spec,input_dtype,input,dtype_policy,dtype,distribute_strategy,distribute_reduction_method,compute_dtype,compiled_metrics,train_on_batch,to_json,test_step,summary,stateless_compute_loss,stateless_call,set_weights,set_state_tree,get_state_tree,get_metrics_result,get_layer,get_config,get_compile_config,get_build_config,export
```
## Comparison of Fit Methods on Tabular Data
- A comparison of the performance of the different fit options in this class on tabular data can be found [here](comparison_of_fit_methods_tabular.md).
- A comparison of the performance of the different fit options in this class on tabular data with autoencoder generation enabled can be found [here](comparison_of_ae_methods_tabular.md).

## Comparison of Fit Methods on Image Data
- A comparison of the performance of the different fit options in this class on image data can be found [here](comparison_of_fit_methods_image.md).
- A comparison of the performance of the different fit options in this class on image data with autoencoder generation enabled can be found [here](comparison_of_ae_methods_image.md).