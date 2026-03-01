from imbal.regression import
from imbal.classification import interpolate_class_weights

print(generate_sample_weight_contenders([1, 0.1, 0.01]))
print(interpolate_class_weights([1, 2, 3, 4], [4, 3, 2 ,1]))