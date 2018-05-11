import numpy as np
from sklearn import preprocessing

input_data = np.array([[5.1, -2.9, 3.3],
                       [-1.2, 7.8, -6.1],
                       [3.9, 0.4, 2.1],
                       [7.3, -9.9, -4.5]])

data_binarized = preprocessing.Binarizer(threshold=2.1).transform(input_data)

print(data_binarized)
print("===BEFORE===")
print ("    MEAN: ",input_data.mean(axis=0))
print("    Standard deviation:", input_data.std(axis=0))

data_scaled = preprocessing.scale(input_data)

print("===AFTER===")
print ("    MEAN: ",data_scaled.mean(axis=0))
print("    Standard deviation:", data_scaled.std(axis=0))

data_scaler_min_max = preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaled_min_max = data_scaler_min_max.fit_transform(input_data)
print("\n MIN MAX scaled data:",data_scaled_min_max)

data_normalized_l1 = preprocessing.normalize(input_data,norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data,norm='l2')

print("\nNormalized L1:", data_normalized_l1)
print("Normalized L2:", data_normalized_l2)