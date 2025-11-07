import sys
import os
import keras_tuner
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Layer, Dropout
from tensorflow.keras import layers
from tensorflow.keras.layers import PReLU


# Get the absolute path of the current script's directory
current_dir = os.path.dirname(os.path.abspath("transformer0.ipynb"))

# Get the absolute path of the parent directory (project_folder)
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to the Python path
sys.path.append(parent_dir)

from FNN1_1 import baseline_deviation, baeline_out_deviation, baseline_long_deviation, baseline_relError, absSum
baseline_out_deviation = baeline_out_deviation

from GetXY import x_train, y_train, x_val, y_val, x_test, y_test, out_x_test, out_y_test, long_x_test, long_y_test, outsideExpr, absSum

early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
    monitor='val_loss',
    mode = "min"
)

batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train)).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)


input_shape = x_train[0].shape
from tensorflow.keras import backend as K

def build_model(hp, input_shape):
    K.clear_session()
    num_neurons = hp.Int("num_neurons", 1, 512)
    num_layers = hp.Int("num_layers", 1, 16)
    dropoutTF = hp.Boolean("dropoutT/F")
    lil_model = keras.Sequential()
    lil_model.add(keras.Input(shape=input_shape))
    lil_model.add(layers.Flatten())
    for i in range(num_layers):
        lil_model.add(layers.Dense(num_neurons)),
        lil_model.add(PReLU())
        if dropoutTF == True:
            lil_model.add(layers.Dropout(0.1))
    lil_model.add(layers.Dense(1, activation='linear'))
    lil_model.compile(optimizer="adam", loss="mse")
    return lil_model
build_model(keras_tuner.HyperParameters(), input_shape)

x_test_dataset = tf.data.Dataset.from_tensor_slices(x_test).batch(batch_size)
out_x_test_dataset = tf.data.Dataset.from_tensor_slices(out_x_test).batch(batch_size)
long_x_test_dataset = tf.data.Dataset.from_tensor_slices(long_x_test).batch(batch_size)

tuner = keras_tuner.BayesianOptimization(
    hypermodel=lambda hp: build_model(hp, input_shape),    objective="val_loss",
    max_trials=100,
    executions_per_trial=1,
    overwrite=False,
    directory="FNNTuner",
    project_name="tuner",
)

num_epochs = 100
tuner.search(train_dataset, epochs = num_epochs, validation_data = (val_dataset), verbose = 1, callbacks = [])

best_hps = tuner.get_best_hyperparameters()[0]


benchmarks= []
MAEinRange = []
MREinRange = []
MAEoutRange = []
MREoutRange = []
MAElongRange = []


for progress in range(10):
    
    best_model = build_model(best_hps, input_shape)

    best_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=200,
        #callbacks=[early_stopping],
        verbose=0
    )
    
    predsInRange = best_model.predict(x_test_dataset)
    predsOutRange = best_model.predict(out_x_test_dataset)
    predsLongRange = best_model.predict(long_x_test_dataset)
    
    import numpy as np
    
    reldiffInRange = []
    diffInRange = []
    safe_y_test = np.where(np.isclose(y_test,0.0), 1.0, y_test)
    
    for i in range(len(y_test)):
        diffInRange.append(abs(y_test[i] - predsInRange[i]))
        reldiffInRange.append(abs(y_test[i] - predsInRange[i])/abs(safe_y_test[i]))
    print(len(diffInRange))
    print("MAE in Range: ", np.mean(diffInRange))
    print("MRE in Range: ", np.mean(reldiffInRange))
    MAEinRange.append(np.mean(diffInRange))
    MREinRange.append(np.mean(reldiffInRange))

    diffLongRange = []
    for i in range(200, 300):
        diffLongRange.append(np.array(np.abs(long_y_test[i]) - np.array(predsLongRange[i])))
        
    NEEDdiffLongRange = []
    for i in range(len(long_y_test)):
        NEEDdiffLongRange.append(np.array(np.abs(long_y_test[i]) - np.array(predsLongRange[i])))
    print("MAE longer Expressions: ", np.mean(NEEDdiffLongRange))
    MAElongRange.append(np.mean(NEEDdiffLongRange))
    
    diffOutRange = []
    for i in range(len(out_y_test)):
        diffOutRange.append(abs(out_y_test[i] - predsOutRange[i]))
    safe_out_y_test = np.where(out_y_test == 0, 1, out_y_test)
    diff_out_relError = []
    for i in range(len(out_y_test)):
        diff_out_relError.append(abs(diffOutRange[i] / safe_out_y_test[i]))
    print("MAE out Range: ", np.mean(diffOutRange))
    print("MRE out Range: ", np.mean(diff_out_relError))
    MAEoutRange.append(np.mean(diffOutRange))
    MREoutRange.append(np.mean(diff_out_relError))
    
    placeholder = absSum(outsideExpr)
    diffOutRange = []
    indices_with_placeholder_22 = [i for i, val in enumerate(placeholder) if val == 22] 
    
    for i in indices_with_placeholder_22:
        diffOutRange.append(np.abs(out_y_test[i]-predsOutRange[i]))
    meanDiff_InRange = np.mean(diffInRange)
    meanDiff_OutRange = np.mean(diffOutRange)
    meanDiff_LongRange = np.mean(diffLongRange)
    meanDiff_OutRelRange = np.mean(diff_out_relError)
    
    benchmark = 0
    benchmark += baseline_deviation / (meanDiff_InRange**2) / 4
    print(baseline_deviation / (meanDiff_InRange**2) / 4)
    
    benchmark += baseline_out_deviation / (meanDiff_OutRange**2) / 4
    print(baseline_out_deviation / (meanDiff_OutRange**2) / 4)
    
    benchmark += baseline_long_deviation / (meanDiff_LongRange**2) / 4
    print(baseline_long_deviation / (meanDiff_LongRange**2) / 4)
    
    benchmark += baseline_relError / (meanDiff_OutRelRange**2) / 4
    print(baseline_relError / (meanDiff_OutRelRange**2) / 4)
    
    print(f"Benchmark: {benchmark}")
    benchmarks.append(benchmark)
    print(f"progress: {progress}")


from scipy.stats import kstest, uniform

def ks_uniform_custom(data):
    data = np.asarray(data)
    if data.size == 0:
        raise ValueError("Input data is empty.")
    max_val = np.max(data)
    if max_val <= 0:
        raise ValueError("Maximum value must be positive for Uniform(0, max).")
    return kstest(data, uniform(loc=0, scale=max_val).cdf)

stats1, p_value1 = ks_uniform_custom(MAEinRange)
stats2, p_value2 = ks_uniform_custom(MREinRange)
stats3, p_value3 = ks_uniform_custom(MAEoutRange)
stats4, p_value4 = ks_uniform_custom(MREoutRange)
stats5, p_value5 = ks_uniform_custom(MAElongRange)
stats6, p_value6 = ks_uniform_custom(benchmarks)

print(f"MAE in Range P-value: {p_value1}")
print(f"MRE in Range P-value: {p_value2}")
print(f"MAE out Range P-value: {p_value3}")
print(f"MRE out Range P-value: {p_value4}")
print(f"MAE long Range P-value: {p_value5}")
print(f"benchmark P-value: {p_value6}")

print(f"average MAE in Range: {np.mean(MAEinRange)}")
print(f"average MRE in Range: {np.mean(MREinRange)}")
print(f"average MAE out Range: {np.mean(MAEoutRange)}")
print(f"average MRE out Range: {np.mean(MREoutRange)}")
print(f"average MAE long Range: {np.mean(MAElongRange)}")
print(f"average benchmark: {np.mean(benchmarks)}")

print(f"MAE in Range: {MAEinRange}")
print(f"MRE in Range: {MREinRange}")
print(f"MAE out Range: {MAEoutRange}")
print(f"MRE out Range: {MREoutRange}")
print(f"MAE long Range: {MAElongRange}")
print(f"benchmark: {benchmarks}")



