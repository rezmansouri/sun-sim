import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
from matplotlib.colors import Normalize

# Force TensorFlow to use the CPU and disable Metal backend
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_METAL_USE_MPS"] = "0"
tf.config.set_visible_devices([], 'GPU')

# Function to load HDF5 files
def load_h5_files(directory, file_pattern):
    cubes = []
    for filename in sorted(os.listdir(directory)):
        if filename.startswith(file_pattern) and filename.endswith('.h5'):
            with h5py.File(os.path.join(directory, filename), 'r') as f:
                data = f['Data'][:]  # Do not transpose, keep as is
                cubes.append(data)
    return cubes

# Function to plot the first slice (I, J) of each cube
def plot_first_slices(cubes, title):
    plt.figure(figsize=(10, 10))  # Adjust size for better fit
    for i, cube in enumerate(cubes):
        plt.subplot(len(cubes), 1, i + 1)
        plt.contourf(np.arange(cube.shape[0]), np.arange(cube.shape[1]), cube[:, :, 0].T, cmap='viridis')  # Transpose for (J, I)
        plt.title(f"Cube {i+1}")
        plt.xlabel('I')
        plt.ylabel('J')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Load training data
train_dir = 'train'
train_cubes = load_h5_files(train_dir, 'vr002')

# Plot the first slice of each training cube
plot_first_slices(train_cubes, "First Slice (k=0) of Each Training Cube")

# Data preprocessing (normalize and prepare training data incrementally)
def preprocess_cube(cube):
    cube = cube / np.max(cube)  # Normalize
    X = cube[:, :, :-1].transpose(2, 0, 1)  # Reshape to (k-1, I, J)
    y = cube[:, :, 1:].transpose(2, 0, 1)   # Reshape to (k-1, I, J)
    return X, y

train_X = []
train_y = []
for cube in train_cubes:
    if cube.size == 0:
        print("Warning: Empty cube encountered. Skipping.")
        continue
    X, y = preprocess_cube(cube)
    if X.size == 0 or y.size == 0:
        print("Warning: Preprocessed data is empty. Skipping.")
        continue
    train_X.append(X)
    train_y.append(y)

if len(train_X) == 0 or len(train_y) == 0:
    raise ValueError("Error: No valid training data was generated.")

# Concatenate the preprocessed data
train_X = np.concatenate(train_X, axis=0)  # Stack along the k dimension
train_y = np.concatenate(train_y, axis=0)

# Debugging output
print(f"train_X shape: {train_X.shape}")  # Expected: (k-1, I, J)
print(f"train_y shape: {train_y.shape}")  # Expected: (k-1, I, J)

# Build the model
input_shape = (train_X.shape[1], train_X.shape[2])  # Input shape: (I, J)
model = Sequential([
    LSTM(64, input_shape=input_shape, return_sequences=True),
    Dense(train_X.shape[2])  # Output matches the J dimension
])
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(train_X, train_y, batch_size=1, epochs=10)

# Load test data
test_dir = 'test'
test_cubes = load_h5_files(test_dir, 'vr002')
test_cube = test_cubes[0]
test_cube = test_cube / np.max(test_cube)  # Normalize

# Reshape test data for prediction compatibility
test_cube_X = test_cube[:, :, :-1].transpose(2, 0, 1)  # Reshape to (k-1, I, J)
test_cube_y = test_cube[:, :, 1:].transpose(2, 0, 1)  # Reshape to (k-1, I, J)

# Specify k_predict for intermediate slice prediction
k_predict = 10  # Example: Predict at k=50
if k_predict >= test_cube_X.shape[0]:
    raise ValueError(f"k_predict={k_predict} is out of range for the test data with max k={test_cube_X.shape[0] - 1}.")

# Predict the slice at k_predict incrementally along k
current_slice = test_cube_X[0][np.newaxis, :, :]  # Extract the first slice (k=0) as (1, I, J)
predicted_cube = []
for i in range(1, k_predict + 1):
    prediction = model.predict(current_slice, batch_size=1)
    predicted_cube.append(prediction.squeeze(axis=0))
    current_slice = prediction  # Use the predicted output as the next input
predicted_cube = np.array(predicted_cube)
predicted_cube = predicted_cube.transpose(1, 2, 0)  # Reshape back to (I, J, k)

predicted_k_slice = predicted_cube[:, :, -1]  # Correct indexing for k_predict slice
actual_k_slice = test_cube[:, :, k_predict]  # Correct indexing for k_predict slice
actual_first_slice = test_cube[:, :, 0]  # Slice at k=0

# Plot the actual vs predicted k_predict slice with independent colorbars
plt.figure(figsize=(18, 6))

# Create a normalization object
norm = Normalize(vmin=0.3, vmax=1.0)

plt.subplot(1, 3, 1)
plt.contourf(np.arange(actual_first_slice.shape[0]), np.arange(actual_first_slice.shape[1]), actual_first_slice.T, cmap='viridis', norm=norm)
plt.title(f"Actual Slice (k=0)")
plt.colorbar()
plt.xlabel('I')
plt.ylabel('J')

plt.subplot(1, 3, 2)
plt.contourf(np.arange(actual_k_slice.shape[0]), np.arange(actual_k_slice.shape[1]), actual_k_slice.T, cmap='viridis', norm=norm)
plt.title(f"Actual Slice (k={k_predict})")
plt.colorbar()
plt.xlabel('I')
plt.ylabel('J')

plt.subplot(1, 3, 3)
plt.contourf(np.arange(predicted_k_slice.shape[0]), np.arange(predicted_k_slice.shape[1]), predicted_k_slice.T, cmap='viridis', norm=norm)
plt.title(f"Predicted Slice (k={k_predict})")
plt.colorbar()
plt.xlabel('I')
plt.ylabel('J')

plt.tight_layout()
plt.show()

