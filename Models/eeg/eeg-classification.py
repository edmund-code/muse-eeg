import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import preprocess_eeg as ppEEG
import matplotlib.pyplot as plt
import umap
import timeit
from keras.models import Model


# Parameters
# n_samples = 100
# n_timesteps = 60
# n_features = 4  # Now we have 4 channels in the time series data
# n_classes = 2   # Example with two classes

# Generate synthetic multivariate time series data
# X = np.random.random((n_samples, n_timesteps, n_features))
# y = np.random.randint(0, n_classes, n_samples)

X, y = ppEEG.getData()

X = np.nan_to_num(X)
y = np.nan_to_num(y)

n_timesteps = X.shape[1]
n_features = X.shape[2]  # Now we have 4 channels in the time series data
n_classes = 5   # Example with two classes

# Encode class values as integers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Convert integers to dummy variables (i.e. one hot encoded)
y_dummy = to_categorical(y_encoded)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_dummy, test_size=0.2, random_state=42)

# Define the model
model = Sequential()
model.add(Conv1D(filters=24, kernel_size=8, activation='relu', input_shape=(n_timesteps, n_features)))
model.add(Conv1D(filters=24, kernel_size=8, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=3))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(n_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), 
              metrics=['accuracy']) # tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.F1Score()

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1, validation_data=(X_test, y_test))

# Evaluate the model
#loss, accuracy, precision, recall, f1 = model.evaluate(X_test, y_test, verbose=0)
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

plotUMAP = False
if plotUMAP:
    umap_hparams = {'n_neighbors':10,
                    'min_dist':0.1,
                    'n_components':2,
                    'metric':'euclidean'}


    # label = np.argmax(y_test, axis=1) #all labels

    # intermediate_layer_model = Model(inputs=model.input,
    #                                  outputs=model.layers[-2].output)
    # embedding = intermediate_layer_model.predict(X_test)

    label = y_encoded #all labels

    intermediate_layer_model = Model(inputs=model.input,
                                    outputs=model.layers[-2].output)
    embedding = intermediate_layer_model.predict(X)

    fig, ax = plt.subplots(figsize=(6,5), constrained_layout=False)
    ax.set(xticks=[], yticks=[])

    umap_embedding = umap.UMAP(n_neighbors=umap_hparams['n_neighbors'], min_dist=umap_hparams['min_dist'], n_components=umap_hparams['n_components'], metric=umap_hparams['metric']).fit_transform(embedding)
    scatter = ax.scatter(x = umap_embedding[:,0], y = umap_embedding[:,1], s=2, c=label, cmap='tab10')

    cbar = plt.colorbar(scatter, boundaries=np.arange(6)-0.5)
    cbar.set_ticks(np.arange(5))
    cbar.set_ticklabels(np.arange(5))

    plt.title('UMAP of EEG Classes', fontsize=14, fontweight='bold')
    plt.xlabel("UMAP dim 1")
    plt.ylabel("UMAP dim 2")
    plt.show()


y_pred = model.predict(X)
cm = tf.math.confusion_matrix(np.argmax(y_pred, axis=1), np.argmax(y_dummy, axis=1))

print(cm)
