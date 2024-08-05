import fasttext # original Facebook version
import numpy as np
from keras.models import Model
from keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout, Concatenate, Input
import csv
import tensorflow as tf

# Load pre-trained FastText model
model_fasttext = fasttext.load_model('')  # Replace with the path to your pre-trained FastText model

# Function to extract FastText features
def extract_fasttext_features(text):
    # Extract FastText word embeddings for each word in the preprocessed text
    embeddings = [model_fasttext.get_word_vector(word) for word in text]
    return embeddings

# Load text data from CSV file
with open('', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    text_data = [row[0] for row in reader]

numeric_data = np.genfromtxt('', delimiter=',')

# Load labels from CSV file
labels = np.genfromtxt('', dtype=np.int32)

# Extract FastText features
text_features = extract_fasttext_features(text_data)

text_features = np.array(text_features)

num_classes = 2  # Number of classes in your classification task

sequence_length = text_features.shape[0]  # Length of the extracted FastText features
embedding_dim = text_features.shape[1]  # Dimensionality of the extracted FastText features


text_input = Input(shape=(105,1))
text_conv1 = Conv1D(filters=128, kernel_size=3, activation='relu')(text_input)
text_pool1 = GlobalMaxPooling1D()(text_conv1)
text_conv2 = Conv1D(filters=64, kernel_size=4, activation='relu')(text_input)
text_pool2 = GlobalMaxPooling1D()(text_conv2)
text_concat = Concatenate()([text_pool1, text_pool2])

num_input = Input(shape=(numeric_data.shape[1],))
num_dense1 = Dense(64, activation='relu')(num_input)
num_dense2 = Dense(32, activation='relu')(num_dense1)
num_dense3 = Dense(16, activation='relu')(num_dense2)

combined = Concatenate()([text_concat, num_dense3])
output = Dense(1, activation='sigmoid')(combined)

model = Model(inputs=[text_input, num_input], outputs=output)

# Print model summary
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Reshape the input tensors to remove the extra dimension
#text_features = np.expand_dims(text_features, axis=0)
#labels = np.expand_dims(labels, axis=0)

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss',  # Monitor the validation loss for early stopping
    patience=5,           # Number of epochs with no improvement before stopping
    restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
)

# Train and evaluate the model
model.fit([text_features, numeric_data], labels, epochs=100, batch_size=32, callbacks=[early_stopping])

# Save the model
model.save('')
