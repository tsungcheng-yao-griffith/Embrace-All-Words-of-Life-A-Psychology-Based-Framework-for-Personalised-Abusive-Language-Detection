from keras.models import Model
from keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Concatenate
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import numpy as np
import csv
import tensorflow as tf

# Load text data from CSV file
with open('', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    text_data = [row[0] for row in reader]

# Load numeric data from CSV file
numeric_data = np.genfromtxt('', delimiter=',')

# Load labels from CSV file
labels = np.genfromtxt('', dtype=np.int32)

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_data)
text_data = tokenizer.texts_to_sequences(text_data)
max_text_length = max([len(x) for x in text_data])
text_data = pad_sequences(text_data, maxlen=max_text_length)


# Load pre-trained word vectors
word_vectors = '../crawl-300d-2M.vec' #can be found on Fasttext page

# Create an embedding matrix
num_words = len(tokenizer.word_index) + 1
embedding_dim = 300
embedding_matrix = np.zeros((num_words, embedding_dim))
with open(word_vectors, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        if word in tokenizer.word_index:
            embedding_matrix[tokenizer.word_index[word]] = np.asarray(values[1:], dtype='float32')

# Define the CNN model
text_input = Input(shape=(max_text_length,))
text_embedded = Embedding(input_dim=num_words, output_dim=embedding_dim, weights=[embedding_matrix], trainable=True)(text_input)
text_conv1 = Conv1D(filters=128, kernel_size=3, activation='relu')(text_embedded)
text_pool1 = GlobalMaxPooling1D()(text_conv1)
text_conv2 = Conv1D(filters=64, kernel_size=4, activation='relu')(text_embedded)
text_pool2 = GlobalMaxPooling1D()(text_conv2)
text_concat = Concatenate()([text_pool1, text_pool2])

num_input = Input(shape=(numeric_data.shape[1],))
num_dense1 = Dense(64, activation='relu')(num_input)
num_dense2 = Dense(32, activation='relu')(num_dense1)
num_dense3 = Dense(16, activation='relu')(num_dense2)

combined = Concatenate()([text_concat, num_dense3])
output = Dense(1, activation='sigmoid')(combined)

model = Model(inputs=[text_input, num_input], outputs=output)

model.summary()


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy')

model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss',  # Monitor the validation loss for early stopping
    patience=5,           # Number of epochs with no improvement before stopping
    restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
)



# Train the model
model.fit([text_data, numeric_data], labels, epochs=150, batch_size=32, callbacks=[early_stopping])

model.save('')