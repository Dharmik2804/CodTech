# 📝 TASK 2: NLP Classification using TensorFlow (IMDb Sentiment)

# ✅ STEP 1: IMPORT LIBRARIES
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, GlobalAveragePooling1D
import matplotlib.pyplot as plt

# ✅ STEP 2: LOAD AND PREPROCESS DATA

# Load IMDb dataset (top 10,000 most frequent words)
vocab_size = 10000
max_len = 200

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Pad sequences to ensure same length
x_train = pad_sequences(x_train, maxlen=max_len, padding='post')
x_test = pad_sequences(x_test, maxlen=max_len, padding='post')

# ✅ STEP 3: BUILD MODEL

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len),
    LSTM(64, return_sequences=True),
    GlobalAveragePooling1D(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Model Summary
model.summary()

# ✅ STEP 4: COMPILE MODEL

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# ✅ STEP 5: TRAIN MODEL

history = model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=5,
    batch_size=64
)

# ✅ STEP 6: PLOT ACCURACY & LOSS

def plot_metrics(history):
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label="Train Accuracy")
    plt.plot(history.history['val_accuracy'], label="Val Accuracy")
    plt.title("Accuracy vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label="Train Loss")
    plt.plot(history.history['val_loss'], label="Val Loss")
    plt.title("Loss vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()

plot_metrics(history)

# ✅ STEP 7: EVALUATE MODEL

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n✅ Final Test Accuracy: {test_acc:.4f}")

# ✅ STEP 8 (Optional): SAVE MODEL

model.save("imdb_sentiment_model.h5")
print("✅ Model saved as 'imdb_sentiment_model.h5'")
