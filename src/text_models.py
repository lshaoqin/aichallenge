from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Flatten, Conv1D, GlobalMaxPooling1D, TextVectorization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.layers import Dropout
import dataprep

"""
Neural networks have way more parameters, so there are limited customisation
options. Users can modify the code directly to change the architecture of the
network.
"""

max_features = 10000
sequence_length = 1000

def train_LSTM_NN(x_train, y_train, x_val, y_val, learning_rate = 0.0002, epochs = 10):

    model = Sequential([
        Input(shape=(sequence_length,)),
        Embedding(max_features, 64, input_length=sequence_length),
        LSTM(64, return_sequences=True, dropout=0.5, recurrent_dropout=0.5),
        Flatten(),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        Dense(5, activation='softmax')
    ])
    
    callback = EarlyStopping(monitor='val_loss', patience=3)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val), callbacks=[callback])
    return model
    
def train_CNN(x_train, y_train, x_val, y_val, learning_rate = 0.0002, epochs = 10):

    model = Sequential([
        Input(shape=(sequence_length,)),
        Embedding(max_features, 64, input_length=sequence_length),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dense(5, activation='softmax')
    ])

    callback = EarlyStopping(monitor='val_loss', patience=3)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val), callbacks=[callback])
    return model


def pipeline(df, model = "CNN", correct_spellings = True, save = False, filename = 'text_model.pkl'):
    """
    Preprocess the data and train the model
    """
    x_text, x_cat, y = dataprep.split_text_categorical(df)
    x_text = dataprep.preprocess_text(x_text)

    vectorize_layer = TextVectorization(
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length
    )

    vectorize_layer.adapt(X)

    x_train, x_test, x_val, y_train, y_test, y_val = dataprep.split_train_test_val(X, y)

    # Apply the vectorization layer to the text data
    X_train = vectorize_layer(x_train)
    X_test = vectorize_layer(x_test)
    X_val = vectorize_layer(x_val)

    if correct_spellings:
        x_text = dataprep.correct_spellings(x_text)
    if model == "LSTM":
        model = train_LSTM_NN(X_train, y_train, X_val, y_val)
    elif model == "CNN":
        model = train_CNN(X_train, y_train, X_val, y_val)

    # Evaluate the model
    accuracy = model.evaluate(X_test, y_test)
    print(f"{model} accuracy: {accuracy}")

    if save:
        model.save(filename)
        print(f"Model saved to {filename}")

    return model