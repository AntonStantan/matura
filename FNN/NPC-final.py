import sys
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, PReLU, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input

#used Gemini to shorten this file:

# Suppress TensorFlow informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.path.dirname(os.path.abspath("FNN5reciever.py"))

parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from GetXY import tokenizer

def create_and_load_model(weights_path):
    # This input shape must match the output shape of your tokenizer
    input_shape = (15,)

    # These hyperparameters MUST EXACTLY MATCH the model that was saved.
    # The error message indicated that the saved model used 345 neurons.
    best_hps = {
        "num_layers": 1,
        "num_neurons": 345,
        "dropout": False,
        "dropout_rate": 0.1
    }

    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Flatten())

    for _ in range(best_hps["num_layers"]):
        model.add(Dense(best_hps["num_neurons"]))
        model.add(PReLU())
        if best_hps["dropout"]:
            model.add(Dropout(best_hps["dropout_rate"]))

    model.add(Dense(1, activation='linear'))

    # The optimizer and loss are required for compilation but not used for inference.
    model.compile(optimizer="adam", loss="mse")

    # Load the pre-trained weights into the newly defined model structure.
    model.load_weights(weights_path)

    return model

if __name__ == "__main__":
    WEIGHTS_FILE = "FNN2_weights.weights.h5"

    print("Creating model architecture and loading weights...")
    best_model = create_and_load_model(WEIGHTS_FILE)
    print("Model loaded successfully.")
    best_model.summary()

    print("\n---------------------------------------------------")
    print("HELLO AND WELCOME TO THE NEURAL PREDICTIVE CALCULATOR - NPC")
    print("---------------------------------------------------")

    while True:
        try:
            user_expr = input("What do you want me to calculate? (or type 'exit' to quit): ")
            if user_expr.lower() == 'exit':
                break

            # Use the tokenizer to convert the user's string into the model's expected numerical input.
            tokenized_input = tokenizer([user_expr])

            # Use the loaded model to predict the result from the tokenized input.
            model_pred = best_model.predict(tokenized_input, verbose=0) # verbose=0 hides the progress bar

            print(f"What you asked: {user_expr}")
            print(f"This should be the answer: {model_pred[0][0]:.4f}")
            print("---------------------------------------------------")

        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please ensure your input is a valid mathematical expression.")
            print("---------------------------------------------------")