from pathlib import Path
import joblib
import sys


# Path to the trained model
MODEL_PATH = Path("model/spam_model.pkl")


def load_model(path: Path):
    """
    Loads the trained spam detection model from disk.
    """
    return joblib.load(path)


def main():
    # Check if a message was provided in the command line
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py \"Your message here\"")
        return

    # Get the message from the command line
    message = sys.argv[1]

    # Load the trained model
    model = load_model(MODEL_PATH)

    # Make a prediction
    prediction = model.predict([message])[0]

    # Display the result
    if prediction == "spam":
        print("Result: Spam")
    else:
        print("Result: Not Spam")


if __name__ == "__main__":
    main()
