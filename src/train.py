from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib



# Path to the dataset file
DATA_PATH = Path("data/sms_spam_collection.tsv")

# Path where the trained model will be saved
MODEL_PATH = Path("model/spam_model.pkl")



# Data loading function

def load_data(path: Path) -> pd.DataFrame:
    """
    Loads the SMS Spam dataset from a TSV file.

    The dataset contains:
    - Column 1: label (spam or ham)
    - Column 2: message text
    """
    
    # Read the TSV file using pandas
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"])
    
    return df



def main():
    #Load the dataset
    df = load_data(DATA_PATH)

    # Separate features (messages) and labels (spam/ham)
    X = df["text"]
    y = df["label"]

    # Split the dataset into training (80%) and testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Create a pipeline:
    #    - TF-IDF for text vectorization
    #    - Naive Bayes for classification
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True)),
        ("classifier", MultinomialNB())
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Display evaluation results
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Create the model directory if it does not exist
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save the trained model to disk
    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
