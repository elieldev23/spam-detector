# spam-detector

This project implements a machine learning model to detect **fraudulent
and scam SMS messages** using natural language processing (NLP).

The system classifies incoming SMS messages into: - **Spam** (fraud /
scam) - **Not Spam** (legitimate messages)

The model is trained on the UCI SMS Spam Collection dataset and uses a
TF-IDF + Naive Bayes classification pipeline.

------------------------------------------------------------------------

## Project Structure

    spam-detector/
    │
    ├── data/
    │   └── sms_spam_collection.tsv
    │
    ├── model/
    │   └── spam_model.pkl
    │
    ├── notebooks/
    │   └── exploration.ipynb
    │
    ├── src/
    │   ├── train.py
    │   ├── predict.py
    │   └── utils.py
    │
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## Dataset

The project uses the **SMS Spam Collection Dataset** from UCI.

Each message is labeled as: - **ham** → Legitimate SMS\
- **spam** → Fraudulent or scam SMS

The dataset contains real-world examples of common scam messages such
as: - Fake prize notifications\
- Urgent requests\
- Suspicious links and phone numbers

------------------------------------------------------------------------

## Model

The classification pipeline includes:

-   **TF-IDF Vectorization** for text feature extraction\
-   **Multinomial Naive Bayes** for spam classification

This approach is lightweight, fast, and well-suited for text
classification tasks.

------------------------------------------------------------------------

## Training the Model

To train the model locally:

``` bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
```

The trained model will be saved to:

    model/spam_model.pkl

------------------------------------------------------------------------

## Making Predictions

You can test the model from the terminal:

``` bash
python src/predict.py 'WINNER!! You have won a £1000 prize. Call now.'
```

Output:

    Result: Spam

Another example:

``` bash
python src/predict.py 'On se voit à 18h ?'
```

Output:

    Result: Not Spam

------------------------------------------------------------------------

## Notebook

The `notebooks/exploration.ipynb` file contains a short data exploration
of:

-   Spam vs Ham distribution\
-   Typical spam message patterns\
-   Message length differences

This provides context for how the model learns to detect fraud.

------------------------------------------------------------------------

## Scope of the Model

This model is specifically trained to detect:

-   Fraudulent SMS\
-   Scam messages

It does **not** currently classify:

-   Legal marketing SMS\
-   Promotional campaigns\
-   Brand advertisements

These could be added in future versions by expanding the dataset.

------------------------------------------------------------------------

## Technologies Used

-   Python\
-   Pandas\
-   Scikit-learn\
-   Joblib\
-   Jupyter Notebook

------------------------------------------------------------------------

## Future Improvements

Possible extensions include:

-   Detecting marketing SMS\
-   Adding more datasets\
-   Trying other classifiers\
-   Building a web API (Flask)

------------------------------------------------------------------------

## Author

Eliel Beonao\
GitHub: elieldev23
