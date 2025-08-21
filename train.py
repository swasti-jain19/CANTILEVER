import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from preprocess import clean_text

DATA_DIR = os.environ.get("DATA_DIR", "data")
FAKE_PATH = os.path.join(DATA_DIR, "Fake.csv")
TRUE_PATH = os.path.join(DATA_DIR, "True.csv")
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data() -> pd.DataFrame:
    df_fake = pd.read_csv(FAKE_PATH)

    df_true = pd.read_csv(TRUE_PATH)

    # Normalize column names

    def _get_text(df):

        for c in ["text", "content", "article", "body", "title"]:

            if c in df.columns:

                return df[c].astype(str)

        raise KeyError("Couldn't find a text/content column in the CSVs.")

    fake = _get_text(df_fake)

    true = _get_text(df_true)

    df_fake = pd.DataFrame({"text": fake, "label": 1})

    df_true = pd.DataFrame({"text": true, "label": 0})

    df = pd.concat([df_fake, df_true], ignore_index=True)

    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)


def train_and_save():

    df = load_data()

    df["text_clean"] = df["text"].map(clean_text)

    X = df["text_clean"].values

    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=50000)

    Xv_train = vectorizer.fit_transform(X_train)



    model = LogisticRegression(max_iter=1000, n_jobs=None)

    model.fit(Xv_train, y_train)



    # quick eval

    Xv_test = vectorizer.transform(X_test)

    y_pred = model.predict(Xv_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))

    print(classification_report(y_test, y_pred))



    with open(os.path.join(MODELS_DIR, "vectorizer.pkl"), "wb") as f:

        pickle.dump(vectorizer, f)

    with open(os.path.join(MODELS_DIR, "model.pkl"), "wb") as f:

        pickle.dump(model, f)

    print("Saved models to", MODELS_DIR)



if __name__ == "__main__":

    train_and_save()

