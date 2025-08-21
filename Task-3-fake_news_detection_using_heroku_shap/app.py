import os
import io
import pickle
import numpy as np
from flask import Flask, request, render_template
from preprocess import clean_text

# Optional: try to import shap; app works even if shap is unavailable
try:

    import shap

    HAVE_SHAP = True

except Exception:

    HAVE_SHAP = False


MODELS_DIR = os.environ.get("MODELS_DIR", "models")
with open(os.path.join(MODELS_DIR, "vectorizer.pkl"), "rb") as f:

    vectorizer = pickle.load(f)

with open(os.path.join(MODELS_DIR, "model.pkl"), "rb") as f:

    model = pickle.load(f)


app = Flask(__name__)

TARGET_NAMES = {0: "Real", 1: "Fake"}

def explain_linear(text, top_k=10):

    """Return top positive/negative token contributions using linear model coefficients.

    Works without SHAP and is theoretically equivalent to SHAP for linear models under common settings.

    """

    x_clean = clean_text(text)

    Xv = vectorizer.transform([x_clean])  # sparse row

    coef = model.coef_[0]  # shape [n_features]

    # elementwise multiply sparse row by coef -> contributions

    contrib = Xv.multiply(coef)

    # Convert to dense index/value pairs of nonzeros

    indices = contrib.nonzero()[1]

    values = np.array(contrib.data)

    # Map back to terms

    feats = vectorizer.get_feature_names_out()

    terms = [(feats[i], float(v)) for i, v in zip(indices, values)]

    # Sort by contribution

    terms_sorted = sorted(terms, key=lambda t: t[1], reverse=True)

    pos = terms_sorted[:top_k]

    neg = list(reversed(terms_sorted[-top_k:])) if len(terms_sorted) > top_k else []

    return pos, neg, float(model.decision_function(Xv)[0])


def predict(text):

    x_clean = clean_text(text)

    Xv = vectorizer.transform([x_clean])

    proba = model.predict_proba(Xv)[0, 1]

    pred = int(proba >= 0.5)

    return pred, float(proba)


@app.route("/", methods=["GET", "POST"]) 

def home():

    explanation = None

    pos_terms = []

    neg_terms = []

    decision = None

    pred_label = None

    proba = None

    user_text = ""



    if request.method == "POST":

        user_text = request.form.get("news_text", "")

        if user_text.strip():

            pred, proba = predict(user_text)

            pred_label = TARGET_NAMES[pred]

            pos_terms, neg_terms, decision = explain_linear(user_text, top_k=10)



    return render_template(

        "index.html",

        prediction=pred_label,

        probability=proba,

        pos_terms=pos_terms,

        neg_terms=neg_terms,

        decision=decision,

        user_text=user_text,

        have_shap=HAVE_SHAP,

    )



if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)

