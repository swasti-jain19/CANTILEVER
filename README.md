# Fake News Detector with SHAP (Heroku)

A simple Flask app that serves a TF–IDF + Logistic Regression fake-news classifier and shows SHAP-style token attributions.

## Local quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python train.py                                     # expects data/Fake.csv and data/True.csv
python app.py
```

Then visit http://localhost:5000

## Data
Place `Fake.csv` and `True.csv` inside `data/`. At least one column containing text should be present (e.g., `text`, `content`, `title`, `body`).

## Heroku deploy

```bash
heroku login
heroku create fake-news-shap-<your-suffix>
git init
git add .
git commit -m "Initial commit: fake news + SHAP"
heroku buildpacks:set heroku/python
git push heroku main  # or 'master' depending on your git default branch
heroku ps:scale web=1
heroku open
```

If you need to retrain on Heroku, you can `heroku run python train.py` after uploading CSVs to the app or committing your trained `models/*.pkl` to the repo.

## Notes on SHAP
For linear models like Logistic Regression with TF–IDF features, per-feature contributions computed as (coef * feature_value) match SHAP values under common settings. If the `shap` library is available, you can extend the app to generate force plots; however, this app keeps it lightweight for Heroku by showing the top pushing terms toward each class.
