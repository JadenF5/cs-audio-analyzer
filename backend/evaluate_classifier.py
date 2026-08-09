"""
evaluate_classifier.py — Run this locally after unzipping UrbanSound8k
and LibriSpeech dev-clean
========================================================================
Retrains the classifier on real data (UrbanSound8k for tone/noise/music,
LibriSpeech for speech), runs 5-fold cross-validation, and prints a
confusion matrix so you can see exactly where the model confuses classes.

Usage:
    # from the backend/ folder, with UrbanSound8K/ and LibriSpeech/
    # unzipped alongside it:
    python evaluate_classifier.py

    # or if your folders live elsewhere:
    URBANSOUND8K_ROOT=/path/to/UrbanSound8K LIBRISPEECH_ROOT=/path/to/LibriSpeech/dev-clean python evaluate_classifier.py
"""

import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report

from classifier import generate_dataset, CLASSES, MODEL_CACHE_PATH
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import os


def main():
    print("Building dataset (real UrbanSound8k for tone/noise/music, "
          "real LibriSpeech for speech)...")
    X, y = generate_dataset()
    print(f"Total samples: {len(y)}")
    for cls in CLASSES:
        print(f"  {cls:8s}: {np.sum(y == cls)}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1,
            class_weight="balanced",
        )),
    ])

    print("\nRunning 5-fold stratified cross-validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=skf)
    print(f"CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    print(f"Fold scores: {[round(s, 3) for s in scores]}")

    # Confusion matrix from HELD-OUT predictions (cross_val_predict), not
    # fit-then-predict-on-training-data — a RandomForest can trivially
    # memorize 600 training rows, so a same-data matrix is meaningless.
    print("\nGenerating held-out predictions for confusion matrix...")
    from sklearn.model_selection import cross_val_predict
    preds = cross_val_predict(pipeline, X, y, cv=skf)

    labels = sorted(set(y))
    cm = confusion_matrix(y, preds, labels=labels)
    print("\nConfusion matrix (rows=true, cols=predicted):")
    print("        " + "  ".join(f"{l[:6]:>6s}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{l[:6]:>6s}  " + "  ".join(f"{cm[i][j]:6d}" for j in range(len(labels))))

    print("\nClassification report (held-out predictions):")
    print(classification_report(y, preds))

    # Now fit the final model on ALL data (real prod behavior — no reason
    # to hold data back once you're done evaluating) and cache it, so a
    # server restart doesn't need to retrain.
    print("\nFitting final model on full data and caching...")
    pipeline.fit(X, y)
    import joblib
    joblib.dump(pipeline, MODEL_CACHE_PATH)
    print(f"Saved to {MODEL_CACHE_PATH}")

if __name__ == "__main__":
    main()