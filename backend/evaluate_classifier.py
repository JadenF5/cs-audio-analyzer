"""
evaluate_classifier.py — Evaluate and rebuild the audio classifier
==================================================================
Run this after pulling the updated classifier.py to:
  1. Retrain on real UrbanSound8k data + synthetic noise augmentation
  2. Run 5-fold cross-validation and print confusion matrix
  3. Save the improved model to audio_classifier_model.joblib

Usage (from backend/ folder):
    python evaluate_classifier.py

    # or with custom dataset path:
    URBANSOUND8K_ROOT=/path/to/UrbanSound8K python evaluate_classifier.py
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib

from classifier import (
    generate_dataset, CLASSES, MODEL_CACHE_PATH,
    extract_features, N_SYNTHETIC_NOISE_AUGMENT,
)


def main():
    print("=" * 60)
    print("AUDIO CLASSIFIER EVALUATION — v2 (36 features)")
    print("=" * 60)
    print("\nKey improvements over v1:")
    print("  + spectral_flatness  (key: noise=~0.5, tone=~0.004)")
    print("  + spectral_bandwidth (key: noise>music>tone)")
    print("  + chroma_variance    (key: music varies, tone fixed)")
    print(f"  + {N_SYNTHETIC_NOISE_AUGMENT} synthetic Gaussian noise samples injected")
    print("    into noise class to cover flat-spectrum inference cases")
    print("  + HistGradientBoostingClassifier (replaces RandomForest)")
    print()

    print("Building dataset...")
    X, y = generate_dataset()
    print(f"Total samples: {len(y)}")
    for cls in sorted(set(y)):
        print(f"  {cls:8s}: {np.sum(y == cls)}")
    print(f"Feature vector: {X.shape[1]} dimensions")
    print()

    # Verify new features are actually different across classes
    print("Spectral flatness by class (should be: noise >> music > tone):")
    # flatness is feature index 29 (13+13+1+1+1 = 29th feature, 0-indexed)
    FLATNESS_IDX = 29
    for cls in sorted(set(y)):
        vals = X[y == cls, FLATNESS_IDX]
        print(f"  {cls:8s}: mean={vals.mean():.4f}  std={vals.std():.4f}  "
              f"min={vals.min():.4f}  max={vals.max():.4f}")
    print()

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(
            max_iter=300,
            max_depth=6,
            learning_rate=0.08,
            min_samples_leaf=8,
            l2_regularization=0.1,
            random_state=42,
            class_weight="balanced",
        )),
    ])

    print("Running 5-fold stratified cross-validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=skf, scoring="accuracy")
    print(f"CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    print(f"Fold scores: {[round(s, 3) for s in scores]}")
    print()

    print("Generating held-out confusion matrix...")
    preds = cross_val_predict(pipeline, X, y, cv=skf)
    labels = sorted(set(y))
    cm = confusion_matrix(y, preds, labels=labels)

    print("Confusion matrix (rows=true, cols=predicted):")
    print("         " + "  ".join(f"{l[:6]:>6s}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{l[:6]:>6s}   " + "  ".join(f"{cm[i][j]:6d}" for j in range(len(labels))))
    print()
    print(classification_report(y, preds))

    # Spot check new discriminative features
    print("Spot check — pure Gaussian noise (should be 'noise'):")
    import numpy as _np
    rng = _np.random.default_rng(999)
    noise_sig = rng.standard_normal(256).astype(_np.float32)
    noise_feat = extract_features(noise_sig).reshape(1, -1)
    pipeline.fit(X, y)
    pred = pipeline.predict(noise_feat)[0]
    prob = max(pipeline.predict_proba(noise_feat)[0])
    flat = noise_feat[0, FLATNESS_IDX]
    print(f"  predicted={pred}  confidence={prob:.3f}  flatness={flat:.4f}")
    print(f"  {'PASS' if pred == 'noise' else 'FAIL — check feature extraction'}")
    print()

    print("Fitting final model on full data and saving...")
    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_CACHE_PATH)
    print(f"Saved to {MODEL_CACHE_PATH}")
    print()
    print("Next steps:")
    print("  1. git add audio_classifier_model.joblib classifier.py")
    print("  2. git commit -m 'chore: rebuild classifier v2 (spectral flatness fix)'")
    print("  3. git push  →  Render redeploys automatically")


if __name__ == "__main__":
    main()