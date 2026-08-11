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

# Safe imports — no constants that might not exist in older classifier versions
from classifier import generate_dataset, CLASSES, MODEL_CACHE_PATH, extract_features

# Read augment count safely — defaults to 60 if old classifier.py is used
try:
    from classifier import N_SYNTHETIC_NOISE_AUGMENT
except ImportError:
    N_SYNTHETIC_NOISE_AUGMENT = 60
    print("WARNING: N_SYNTHETIC_NOISE_AUGMENT not found in classifier.py")
    print("         You may be using an old classifier.py — please replace it.")
    print()


def main():
    print("=" * 60)
    print("AUDIO CLASSIFIER EVALUATION — v2 (36 features)")
    print("=" * 60)
    print("\nKey improvements over v1:")
    print("  + spectral_flatness  (key: noise~0.5, tone~0.004, music~0.07)")
    print("  + spectral_bandwidth (noise > music > tone)")
    print("  + chroma_variance    (music varies, tone fixed)")
    print(f"  + {N_SYNTHETIC_NOISE_AUGMENT} synthetic Gaussian noise samples")
    print("    injected into noise class (fixes flat-spectrum misclassification)")
    print("  + HistGradientBoostingClassifier (replaces RandomForest)")
    print()

    print("Building dataset...")
    X, y = generate_dataset()
    y = np.array(y)
    print(f"Total samples: {len(y)}")
    for cls in sorted(set(y)):
        print(f"  {cls:8s}: {np.sum(y == cls)}")
    print(f"Feature vector: {X.shape[1]} dimensions")
    print()

    # ── Verify spectral flatness separates the classes ──────
    # In the 36-feature vector layout:
    #   0-12:  MFCC means
    #   13-25: MFCC stds
    #   26:    spectral centroid
    #   27:    spectral bandwidth
    #   28:    spectral rolloff
    #   29:    spectral flatness  ← key
    #   30:    ZCR
    #   31:    RMS
    #   32:    chroma variance
    if X.shape[1] >= 30:
        FLATNESS_IDX = 29
        print("Spectral flatness by class (noise >> music > tone = good separation):")
        for cls in sorted(set(y)):
            vals = X[y == cls, FLATNESS_IDX]
            print(f"  {cls:8s}: mean={vals.mean():.4f}  "
                  f"std={vals.std():.4f}  "
                  f"min={vals.min():.4f}  max={vals.max():.4f}")
        print()
    else:
        print(f"WARNING: only {X.shape[1]} features found (expected 36).")
        print("         spectral_flatness may be missing — check classifier.py")
        FLATNESS_IDX = None
        print()

    # ── Cross-validation ────────────────────────────────────
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
    print(f"Fold scores: {[round(float(s), 3) for s in scores]}")
    print()

    # ── Confusion matrix ────────────────────────────────────
    print("Generating held-out confusion matrix...")
    preds = cross_val_predict(pipeline, X, y, cv=skf)
    labels = sorted(set(y))
    cm = confusion_matrix(y, preds, labels=labels)

    print("Confusion matrix (rows=true, cols=predicted):")
    print("         " + "  ".join(f"{l[:6]:>6s}" for l in labels))
    for i, l in enumerate(labels):
        row = "  ".join(f"{cm[i][j]:6d}" for j in range(len(labels)))
        print(f"{l[:6]:>6s}   {row}")
    print()
    print(classification_report(y, preds))

    # ── Spot check: pure Gaussian noise ────────────────────
    print("Spot check — pure Gaussian noise (should always → 'noise'):")
    pipeline.fit(X, y)
    results = []
    for seed in [42, 99, 123, 777, 2024]:
        rng = np.random.default_rng(seed)
        sig = rng.standard_normal(256).astype(np.float32)
        feat = extract_features(sig).reshape(1, -1)
        pred = pipeline.predict(feat)[0]
        prob = float(max(pipeline.predict_proba(feat)[0]))
        flat = float(feat[0, FLATNESS_IDX]) if FLATNESS_IDX is not None else -1
        ok = "PASS" if pred == "noise" else "FAIL"
        results.append(ok)
        print(f"  seed={seed}: predicted={pred:6s}  conf={prob:.3f}  "
              f"flatness={flat:.4f}  {ok}")
    passed = results.count("PASS")
    print(f"\n  {passed}/5 noise tests passed "
          f"({'all good!' if passed == 5 else 'needs tuning'})")
    print()

    # ── Spot check: pure tone ──────────────────────────────
    print("Spot check — pure sine tone (should always → 'tone'):")
    sr = 8000; N = 256
    tone_tests = [220, 440, 880, 1000, 1500]
    t_results = []
    for freq in tone_tests:
        t = np.linspace(0, N/sr, N, endpoint=False)
        sig = np.sin(2*np.pi*freq*t).astype(np.float32)
        feat = extract_features(sig).reshape(1, -1)
        pred = pipeline.predict(feat)[0]
        prob = float(max(pipeline.predict_proba(feat)[0]))
        flat = float(feat[0, FLATNESS_IDX]) if FLATNESS_IDX is not None else -1
        ok = "PASS" if pred == "tone" else "FAIL"
        t_results.append(ok)
        print(f"  {freq}Hz: predicted={pred:6s}  conf={prob:.3f}  "
              f"flatness={flat:.4f}  {ok}")
    t_passed = t_results.count("PASS")
    print(f"\n  {t_passed}/5 tone tests passed "
          f"({'all good!' if t_passed == 5 else 'needs tuning'})")
    print()

    # ── Save final model ────────────────────────────────────
    print("Saving final model...")
    joblib.dump(pipeline, MODEL_CACHE_PATH)
    print(f"Saved to {MODEL_CACHE_PATH}")
    print()
    print("Next steps:")
    print("  1. git add backend/classifier.py backend/audio_classifier_model.joblib")
    print("  2. git commit -m 'fix: classifier v2 - spectral flatness fixes tone/noise'")
    print("  3. git push  ->  Render redeploys automatically")
    print()
    total_pass = passed + t_passed
    if total_pass == 10:
        print("All spot checks passed — classifier is working correctly!")
    else:
        print(f"WARNING: {10 - total_pass} spot checks failed.")
        print("Check that classifier.py has spectral_flatness in extract_features().")


if __name__ == "__main__":
    main()