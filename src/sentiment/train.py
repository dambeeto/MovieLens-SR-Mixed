"""Train + evaluate sentiment classifiers on VADER-labelled tags."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import RANDOM_STATE
from src.sentiment.features import build_vectorizer


def _model_zoo() -> dict[str, object]:
    """Three short-text-classification baselines.

    LinearSVC wrapped in CalibratedClassifierCV so it exposes predict_proba uniformly.
    """
    return {
        "logreg": LogisticRegression(
            # lbfgs handles multiclass natively (3 classes here); liblinear in
            # newer sklearn refuses multinomial without an OvR wrapper.
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "nb": MultinomialNB(),
        "linsvc": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", max_iter=4000, random_state=RANDOM_STATE),
            cv=3,
            method="sigmoid",
        ),
    }


def _make_pipeline(clf) -> Pipeline:
    return Pipeline([("vec", build_vectorizer()), ("clf", clf)])


@dataclass
class TrainingResult:
    pipeline: Pipeline
    best_model_name: str
    cv_results: pd.DataFrame
    model_comparison: pd.DataFrame
    test_classification_report: str
    confusion: np.ndarray
    classes: np.ndarray
    y_test: np.ndarray
    y_pred: np.ndarray


def _cross_validate(X: pd.Series, y: np.ndarray, n_splits: int = 5) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, clf in _model_zoo().items():
        print(f"[cv ] model={name}")
        for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
            pipe = _make_pipeline(_model_zoo()[name])
            pipe.fit(X.iloc[tr], y[tr])
            pred = pipe.predict(X.iloc[va])
            acc = accuracy_score(y[va], pred)
            mf1 = f1_score(y[va], pred, average="macro", zero_division=0)
            rows.append({"model": name, "fold": fold, "accuracy": acc, "macro_f1": mf1})
            print(f"      fold {fold}: acc={acc:.4f}  macro_f1={mf1:.4f}")
    return pd.DataFrame(rows)


def train_and_evaluate(
    df_labeled: pd.DataFrame,
    text_col: str = "tag",
    label_col: str = "sentiment",
    test_size: float = 0.2,
) -> TrainingResult:
    """Stratified split -> 5-fold CV across model zoo -> refit best on train -> evaluate on test."""
    df = df_labeled.dropna(subset=[text_col, label_col]).copy()
    df[text_col] = df[text_col].astype("string").fillna("")
    df[label_col] = df[label_col].astype(str)

    X = df[text_col].reset_index(drop=True)
    y = df[label_col].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )

    print(f"[train] n_train={len(X_train)} n_test={len(X_test)} classes={sorted(set(y))}")
    cv_results = _cross_validate(X_train, y_train)
    summary = (
        cv_results.groupby("model")["macro_f1"]
        .agg(["mean", "std"])
        .sort_values("mean", ascending=False)
        .reset_index()
    )
    summary["macro_f1_mean_std"] = summary.apply(
        lambda r: f"{r['mean']:.4f} +/- {r['std']:.4f}", axis=1
    )
    print("[train] CV summary:")
    print(summary.to_string(index=False))

    best_model_name = summary.iloc[0]["model"]
    print(f"[train] selected: {best_model_name}")

    pipe = _make_pipeline(_model_zoo()[best_model_name])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    report = classification_report(y_test, y_pred, zero_division=0)
    classes = pipe.classes_
    conf = confusion_matrix(y_test, y_pred, labels=classes)
    print("[train] test classification report:")
    print(report)

    return TrainingResult(
        pipeline=pipe,
        best_model_name=str(best_model_name),
        cv_results=cv_results,
        model_comparison=summary,
        test_classification_report=report,
        confusion=conf,
        classes=classes,
        y_test=y_test,
        y_pred=y_pred,
    )


def top_words_per_class(pipeline: Pipeline, n: int = 15) -> pd.DataFrame:
    """Return top-n most discriminative word-features per class.

    Works only when the underlying estimator exposes ``coef_`` and the vectorizer
    union contains the 'word' branch. NB and LR satisfy this; calibrated SVC does not.
    """
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return pd.DataFrame(columns=["class", "rank", "feature", "weight"])

    vec = pipeline.named_steps["vec"]
    word_vec = dict(vec.transformer_list).get("word")
    if word_vec is None:
        return pd.DataFrame(columns=["class", "rank", "feature", "weight"])
    word_names = word_vec.get_feature_names_out()

    # coef_ alignment: word features come first in FeatureUnion order.
    coef = clf.coef_[:, : len(word_names)]
    rows = []
    for cls_idx, cls in enumerate(clf.classes_):
        order = np.argsort(coef[cls_idx])[::-1][:n]
        for rank, idx in enumerate(order, start=1):
            rows.append(
                {
                    "class": cls,
                    "rank": rank,
                    "feature": word_names[idx],
                    "weight": float(coef[cls_idx, idx]),
                }
            )
    return pd.DataFrame(rows)
