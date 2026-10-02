from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


NON_FEATURE_COLUMNS = {"start_index", "end_index", "activity", "label_purity"}


@dataclass
class ModelResult:
    name: str
    accuracy: float
    macro_f1: float
    classification_report: dict
    confusion_matrix: list[list[int]]
    labels: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def feature_matrix(windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_columns = [column for column in windows.columns if column not in NON_FEATURE_COLUMNS]
    return windows[feature_columns], windows["activity"].astype(str)


def _evaluate(name: str, model, x_train, x_test, y_train, y_test) -> ModelResult:
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    labels = sorted(y_test.unique().tolist())
    return ModelResult(
        name=name,
        accuracy=float(accuracy_score(y_test, predictions)),
        macro_f1=float(f1_score(y_test, predictions, average="macro")),
        classification_report=classification_report(y_test, predictions, output_dict=True, zero_division=0),
        confusion_matrix=confusion_matrix(y_test, predictions, labels=labels).tolist(),
        labels=labels,
    )


def train_baselines(windows: pd.DataFrame, random_state: int = 42) -> dict:
    x, y = feature_matrix(windows)
    if y.nunique() < 2:
        raise ValueError("At least two activity classes are required for classification")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=y,
    )

    logistic = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    forest = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )

    results = [
        _evaluate("Logistic Regression", logistic, x_train, x_test, y_train, y_test),
        _evaluate("Random Forest", forest, x_train, x_test, y_train, y_test),
    ]
    results.sort(key=lambda item: item.macro_f1, reverse=True)

    return {
        "train_windows": int(len(x_train)),
        "test_windows": int(len(x_test)),
        "feature_count": int(x.shape[1]),
        "class_counts": y.value_counts().sort_index().to_dict(),
        "models": [result.to_dict() for result in results],
        "best_model": results[0].name,
    }
