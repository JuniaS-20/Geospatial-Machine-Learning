"""Feature experiments, baseline comparison, spatial tuning and model selection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import balanced_accuracy_score, f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import ModelingConfig
from .features import FeatureDefinition, FeatureSubsetTransformer


@dataclass
class ModelSelectionResult:
    estimator: Any
    model_name: str
    feature_experiment: str
    feature_results: dict
    model_results: dict
    best_params: dict
    selected_feature_names: list[str]


def _scoring() -> dict:
    return {
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
        "macro_f1": make_scorer(f1_score, average="macro", zero_division=0),
        "accuracy": "accuracy",
    }


def _feature_step(experiment: str, feature_def: FeatureDefinition, cfg: ModelingConfig):
    if experiment == "raw_bands":
        return FeatureSubsetTransformer(feature_def.raw_indices)
    if experiment == "bands_plus_indices":
        return "passthrough"
    if experiment == "embedded_selection":
        selector_model = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
        )
        return SelectFromModel(selector_model, threshold=cfg.feature_selector_threshold)
    raise ValueError(experiment)


def _rf_pipeline(feature_step, cfg: ModelingConfig) -> Pipeline:
    return Pipeline([
        ("features", feature_step),
        ("classifier", RandomForestClassifier(random_state=cfg.random_state, n_jobs=cfg.n_jobs)),
    ])


def _svm_pipeline(feature_step, cfg: ModelingConfig) -> Pipeline:
    return Pipeline([
        ("features", feature_step),
        ("scaler", StandardScaler()),
        ("classifier", SVC(class_weight="balanced", probability=True, random_state=cfg.random_state)),
    ])


def evaluate_feature_experiments(
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    feature_def: FeatureDefinition,
    cfg: ModelingConfig,
) -> tuple[str, dict]:
    """Compare A/B/C using only development CV.

    Experiment C performs label-dependent feature selection *inside each fold* via
    a sklearn Pipeline, avoiding the leakage that would occur if feature selection
    were fitted once before cross-validation.
    """
    experiments = ["raw_bands", "bands_plus_indices"]
    if cfg.run_feature_experiments:
        experiments.append("embedded_selection")

    results: dict[str, dict] = {}
    for name in experiments:
        pipe = _rf_pipeline(_feature_step(name, feature_def, cfg), cfg)
        scores = cross_validate(pipe, X, y, cv=cv_splits, scoring=_scoring(), n_jobs=cfg.n_jobs)
        results[name] = {
            metric: {
                "mean": float(np.mean(scores[f"test_{metric}"])),
                "std": float(np.std(scores[f"test_{metric}"], ddof=1)) if len(cv_splits) > 1 else 0.0,
                "folds": [float(v) for v in scores[f"test_{metric}"]],
            }
            for metric in ("balanced_accuracy", "macro_f1", "accuracy")
        }
    best = max(experiments, key=lambda n: results[n][cfg.primary_metric]["mean"])
    return best, results


def _dummy_result(X, y, cv_splits, feature_step, cfg: ModelingConfig) -> dict:
    pipe = Pipeline([("features", feature_step), ("classifier", DummyClassifier(strategy="prior"))])
    scores = cross_validate(pipe, X, y, cv=cv_splits, scoring=_scoring(), n_jobs=cfg.n_jobs)
    return {
        metric: {
            "mean": float(np.mean(scores[f"test_{metric}"])),
            "std": float(np.std(scores[f"test_{metric}"], ddof=1)) if len(cv_splits) > 1 else 0.0,
        }
        for metric in ("balanced_accuracy", "macro_f1", "accuracy")
    }


def tune_and_select_model(
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    feature_def: FeatureDefinition,
    cfg: ModelingConfig,
) -> ModelSelectionResult:
    best_feature_exp, feature_results = evaluate_feature_experiments(X, y, cv_splits, feature_def, cfg)
    feature_step = _feature_step(best_feature_exp, feature_def, cfg)
    scoring = _scoring()

    model_results: dict[str, dict] = {}
    if cfg.include_dummy:
        model_results["dummy"] = _dummy_result(X, y, cv_splits, clone(feature_step) if feature_step != "passthrough" else "passthrough", cfg)

    searches: dict[str, RandomizedSearchCV] = {}
    rf = _rf_pipeline(clone(feature_step) if feature_step != "passthrough" else "passthrough", cfg)
    rf_search = RandomizedSearchCV(
        rf,
        param_distributions=cfg.rf_search_space,
        n_iter=cfg.randomized_search_iterations,
        scoring=scoring,
        refit=cfg.primary_metric,
        cv=cv_splits,
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
        return_train_score=False,
    )
    rf_search.fit(X, y)
    searches["random_forest"] = rf_search

    if cfg.include_svm:
        svm = _svm_pipeline(clone(feature_step) if feature_step != "passthrough" else "passthrough", cfg)
        svm_search = RandomizedSearchCV(
            svm,
            param_distributions=cfg.svm_search_space,
            n_iter=min(cfg.randomized_search_iterations, 15),
            scoring=scoring,
            refit=cfg.primary_metric,
            cv=cv_splits,
            n_jobs=cfg.n_jobs,
            random_state=cfg.random_state,
            return_train_score=False,
        )
        svm_search.fit(X, y)
        searches["svm"] = svm_search

    for name, search in searches.items():
        idx = search.best_index_
        model_results[name] = {
            "best_cv_balanced_accuracy": float(search.cv_results_["mean_test_balanced_accuracy"][idx]),
            "best_cv_macro_f1": float(search.cv_results_["mean_test_macro_f1"][idx]),
            "best_cv_accuracy": float(search.cv_results_["mean_test_accuracy"][idx]),
            "best_params": search.best_params_,
            "selection_metric": cfg.primary_metric,
        }

    selected_name = max(
        searches,
        key=lambda n: model_results[n][f"best_cv_{cfg.primary_metric}"],
    )
    selected_search = searches[selected_name]
    estimator = selected_search.best_estimator_

    # Best estimator is currently fit on all development data by RandomizedSearchCV.
    feature_transformer = estimator.named_steps["features"]
    if feature_transformer == "passthrough":
        selected_names = list(feature_def.all_names)
    elif isinstance(feature_transformer, FeatureSubsetTransformer):
        selected_names = [feature_def.all_names[i] for i in feature_transformer.indices]
    elif hasattr(feature_transformer, "get_support"):
        support = feature_transformer.get_support()
        selected_names = [n for n, keep in zip(feature_def.all_names, support) if keep]
    else:
        selected_names = list(feature_def.all_names)

    return ModelSelectionResult(
        estimator=estimator,
        model_name=selected_name,
        feature_experiment=best_feature_exp,
        feature_results=feature_results,
        model_results=model_results,
        best_params=selected_search.best_params_,
        selected_feature_names=selected_names,
    )
