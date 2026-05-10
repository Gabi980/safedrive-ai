import argparse
from pathlib import Path
import time

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


BASE_FEATURE_COLUMNS = ["ear", "mar", "pitch", "yaw", "roll"]
RULE_FEATURE_COLUMNS = ["score", "alert_level"]
TARGET_COLUMN = "label"
LABEL_NAMES = ["alert", "drowsy"]


def load_dataset(csv_path, include_rule_features=False):
    df = pd.read_csv(csv_path)
    feature_columns = BASE_FEATURE_COLUMNS.copy()

    if include_rule_features:
        feature_columns.extend(RULE_FEATURE_COLUMNS)

    required_columns = feature_columns + [TARGET_COLUMN, "face_detected"]
    missing_columns = [
        column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing columns in {csv_path}: {missing_columns}")

    before_rows = len(df)
    df = df[df["face_detected"] == True].copy()  # noqa: E712
    df = df.dropna(subset=feature_columns + [TARGET_COLUMN])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    print(f"Loaded: {csv_path}")
    print(f"Rows before cleaning: {before_rows}")
    print(f"Rows after cleaning:  {len(df)}")
    print("Class distribution:")
    print(df[TARGET_COLUMN].value_counts().sort_index().to_string())

    return df, feature_columns


def split_dataset(df, feature_columns, test_size, random_state):
    x = df[feature_columns]
    y = df[TARGET_COLUMN]

    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_models(random_state):
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=18,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features="sqrt",
            bootstrap=True,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "svm_rbf": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        C=3.0,
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=random_state,
        ),
    }


def select_models(models, selected_model_names):
    if not selected_model_names:
        return models

    unknown_models = [
        name for name in selected_model_names if name not in models]

    if unknown_models:
        raise ValueError(
            f"Unknown model names: {unknown_models}. Available: {list(models)}")

    return {name: models[name] for name in selected_model_names}


def get_positive_probabilities(model, x_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, 1]

    return None


def evaluate_model(name, model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_prob = get_positive_probabilities(model, x_test)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else None,
    }

    report = classification_report(
        y_test,
        y_pred,
        target_names=LABEL_NAMES,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, y_pred)

    return metrics, report, matrix, y_prob


def save_confusion_matrix(matrix, model_name, output_dir):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_dir / f"confusion_matrix_{model_name}.png", dpi=160)
    plt.close()


def save_roc_curve(y_test, y_prob, model_name, output_dir):
    if y_prob is None:
        return

    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(false_positive_rate, true_positive_rate, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(f"ROC Curve - {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / f"roc_curve_{model_name}.png", dpi=160)
    plt.close()


def save_model_comparison(metrics_df, output_dir):
    metrics_df.to_csv(output_dir / "model_metrics.csv", index=False)

    plot_df = metrics_df.melt(
        id_vars=["model"],
        value_vars=["accuracy", "precision", "recall", "f1", "roc_auc"],
        var_name="metric",
        value_name="value",
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x="model", y="value", hue="metric")
    plt.ylim(0, 1)
    plt.title("Model Comparison")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=160)
    plt.close()


def save_feature_importance(model, feature_columns, output_dir, model_name):
    estimator = model

    if isinstance(model, Pipeline):
        estimator = model.named_steps.get("model")

    if not hasattr(estimator, "feature_importances_"):
        return

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": estimator.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance_df.to_csv(
        output_dir / f"feature_importance_{model_name}.csv", index=False)

    plt.figure(figsize=(7, 5))
    sns.barplot(data=importance_df, x="importance", y="feature")
    plt.title(f"Feature Importance - {model_name}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_dir / f"feature_importance_{model_name}.png", dpi=160)
    plt.close()


def train_and_evaluate(df, feature_columns, args):
    x_train, x_test, y_train, y_test = split_dataset(
        df,
        feature_columns,
        args.test_size,
        args.random_state,
    )
    models = select_models(build_models(args.random_state), args.models)
    output_dir = Path(args.reports_dir)
    model_dir = Path(args.models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Train rows: {len(x_train)}", flush=True)
    print(f"Test rows:  {len(x_test)}", flush=True)
    print(f"Features:   {feature_columns}", flush=True)
    print(f"Models:     {list(models)}", flush=True)

    trained_models = {}
    all_metrics = []

    for name, model in models.items():
        model_x_train = x_train
        model_y_train = y_train

        if name == "svm_rbf" and args.svm_max_train_rows > 0:
            sample_size = min(args.svm_max_train_rows, len(x_train))
            sampled = x_train.copy()
            sampled[TARGET_COLUMN] = y_train
            rows_per_class = max(1, sample_size // 2)
            sampled = sampled.groupby(TARGET_COLUMN, group_keys=False).sample(
                n=rows_per_class,
                random_state=args.random_state,
                replace=False,
            )
            model_x_train = sampled[feature_columns]
            model_y_train = sampled[TARGET_COLUMN]
            print(
                f"\nTraining {name} on {len(model_x_train)} sampled rows "
                f"(use --svm-max-train-rows 0 for full SVM training)...",
                flush=True,
            )
        else:
            print(
                f"\nTraining {name} on {len(model_x_train)} rows...", flush=True)

        started_at = time.time()
        model.fit(model_x_train, model_y_train)
        elapsed = time.time() - started_at
        print(f"{name} trained in {elapsed:.1f}s", flush=True)
        print(f"Evaluating {name}...", flush=True)

        metrics, report, matrix, y_prob = evaluate_model(
            name, model, x_test, y_test)
        all_metrics.append(metrics)
        trained_models[name] = model

        print(f"\n{name} metrics:", flush=True)
        print(pd.Series(metrics).to_string(), flush=True)
        print("\nClassification report:", flush=True)
        print(report, flush=True)

        save_confusion_matrix(matrix, name, output_dir)
        save_roc_curve(y_test, y_prob, name, output_dir)
        save_feature_importance(model, feature_columns, output_dir, name)

    metrics_df = pd.DataFrame(all_metrics).sort_values("f1", ascending=False)
    save_model_comparison(metrics_df, output_dir)

    best_model_name = metrics_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]
    bundle = {
        "model": best_model,
        "model_name": best_model_name,
        "feature_columns": feature_columns,
        "label_names": LABEL_NAMES,
        "metrics": metrics_df.to_dict(orient="records"),
    }
    best_model_path = model_dir / "best_model.joblib"
    joblib.dump(bundle, best_model_path)

    print("\nModel comparison:")
    print(metrics_df.to_string(index=False))
    print(f"\nBest model: {best_model_name}")
    print(f"Saved best model to: {best_model_path}")
    print(f"Reports saved to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SafeDrive AI ML models.")
    parser.add_argument("--csv", default="data/features_combined.csv")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["random_forest", "svm_rbf", "xgboost"],
        default=None,
        help="Train only selected models. Default: all models.",
    )
    parser.add_argument(
        "--svm-max-train-rows",
        type=int,
        default=20000,
        help="Limit SVM training rows because RBF SVM is slow on large datasets. Use 0 for full SVM.",
    )
    parser.add_argument(
        "--include-rule-features",
        action="store_true",
        help="Include score and alert_level as extra features.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df, feature_columns = load_dataset(args.csv, args.include_rule_features)
    train_and_evaluate(df, feature_columns, args)


if __name__ == "__main__":
    main()
