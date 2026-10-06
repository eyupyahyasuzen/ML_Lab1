import os
import time
import joblib
import numpy as np
import pandas as pd

from memory_profiler import memory_usage
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def main():

    # Load dataset
    X, y = load_breast_cancer(return_X_y=True)

    # 70% train, 30% test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    results = []

    for model_name, model in models.items():

        print(f"\nMeasuring: {model_name}")

        # -------------------------
        # Warm-up
        # -------------------------
        model.fit(X_train, y_train)

        # -------------------------
        # Training time
        # 5 runs, median
        # -------------------------
        training_times = []

        for _ in range(5):

            start = time.perf_counter()

            model.fit(
                X_train,
                y_train
            )

            end = time.perf_counter()

            training_times.append(
                (end - start) * 1000
            )

        median_training_time = np.median(
            training_times
        )

        # -------------------------
        # Inference time
        # 100 runs, median
        # -------------------------
        sample = X_test[0].reshape(1, -1)

        inference_times = []

        for _ in range(100):

            start = time.perf_counter()

            model.predict(sample)

            end = time.perf_counter()

            inference_times.append(
                (end - start) * 1000
            )

        median_inference_time = np.median(
            inference_times
        )

        # -------------------------
        # Peak training memory
        # -------------------------
        def train_model():

            model.fit(
                X_train,
                y_train
            )

        training_memory_values = memory_usage(
            (train_model,),
            interval=0.01,
            timeout=None
        )

        training_peak_memory = max(
            training_memory_values
        )

        # -------------------------
        # Peak inference memory
        # -------------------------
        def run_inference():

            model.predict(sample)

        inference_memory_values = memory_usage(
            (run_inference,),
            interval=0.01,
            timeout=None
        )

        inference_peak_memory = max(
            inference_memory_values
        )

        # -------------------------
        # Save model
        # -------------------------
        safe_name = (
            model_name
            .lower()
            .replace(" ", "_")
        )

        model_path = (
            f"lab01/results/"
            f"{safe_name}_model.joblib"
        )

        joblib.dump(
            model,
            model_path
        )

        # -------------------------
        # Model size
        # -------------------------
        model_size_bytes = os.path.getsize(
            model_path
        )

        model_size_kb = (
            model_size_bytes / 1024
        )

        # -------------------------
        # Store results
        # -------------------------
        results.append({

            "model": model_name,

            "training_time_ms": round(
                median_training_time,
                4
            ),

            "inference_latency_ms": round(
                median_inference_time,
                4
            ),

            "model_size_bytes":
                model_size_bytes,

            "model_size_kb": round(
                model_size_kb,
                4
            ),

            "training_peak_memory_mb": round(
                training_peak_memory,
                4
            ),

            "inference_peak_memory_mb": round(
                inference_peak_memory,
                4
            )
        })

    # -------------------------
    # Save CSV
    # -------------------------
    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        "lab01/results/system_measurements.csv",
        index=False
    )

    print("\nMeasurement results:")
    print(results_df)


if __name__ == "__main__":
    main()