# Lab 1 Report

## Goal

The goal of this lab was to prepare a working Python environment and test two simple machine learning models.

I also measured the performance of the models by checking training time, inference latency, model size, and memory usage.

The last part was to decide where the models could be deployed, such as Cloud, Edge, Mobile, or TinyML.

## Method

I used the Breast Cancer Wisconsin dataset from scikit-learn.

The dataset was split into 70% training data and 30% test data.

I used stratified splitting with `random_state=42`.

Two models were trained:

- Logistic Regression
- Random Forest

The models were created with these settings:

- Logistic Regression: `max_iter=1000`, `random_state=42`
- Random Forest: `n_estimators=100`, `random_state=42`

For training time, I first made one warm-up run. Then I measured the training time 5 times and used the median value.

For inference latency, I used one test sample and repeated prediction 100 times. The median value was used.

The models were saved using `joblib` and their file sizes were measured.

Peak memory usage was also measured during training and inference.

## Results

### Accuracy

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | 0.9415 |
| Random Forest | 0.9357 |

Logistic Regression had slightly higher accuracy than Random Forest in this experiment.

### System Measurements

| Model | Training Time (ms) | Inference Latency (ms) | Model Size (KB) | Training Peak Memory (MB) | Inference Peak Memory (MB) |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 117.2105 | 0.0372 | 1.0830 | 147.4375 | 147.4375 |
| Random Forest | 114.5968 | 1.7603 | 284.0869 | 149.3906 | 149.3906 |

## Deployment Suitability

The deployment budgets were compared with my measurements.

### Logistic Regression

The Logistic Regression model has very low inference latency and a very small model size.

It can be used for:

- Cloud
- Edge
- Mobile

It is not suitable for TinyML because the measured memory usage is much higher than the TinyML memory limit.

### Random Forest

The Random Forest model also has low latency and can run on:

- Cloud
- Edge
- Mobile

It is not suitable for TinyML because both its model size and memory usage are too high.

## Conclusions

1. Logistic Regression gave slightly better accuracy than Random Forest in this lab.
2. Logistic Regression had a much smaller model size and lower inference latency.
3. Both models are suitable for Cloud, Edge, and Mobile environments, but neither model is suitable for TinyML with the measured resource usage.