import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2
LATENCY_RUNS = 1000
THROUGHPUT_BATCH_SIZE = 1000


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------
load_start = time.perf_counter()

df = pd.read_csv("creditcard.csv")

data_load_seconds = time.perf_counter() - load_start

X = df.drop(columns=["Class"])
y = df["Class"]


# --------------------------------------------------
# 2. Train / Test split
# --------------------------------------------------
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=y,
)

# Validation is 20% of the remaining training data.
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=VAL_SIZE,
    random_state=SEED,
    stratify=y_train_val,
)


# --------------------------------------------------
# 3. Model
# --------------------------------------------------
model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    random_state=SEED,
)


# --------------------------------------------------
# 4. Training + early stopping
# --------------------------------------------------
training_start = time.perf_counter()

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[
        lgb.early_stopping(
            stopping_rounds=50,
            verbose=False,
        )
    ],
)

training_seconds = time.perf_counter() - training_start


# --------------------------------------------------
# 5. Final evaluation on TEST only
# --------------------------------------------------
y_probability = model.predict_proba(X_test)[:, 1]
y_prediction = model.predict(X_test)

auc_roc = roc_auc_score(y_test, y_probability)
accuracy = accuracy_score(y_test, y_prediction)
f1 = f1_score(y_test, y_prediction)
precision = precision_score(y_test, y_prediction)
recall = recall_score(y_test, y_prediction)


# --------------------------------------------------
# 6. Single-row latency
# --------------------------------------------------
single_row = X_test.iloc[[0]]

# Warm-up
for _ in range(20):
    model.predict_proba(single_row)

latencies_ms = []

for _ in range(LATENCY_RUNS):
    start = time.perf_counter()

    model.predict_proba(single_row)

    elapsed = time.perf_counter() - start
    latencies_ms.append(elapsed * 1000)

latency_1_row_ms = float(np.mean(latencies_ms))


# --------------------------------------------------
# 7. Throughput: batch of 1000 rows
# --------------------------------------------------
batch = X_test.iloc[:THROUGHPUT_BATCH_SIZE]

# Warm-up
model.predict_proba(batch)

start = time.perf_counter()

model.predict_proba(batch)

batch_seconds = time.perf_counter() - start

throughput = len(batch) / batch_seconds


# --------------------------------------------------
# 8. Save results
# --------------------------------------------------
results = {
    "data_load_seconds": data_load_seconds,

    "split": {
        "test_size": TEST_SIZE,
        "validation_size_of_train": VAL_SIZE,
    },

    "seed": SEED,

    "training_seconds": training_seconds,

    "best_iteration": model.best_iteration_,

    "auc_roc": auc_roc,
    "accuracy": accuracy,
    "f1": f1,
    "precision": precision,
    "recall": recall,

    "latency_1_row_ms": latency_1_row_ms,
    "latency_runs": LATENCY_RUNS,

    "throughput_1000_rows_per_second": throughput,
    "throughput_batch_size": len(batch),
}


with open("benchmark_result.json", "w") as f:
    json.dump(results, f, indent=2)


print(json.dumps(results, indent=2))
