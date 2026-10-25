"""
Real-time phishing URL detection using Random Forest, DNN, and CNN.

Testing was done on Google Colab - High-RAM, CPU-only VM with the below specifications.

Host and package versions:

  CPU: AMD EPYC 7B12 (8 vCPUs), x86_64 (KVM virtualized)
  RAM: 50 GiB
  OS: Ubuntu 22.04.5 LTS

  python: 3.12.12
  pandas: 2.2.2
  numpy: 2.0.2
  scikit-learn: 1.6.1
  tensorflow: 2.19.0

Datasets (Expected to be in the same folder as the .py file with the default .csv names):
    1. Kaggle Phishing Site URLs:
       https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls

    2. UCI PhiUSIIL Phishing URL (Website):
       https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset

Note: As this is run on a CPU-only service, the full execution time for the script is close to 2 hours.
"""

import re
import time
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,classification_report)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


# URL preprocessing & features

CHAR_ALPHABET = list("abcdefghijklmnopqrstuvwxyz0123456789/:.?&=%-_@#")
CHAR_TO_INDEX = {ch: i + 1 for i, ch in enumerate(CHAR_ALPHABET)}  # 0 reserved for PAD/unknown
MAX_URL_LEN = 256  #CNN max seq. length

# Extract domain/hostname from a URL string
def extract_domain(url: str) -> str:

    u = str(url).strip().lower()
    u = re.sub(r"^https?://", "", u)
    parts = u.split("/", 1)
    return parts[0]

# Create 10-dimensional lexical feature vector for each URL
def build_lexical_features(urls: pd.Series, domain_counts: pd.Series | None = None) -> pd.DataFrame:

    urls = urls.fillna("").astype(str)
    domains = urls.apply(extract_domain)

    if domain_counts is None:
        domain_counts = domains.value_counts()

    def domain_popularity(d: str) -> float:
        return float(np.log1p(domain_counts.get(d, 0)))

    url_length = urls.str.len()
    domain_length = domains.str.len()
    num_dots = urls.str.count(r"\.")
    num_slashes = urls.str.count("/")
    num_digits = urls.apply(lambda s: sum(ch.isdigit() for ch in s))
    num_special = urls.apply(lambda s: sum(not ch.isalnum() for ch in s))
    urls_lower = urls.str.lower()
    has_https = urls_lower.str.startswith("https").astype(int)
    kw_login = urls.str.contains("login", case=False, regex=False).astype(int)
    kw_verify = urls.str.contains("verify", case=False, regex=False).astype(int)
    dom_pop = domains.apply(domain_popularity)

    feat_df = pd.DataFrame(
        {
            "url_length": url_length,
            "domain_length": domain_length,
            "num_dots": num_dots,
            "num_slashes": num_slashes,
            "num_digits": num_digits,
            "num_special": num_special,
            "has_https": has_https,
            "kw_login": kw_login,
            "kw_verify": kw_verify,
            "domain_popularity": dom_pop,
        }
    )

    return feat_df

# Convert URLs to fixed-length integer sequences for CNN input.
def urls_to_char_sequences(urls: pd.Series, maxlen: int = MAX_URL_LEN) -> np.ndarray:

    urls = urls.fillna("").astype(str)
    num_samples = len(urls)
    seqs = np.zeros((num_samples, maxlen), dtype=np.int32)

    for i, url in enumerate(urls):
        u = url.strip().lower()
        u = re.sub(r"^https?://", "", u)
        for j, ch in enumerate(u[:maxlen]):
            seqs[i, j] = CHAR_TO_INDEX.get(ch, 0)

    return seqs



# Load Kaggle dataset

def load_kaggle_dataset(csv_path: str) -> pd.DataFrame:

    df = pd.read_csv(csv_path)
    cols = [c.lower() for c in df.columns]

    if "url" in cols:
        url_col = df.columns[cols.index("url")]
    elif "urls" in cols:
        url_col = df.columns[cols.index("urls")]
    else:
        raise ValueError(f"Could not find URL column in {csv_path}")

    label_candidates = ["label", "labels", "class"]
    label_col = None
    for cand in label_candidates:
        if cand in cols:
            label_col = df.columns[cols.index(cand)]
            break
    if label_col is None:
        raise ValueError(f"Could not find label column in {csv_path}")

    df = df[[url_col, label_col]].rename(columns={url_col: "url", label_col: "label"})

    # Map labels to phishing=1, benign=0

    if df["label"].dtype == object:
        lower = df["label"].astype(str).str.lower()
        mapping = {
            "good": 0,
            "benign": 0,
            "legitimate": 0,
            "bad": 1,
            "phishing": 1,
            "malicious": 1,
        }
        df["label"] = lower.map(mapping)
    else:
        # If 0/1 but unknown semantics, assume 1=phishing, 0=benign.
        df["label"] = df["label"].astype(int)

    df = df.dropna(subset=["url", "label"])
    df["url"] = df["url"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["url"])
    df = df[df["label"].isin([0, 1])]

    return df.reset_index(drop=True)

# Load PhiUSIIL dataset

def load_phiusiil_dataset(csv_path: str) -> pd.DataFrame:

    df = pd.read_csv(csv_path)

    cols = [c.lower() for c in df.columns]
    if "url" in cols:
        url_col = df.columns[cols.index("url")]
    else:
        raise ValueError(f"Could not find URL column in {csv_path}")

    label_candidates = ["label", "class", "target"]
    label_col = None
    for cand in label_candidates:
        if cand in cols:
            label_col = df.columns[cols.index(cand)]
            break
    if label_col is None:
        raise ValueError(f"Could not find label column in {csv_path}")

    df = df[[url_col, label_col]].rename(columns={url_col: "url", label_col: "label"})

    df["label"] = df["label"].astype(int)
    df["label"] = (df["label"] == 0).astype(int)  # Map 1=phishing, 0=benign

    df = df.dropna(subset=["url", "label"])
    df["url"] = df["url"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["url"])

    return df.reset_index(drop=True)


#Build Somesha-style DNN model

def build_dnn(input_dim: int) -> tf.keras.Model:

    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(16, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# Build Wei-style CNN
def build_cnn(vocab_size: int, maxlen: int = MAX_URL_LEN) -> tf.keras.Model:
    model = models.Sequential(
        [
            layers.Input(shape=(maxlen,)),
            layers.Embedding(input_dim=vocab_size, output_dim=32),
            layers.Conv1D(128, kernel_size=5, activation="relu"),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(128, kernel_size=5, activation="relu"),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(128, kernel_size=5, activation="relu"),
            layers.GlobalMaxPooling1D(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# Report model metrics
def evaluate_predictions(
    name: str, y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    y_pred = (y_proba >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred)

    print(f"\n=== {name} ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print("Confusion matrix [TN FP; FN TP]:")
    print(cm)
    print("Classification report:")
    print(classification_report(y_true, y_pred, digits=4))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm,
    }

# Report model latency for URLs
def measure_latency(predict_fn, X: np.ndarray, runs: int = 3, batch_size: int = 1024):
    warm_n = min(batch_size, len(X))
    _ = predict_fn(X[:warm_n])

    total_time = 0.0
    total_samples = len(X) * runs

    for _ in range(runs):
        start = time.perf_counter()
        if batch_size and batch_size < len(X):
            probs = []
            for i in range(0, len(X), batch_size):
                probs.append(predict_fn(X[i: i + batch_size]))
            _ = np.concatenate(probs, axis=0)
        else:
            _ = predict_fn(X)
        end = time.perf_counter()
        total_time += end - start

    avg_time_per_sample = total_time / total_samples
    throughput = 1.0 / avg_time_per_sample if avg_time_per_sample > 0 else float("inf")

    print(
        f"Average latency: {avg_time_per_sample * 1e3:.4f} ms/URL, "
        f"Throughput: {throughput:.1f} URLs/sec"
    )
    return avg_time_per_sample, throughput


# Create splits for train, validation, test

def prepare_data(df: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15, random_state: int = 42):
    urls = df["url"]
    labels = df["label"].astype(int).values

    X_train_idx, X_temp_idx, y_train, y_temp = train_test_split(
        np.arange(len(df)),
        labels,
        test_size=(val_size + test_size),
        stratify=labels,
        random_state=random_state,
    )

    rel_test_size = test_size / (val_size + test_size)
    X_val_idx, X_test_idx, y_val, y_test = train_test_split(
        X_temp_idx,
        y_temp,
        test_size=rel_test_size,
        stratify=y_temp,
        random_state=random_state,
    )

    train_domains = df.loc[X_train_idx, "url"]
    train_domain_counts = train_domains.apply(extract_domain).value_counts()

    lex_all = build_lexical_features(df["url"], domain_counts=train_domain_counts)
    seq_all = urls_to_char_sequences(df["url"])

    X_train_lex = lex_all.iloc[X_train_idx].values
    X_val_lex = lex_all.iloc[X_val_idx].values
    X_test_lex = lex_all.iloc[X_test_idx].values

    X_train_seq = seq_all[X_train_idx]
    X_val_seq = seq_all[X_val_idx]
    X_test_seq = seq_all[X_test_idx]

    return (
        (X_train_lex, X_val_lex, X_test_lex),
        (X_train_seq, X_val_seq, X_test_seq),
        (y_train, y_val, y_test),
    )


# Train and evaluate RF, DNN, and CNN on a single dataset using its own train/val/test splits.
def run_experiments_on_dataset(name: str, df: pd.DataFrame):

    print(f"\n *****DATASET: {name}*****")
    print(f"Total unique URLs: {len(df)}")
    print(f"Phishing ratio: {df['label'].mean():.4f}")

    (X_train_lex, X_val_lex, X_test_lex), (
        X_train_seq,
        X_val_seq,
        X_test_seq,
    ), (y_train, y_val, y_test) = prepare_data(df)

    # Standardize lexical features for DNN
    scaler = StandardScaler()
    X_train_lex_scaled = scaler.fit_transform(X_train_lex)
    X_val_lex_scaled = scaler.transform(X_val_lex)
    X_test_lex_scaled = scaler.transform(X_test_lex)


    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train_lex, y_train)
    rf_proba = rf.predict_proba(X_test_lex)[:, 1]
    eval_rf = evaluate_predictions(f"{name} - Random Forest", y_test, rf_proba)

    print(f"Latency (RF) on {name} test set:")
    measure_latency(lambda X: rf.predict_proba(X)[:, 1], X_test_lex, runs=3)


    dnn = build_dnn(input_dim=X_train_lex_scaled.shape[1])
    es_cb = callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    dnn.fit(
        X_train_lex_scaled,
        y_train,
        validation_data=(X_val_lex_scaled, y_val),
        epochs=50,
        batch_size=256,
        callbacks=[es_cb],
        verbose=2,
    )
    dnn_proba = dnn.predict(X_test_lex_scaled, batch_size=1024, verbose=0).ravel()
    eval_dnn = evaluate_predictions(f"{name} - DNN (lexical)", y_test, dnn_proba)

    print(f"Latency (DNN) on {name} test set:")
    measure_latency(
        lambda X: dnn.predict(scaler.transform(X), batch_size=1024).ravel(),
        X_test_lex,
        runs=3,
        batch_size=1024,
    )


    vocab_size = len(CHAR_ALPHABET) + 1
    cnn = build_cnn(vocab_size=vocab_size, maxlen=MAX_URL_LEN)
    es_cb_cnn = callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    cnn.fit(
        X_train_seq,
        y_train,
        validation_data=(X_val_seq, y_val),
        epochs=50,
        batch_size=256,
        callbacks=[es_cb_cnn],
        verbose=2,
    )
    cnn_proba = cnn.predict(X_test_seq, batch_size=1024, verbose=0).ravel()
    eval_cnn = evaluate_predictions(f"{name} - CNN (char-level)", y_test, cnn_proba)

    print(f"Latency (CNN) on {name} test set:")
    measure_latency(
        lambda X: cnn.predict(X, batch_size=1024).ravel(),
        X_test_seq,
        runs=3,
        batch_size=1024,
    )

    return {
        "rf": eval_rf,
        "dnn": eval_dnn,
        "cnn": eval_cnn,
    }


#Train all models on the training split of train_df and evaluate on all of test_df (no tuning on test_df).
def run_cross_dataset_experiments(
    train_name: str,
    train_df: pd.DataFrame,
    test_name: str,
    test_df: pd.DataFrame,
    val_size: float = 0.15,
    random_state: int = 42,
):
    print(f"\n***** CROSS-DATASET: Train on {train_name}, Test on {test_name} *****")

    labels = train_df["label"].astype(int).values
    idx_all = np.arange(len(train_df))

    test_size = val_size
    temp_size = val_size + test_size

    train_idx, temp_idx, y_train, y_temp = train_test_split(
        idx_all,
        labels,
        test_size=temp_size,
        stratify=labels,
        random_state=random_state,
    )


    rel_test_size = test_size / temp_size
    val_idx, _test_idx_unused, y_val, _y_test_unused = train_test_split(
        temp_idx,
        y_temp,
        test_size=rel_test_size,
        stratify=y_temp,
        random_state=random_state,
    )


    train_domains = train_df["url"].iloc[train_idx].apply(extract_domain)
    train_domain_counts = train_domains.value_counts()


    lex_all = build_lexical_features(train_df["url"], domain_counts=train_domain_counts)
    seq_all = urls_to_char_sequences(train_df["url"])

    X_train_lex = lex_all.iloc[train_idx].values
    X_val_lex = lex_all.iloc[val_idx].values

    X_train_seq = seq_all[train_idx]
    X_val_seq = seq_all[val_idx]

    test_lex_df = build_lexical_features(test_df["url"], domain_counts=train_domain_counts)
    X_test_lex = test_lex_df.values
    test_seq = urls_to_char_sequences(test_df["url"])
    y_test = test_df["label"].astype(int).values


    scaler = StandardScaler()
    X_train_lex_scaled = scaler.fit_transform(X_train_lex)
    X_val_lex_scaled = scaler.transform(X_val_lex)
    X_test_lex_scaled = scaler.transform(X_test_lex)


    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )

    rf.fit(X_train_lex, y_train)
    rf_proba = rf.predict_proba(X_test_lex)[:, 1]
    eval_rf = evaluate_predictions(
        f"Cross {train_name}->{test_name} - Random Forest", y_test, rf_proba
    )


    dnn = build_dnn(input_dim=X_train_lex_scaled.shape[1])
    es_cb = callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    dnn.fit(
        X_train_lex_scaled,
        y_train,
        validation_data=(X_val_lex_scaled, y_val),
        epochs=50,
        batch_size=256,
        callbacks=[es_cb],
        verbose=2,
    )
    dnn_proba = dnn.predict(X_test_lex_scaled, batch_size=1024).ravel()
    eval_dnn = evaluate_predictions(
        f"Cross {train_name}->{test_name} - DNN", y_test, dnn_proba
    )


    vocab_size = len(CHAR_ALPHABET) + 1
    cnn = build_cnn(vocab_size=vocab_size, maxlen=MAX_URL_LEN)
    es_cb_cnn = callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    cnn.fit(
        X_train_seq,
        y_train,
        validation_data=(X_val_seq, y_val),
        epochs=50,
        batch_size=256,
        callbacks=[es_cb_cnn],
        verbose=2,
    )
    cnn_proba = cnn.predict(test_seq, batch_size=1024).ravel()
    eval_cnn = evaluate_predictions(
        f"Cross {train_name}->{test_name} - CNN", y_test, cnn_proba
    )

    return {
        "rf": eval_rf,
        "dnn": eval_dnn,
        "cnn": eval_cnn,
    }


def main():
    kaggle_df = load_kaggle_dataset("./phishing_site_urls.csv")
    phiusiil_df = load_phiusiil_dataset("./PhiUSIIL_Phishing_URL_Dataset.csv")

    # Within-dataset experiments
    results_kaggle = run_experiments_on_dataset("Kaggle Phishing Site URLs", kaggle_df)
    results_phiusiil = run_experiments_on_dataset("UCI PhiUSIIL", phiusiil_df)

    # Cross-dataset experiments
    cross_kaggle_to_phi = run_cross_dataset_experiments(
        "Kaggle", kaggle_df, "PhiUSIIL", phiusiil_df
    )
    cross_phi_to_kaggle = run_cross_dataset_experiments(
        "PhiUSIIL", phiusiil_df, "Kaggle", kaggle_df
    )

if __name__ == "__main__":
    main()
