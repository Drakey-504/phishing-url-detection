"""
URL-format artifact check (added after the course).

Questions: do the near-perfect PhiUSIIL results come from detecting phishing,
or from how the dataset happens to format its URLs? And does the formatting
difference between the two datasets explain the cross-dataset collapse?

The script:
  1. Compares URL formatting (scheme, "https", "www.", path) between phishing
     and legitimate URLs in both datasets.
  2. Scores a one-line rule on PhiUSIIL: "phishing unless the URL is exactly
     https://www.<domain>", on the full dataset and on the same test split the
     Random Forest is scored on.
  3. Retrains the course Random Forest (same split, features and settings as
     phishing_url_detection.py) within PhiUSIIL and across datasets in both
     directions, once on the URLs as given and once with the scheme and a
     leading "www." stripped (which also zeroes the HTTPS flag), to see how much
     each result depends on formatting.

The loaders and lexical features below mirror phishing_url_detection.py so this
check runs without TensorFlow.

Usage (from the project root, with both CSVs in the root as for the main script):
    python scripts/check_url_format_artifacts.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_CSV = ROOT / "phishing_site_urls.csv"
PHIUSIIL_CSV = ROOT / "PhiUSIIL_Phishing_URL_Dataset.csv"
HOMEPAGE_RE = r"^https://www\.[^/]+/?$"


def load_kaggle(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).rename(columns={"URL": "url", "Label": "label"})
    df["label"] = df["label"].astype(str).str.lower().map({"good": 0, "bad": 1})
    df = df.dropna(subset=["url", "label"])
    df["url"] = df["url"].astype(str).str.strip()
    return df.drop_duplicates(subset=["url"]).reset_index(drop=True)


def load_phiusiil(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["URL", "label"]).rename(columns={"URL": "url"})
    df["label"] = (df["label"].astype(int) == 0).astype(int)  # 1 = phishing
    df = df.dropna(subset=["url", "label"])
    df["url"] = df["url"].astype(str).str.strip()
    return df.drop_duplicates(subset=["url"]).reset_index(drop=True)


def extract_domain(url: str) -> str:
    u = re.sub(r"^https?://", "", str(url).strip().lower())
    return u.split("/", 1)[0]


def lexical_features(urls: pd.Series, domain_counts: pd.Series) -> np.ndarray:
    urls = urls.fillna("").astype(str)
    domains = urls.apply(extract_domain)
    return pd.DataFrame({
        "url_length": urls.str.len(),
        "domain_length": domains.str.len(),
        "num_dots": urls.str.count(r"\."),
        "num_slashes": urls.str.count("/"),
        "num_digits": urls.apply(lambda s: sum(ch.isdigit() for ch in s)),
        "num_special": urls.apply(lambda s: sum(not ch.isalnum() for ch in s)),
        "has_https": urls.str.lower().str.startswith("https").astype(int),
        "kw_login": urls.str.contains("login", case=False, regex=False).astype(int),
        "kw_verify": urls.str.contains("verify", case=False, regex=False).astype(int),
        "domain_popularity": domains.apply(lambda d: float(np.log1p(domain_counts.get(d, 0)))),
    }).values


def format_stats(name: str, df: pd.DataFrame) -> list[dict]:
    u = df["url"].str.lower()
    host_and_path = u.str.replace(r"^https?://", "", regex=True)
    rows = []
    for label, cls in [(1, "phishing"), (0, "legitimate")]:
        m = df["label"] == label
        rows.append({
            "dataset": name, "class": cls, "n": int(m.sum()),
            "has_scheme": round(u[m].str.match(r"https?://").mean(), 4),
            "starts_https": round(u[m].str.startswith("https").mean(), 4),
            "starts_www": round(host_and_path[m].str.startswith("www.").mean(), 4),
            "has_path": round(host_and_path[m].str.contains(r"/.", regex=True).mean(), 4),
            "exact_https_www_homepage": round(u[m].str.match(HOMEPAGE_RE).mean(), 4),
        })
    return rows


def split_indices(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Same 70/15/15 stratified split as prepare_data() in phishing_url_detection.py."""
    y = df["label"].astype(int).values
    train_idx, temp_idx, _, y_temp = train_test_split(
        np.arange(len(df)), y, test_size=0.30, stratify=y, random_state=42)
    _, test_idx, _, _ = train_test_split(
        temp_idx, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    return train_idx, test_idx


def strip_format(urls: pd.Series) -> pd.Series:
    """Remove the scheme and a leading "www." (this also zeroes the HTTPS flag)."""
    urls = urls.str.replace(r"^https?://", "", regex=True, flags=re.I)
    return urls.str.replace(r"^www\.", "", regex=True, flags=re.I)


def random_forest(train: pd.DataFrame, test: pd.DataFrame | None, strip: bool) -> dict:
    """Course Random Forest, trained on the course training split of `train`.

    With test=None it is scored on the course test split of `train`; otherwise on
    all of `test` (the course cross-dataset setup).
    """
    train_urls = strip_format(train["url"]) if strip else train["url"]
    train_idx, test_idx = split_indices(train)
    counts = train_urls.iloc[train_idx].apply(extract_domain).value_counts()
    X_train = lexical_features(train_urls, counts)
    y_train = train["label"].astype(int).values
    if test is None:
        X_test, y_test = X_train[test_idx], y_train[test_idx]
    else:
        test_urls = strip_format(test["url"]) if strip else test["url"]
        X_test, y_test = lexical_features(test_urls, counts), test["label"].astype(int).values
    rf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_leaf=2,
                                class_weight="balanced_subsample", n_jobs=-1, random_state=42)
    rf.fit(X_train[train_idx], y_train[train_idx])
    proba = rf.predict_proba(X_test)[:, 1]
    return {
        "urls": "scheme and www. stripped" if strip else "as given (course setup)",
        "accuracy": round(accuracy_score(y_test, (proba >= 0.5).astype(int)), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "flagged_as_phishing": round(float((proba >= 0.5).mean()), 4),
    }


def main() -> int:
    kaggle, phi = load_kaggle(KAGGLE_CSV), load_phiusiil(PHIUSIIL_CSV)

    stats = pd.DataFrame(format_stats("Kaggle", kaggle) + format_stats("PhiUSIIL", phi))
    print("URL formatting by dataset and class (share of URLs):")
    print(stats.to_string(index=False))

    rule = ~phi["url"].str.lower().str.match(HOMEPAGE_RE)
    y = phi["label"] == 1
    tp, fp = int((rule & y).sum()), int((rule & ~y).sum())
    _, test_idx = split_indices(phi)
    rule_row = {"check": "homepage_rule_phiusiil",
                "accuracy": round(float((rule == y).mean()), 4),
                "precision": round(tp / (tp + fp), 4), "recall": round(tp / int(y.sum()), 4),
                "accuracy_test_split": round(float((rule.values[test_idx] == y.values[test_idx]).mean()), 4)}
    print("\nPhiUSIIL, rule 'phishing unless exactly https://www.<domain>' "
          "(full dataset; accuracy_test_split = Random Forest's test split):", rule_row)

    rf_rows = []
    for name, train, test in [("PhiUSIIL (within dataset)", phi, None),
                              ("Kaggle -> PhiUSIIL", kaggle, phi),
                              ("PhiUSIIL -> Kaggle", phi, kaggle)]:
        for strip in (False, True):
            rf_rows.append({"check": f"random_forest {name}", **random_forest(train, test, strip)})
    print("\nRandom Forest (course features and settings), URLs as given vs. formatting stripped:")
    print(pd.DataFrame(rf_rows).to_string(index=False))

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    stats.to_csv(out / "url_format_stats.csv", index=False)
    pd.DataFrame([rule_row] + rf_rows).to_csv(out / "url_format_check.csv", index=False)
    print(f"\nWrote {out / 'url_format_stats.csv'} and {out / 'url_format_check.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
