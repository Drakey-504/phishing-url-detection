# Real-Time Phishing URL Detection: Random Forest vs. DNN vs. Character-Level CNN

Given only a URL string, how accurately and how quickly can a model tell phishing from legitimate? This project compares three model families on two public datasets, measures CPU inference speed, and tests whether models trained on one dataset still work on the other.

**Course:** 95-747 Security Data Analytics, Carnegie Mellon University, Fall 2025

## Key findings

- **The character-level CNN is the strongest model when both classes look alike.** On the Kaggle dataset, where phishing and legitimate URLs share a similar format (almost none include `http(s)://`), the CNN reached 97.6% accuracy, F1 0.947, and ROC-AUC 0.995. The Random Forest and DNN, which only see 10 hand-crafted lexical features, reached F1 0.78 and 0.65.
- **Models that look excellent on one dataset fail on the other.** Every model trained on one dataset dropped to ROC-AUC 0.24–0.71 when tested on the other, including models that scored above 99% within their own dataset.
- **All three models are fast enough for real-time use on a CPU.** Averaged over batched inference, the Random Forest and DNN took about 0.04–0.05 ms per URL (20K–24K URLs/sec) and the CNN about 0.21 ms (about 4.7K URLs/sec).
- **The near-perfect PhiUSIIL scores reflect how that dataset formats its URLs.** See [Follow-up after the course](#follow-up-after-the-course-october-2026): a one-line formatting rule matches the Random Forest's accuracy there.

## Results

**Within each dataset** (70/15/15 stratified train/validation/test split; test-set results):

| Dataset | Model | Accuracy | F1 (phishing) | ROC-AUC | Latency (ms/URL) | Throughput (URLs/sec) |
|---|---|---|---|---|---|---|
| Kaggle (507,192 URLs, 22.5% phishing) | Random Forest | 0.896 | 0.781 | 0.955 | 0.050 | 19.9K |
| | DNN | 0.874 | 0.654 | 0.915 | 0.042 | 23.9K |
| | Character CNN | **0.976** | **0.947** | **0.995** | 0.215 | 4.7K |
| PhiUSIIL (235,370 URLs, 42.7% phishing) | Random Forest | 0.996 | 0.995 | 0.998 | 0.047 | 21.4K |
| | DNN | 0.994 | 0.993 | 0.999 | 0.043 | 23.5K |
| | Character CNN | 0.911 | 0.886 | 0.944 | 0.211 | 4.7K |

**Across datasets** (trained on one dataset, tested on all of the other):

| Train → Test | Model | Accuracy | ROC-AUC |
|---|---|---|---|
| Kaggle → PhiUSIIL | Random Forest | 0.696 | 0.708 |
| | DNN | 0.255 | 0.245 |
| | Character CNN | 0.439 | 0.617 |
| PhiUSIIL → Kaggle | Random Forest | 0.225 | 0.574 |
| | DNN | 0.231 | 0.491 |
| | Character CNN | 0.225 | 0.49 |

Trained on PhiUSIIL, all three models flag nearly every Kaggle URL as phishing (recall 0.98–1.00), so their accuracy falls to about Kaggle's 22.5% phishing rate.

Latency is the average time per URL when scoring the test set in batches of 1,024 on CPU (Google Colab, 8 vCPUs), averaged over 3 runs. It covers model inference on precomputed features and character sequences; feature extraction is not timed. Scoring a single URL at a time would likely take longer per URL.

## Models

- **Random Forest:** 200 trees on 10 lexical features: URL and domain length, counts of dots, slashes, digits, and special characters, an HTTPS flag, "login" and "verify" keyword flags, and how often the domain appears in the training set.
- **DNN:** a compact network (32 → 16 units, dropout) on the same 10 features, inspired by Somesha et al.
- **Character-level CNN:** URLs (scheme removed, first 256 characters) mapped to character IDs, then an embedding layer and three 1D convolution layers, inspired by Wei et al.

Both neural networks use early stopping on the validation set. All results come from a single run; the data splits and the Random Forest use fixed seeds, the neural networks do not.

## Follow-up after the course (October 2026)

*Added after the course; not part of the submitted project.*

**What this checks.** Whether the results hold up once the differences in how the datasets format their URLs are taken out. Specifically:
- Do the near-perfect PhiUSIIL scores (99.6% accuracy for the Random Forest) come from recognizing phishing, or from URL formatting?
- Does the formatting difference between the two datasets explain why the models fail across datasets?

`scripts/check_url_format_artifacts.py` compares URL formatting between the classes, scores a one-line formatting rule, and retrains the course Random Forest (same features, settings, and splits) with the scheme and a leading `www.` removed, within PhiUSIIL and across datasets in both directions.

**Why it matters.** In PhiUSIIL, every legitimate URL is a bare homepage written exactly as `https://www.<domain>`, with no path. Phishing URLs mostly are not: only 49% use HTTPS, 42% start with `www.`, and 28% have a path. A model can separate the classes by checking the format alone, which likely would not carry over to real traffic. Kaggle URLs, by contrast, are stored without `http(s)://` in both classes.

**Result.**
- **The PhiUSIIL score is mostly formatting.** The rule "phishing unless the URL is exactly `https://www.<domain>`" scores **99.6% accuracy with 100% precision** on PhiUSIIL, the same as the Random Forest (99.59% vs. 99.61% on the same test split). Retraining the Random Forest with the scheme and a leading `www.` stripped from every URL drops its accuracy to **85.7%** (ROC-AUC 0.998 → 0.877). Some formatting signal still remains, since legitimate URLs never have a path.
- **The cross-dataset collapse is not a formatting artifact.** With the scheme and `www.` stripped from both datasets, transfer does not recover: Kaggle → PhiUSIIL ROC-AUC falls from 0.71 to 0.18, and PhiUSIIL → Kaggle stays near chance (0.57 → 0.49), still flagging almost every Kaggle URL as phishing. The two datasets differ in more than formatting.

**What this means.** The Kaggle results are the more reliable within-dataset comparison, because both classes share a similar format there. The PhiUSIIL scores mostly measure the dataset's formatting rather than phishing detection. The cross-dataset finding holds up: models trained on one source of URLs do not transfer to another, even with the formatting differences removed. These checks cover the Random Forest only; the DNN and CNN were not retrained. Full numbers are in `results/url_format_stats.csv` and `results/url_format_check.csv`.

The script's cross-dataset Random Forest results on unmodified URLs differ slightly from the course run (for example, Kaggle → PhiUSIIL ROC-AUC 0.714 vs. 0.708); the conclusions are the same.

## Running it

1. Download both datasets and place the CSVs in the project root with their default names:
   - [Kaggle Phishing Site URLs](https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls) → `phishing_site_urls.csv`
   - [UCI PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) → `PhiUSIIL_Phishing_URL_Dataset.csv`
2. Install the dependencies and run the experiments:
   ```bash
   pip install -r requirements.txt
   python phishing_url_detection.py
   ```
   The full run takes about 2 hours on CPU. `phishing_url_detection.ipynb` is the same code with the outputs from the reported run.
3. Optional: run the follow-up check (a few minutes, no TensorFlow needed):
   ```bash
   python scripts/check_url_format_artifacts.py
   ```

## Project structure

```
├── phishing_url_detection.py          # Experiments: loading, features, models, cross-dataset tests
├── phishing_url_detection.ipynb       # Same code, with outputs from the reported run
├── scripts/
│   └── check_url_format_artifacts.py  # Post-course URL-format check
├── results/                           # Output of the post-course check
├── requirements.txt
├── .gitignore                         # Datasets are not committed
└── README.md
```

## Stack

Python · scikit-learn · TensorFlow / Keras · pandas · NumPy · Google Colab