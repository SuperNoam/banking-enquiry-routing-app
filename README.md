# Customer-Enquiry Routing App

Foundations of Data Science, Option A. Continued from starter commit `58937af`;
the original 12 intents, source IDs and development split assignments are retained.
This checkpoint completes statistical training and saved evaluation. The UI,
API agent and written report follow in later checkpoints.

## Install and reproduce

Python 3.12 is required. Create and activate a virtual environment, then run:

```bash
python -m pip install -r requirements.txt
python download_models.py
python train_evaluate.py --config config.json
python -m pytest checks
```

Saved classifiers and results are included. Download the fixed MiniLM encoder
once (about 90 MB). Its model ID and immutable revision are in `config.json`;
`download_models.py` retrieves the required files. Encoder weights/caches are
ignored by Git. No local generative LLM is used. Copy `.env.example` to `.env`
and add your API key later; `.env` must never be committed.

## Data and attribution

The supplied `banking_data/train.csv` and `test.csv` are the original starter
snapshot of PolyAI BANKING77. Training has 10,003 rows; test has 3,080 rows across
77 labels. The 12 supported labels are in [intents.json](intents.json).
Source: <https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data>.
Dataset license: **CC BY 4.0**, included in [banking_data/LICENSE](banking_data/LICENSE).
Cite Casanueva et al. (2020),
[*Efficient Intent Detection with Dual Sentence Encoders*](https://arxiv.org/abs/2003.04807).
See [THIRD_PARTY.md](THIRD_PARTY.md).

To download official files alongside the included snapshot:

```bash
python -c "import urllib.request,pathlib; p=pathlib.Path('banking_data'); p.mkdir(exist_ok=True); [urllib.request.urlretrieve('https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/'+n,str(p/('downloaded_'+n))) for n in ['train.csv','test.csv']]"
```

The starter files use LF newlines; official bytes may use CRLF. Original texts
and labels are retained. Local SHA256 values and CSV line numbers are saved
per row. Source IDs use zero-based original row indexes (`train_00000`).

## Statistical deliverables

- [Data audit](data/development_summary.json), [development rows](data/development.csv),
  [official test rows](data/official_test.csv), [exclusions](data/excluded.csv).
- [Settings](config.json), [freeze manifest](artifacts/freeze_manifest.json),
  [classifiers](artifacts/classifiers.joblib), [PCA](artifacts/pca.joblib),
  [group models](artifacts/clustering.joblib).
- [Six selection losses](results/selection_results.csv), [scores](results/metrics.csv),
  [predictions](results/predictions.csv), [per-category scores](results/per_category.csv),
  [confusion counts](results/confusion.csv), [test mistakes](results/test_mistakes.csv).
- [Confidence and review](results/calibration.csv), [bootstrap](results/bootstrap.csv),
  [bootstrap samples](results/bootstrap_samples.csv).
- [All group assignments](results/clusters.csv), [group quality](results/clustering_results.csv),
  [shuffled MI](results/clustering_shuffles.csv), [default](results/grouping_default.json).
- [Checkpoint 1 and commit commands](docs/milestone-01.txt).

Official test files belong in the submission (PDF pages 11–15, 24). Only agent
retrieval of test data is prohibited. Development result partitions are saved
separately for the agent. `calibration.csv` has `kind=confidence_bin`,
`threshold_trial` or `saved_rule`. Undefined values are empty CSV cells/JSON null.
Zero-denominator precision/recall/F1 are 0. Actual losses use stable log-softmax.

## Current validation

9 statistical checks pass. Route uses encoder C=10, T=0.673121, threshold 0.5.
Official test accuracy is 95.833%. Live API agent evaluation is left for the user
after adding a key. This checkpoint is not the final project submission.
