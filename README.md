# Customer-Enquiry Routing App

Foundations of Data Science, Option A. Continued from starter commit `58937af`;
the original 12 intents, source IDs and development split assignments are retained.
Checkpoint 2 adds the five-screen Streamlit workspace and Windows-verified
routing, saved-result inspection and CSV exports. The analysis-agent screen
shows its pending integration status. The API controller and written report
follow in later checkpoints.

## Install and reproduce

Python 3.12 is required. Create and activate a virtual environment, then run:

```bash
python -m pip install -r requirements.txt
python download_models.py
python -m pytest checks
python -m streamlit run app.py
```

The app loads the submitted models and results at startup. To regenerate the
statistical artifacts separately, run:

```bash
python train_evaluate.py --config config.json
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

## Application screens

- **Data overview** filters development messages by subset and category, with
  counts, text-length distributions, source references and a CSV download.
- **Route an enquiry** applies the saved Route model, temperature and review
  rule. It displays the top three categories, all 12 probabilities, confidence,
  status and model/run references, and exports the submitted text and result.
- **Explore request groups** browses all six seed-0 method/count settings,
  group sizes, category counts and up to five ranked representatives. Its
  quality panel includes ARI, entropy, MI and the 100 shuffled MI values.
  The MI histogram uses a labelled logarithmic axis to show the shuffled
  distribution and the much larger observed MI together.
- **Compare models** shows all four models on all four subsets, including
  per-category scores, confusion counts, actual mistakes, selection results,
  original/adjusted reliability, frozen review outcomes and the fixed paired
  bootstrap. Downloads carry the selected model, subset and run references.
- **Ask the analysis agent** provides the question interface and an explicit
  pending status. Its Ask button will be enabled when the API tools and
  controller are integrated in checkpoint 3. Live API evaluation awaits the
  user's key setup and execution.

Source examples are displayed as data. Routing validation clears an earlier
result when a subsequent submission fails. Missing encoder files produce an
actionable error; the saved-result screens can still be used.

Official test files belong in the submission (PDF pages 11–15, 24). Only agent
retrieval of test data is prohibited. Development result partitions are saved
separately for the agent. `calibration.csv` has `kind=confidence_bin`,
`threshold_trial` or `saved_rule`. Undefined values are empty CSV cells/JSON null.
Zero-denominator precision/recall/F1 are 0. Actual losses use stable log-softmax.

## Current validation

45 checks pass on Windows with Python 3.12.16, including the original nine
statistical checks and 36 interface/export checks. The real-inference check
runs when the pinned encoder has been downloaded; pytest reports it as skipped
when the weights are unavailable. Browser checks also exercised navigation,
routing and actual CSV downloads.

Route uses encoder C=10, T=0.673121, threshold 0.5. Official test accuracy is
95.833%. A delivery enquiry phrased as "My payment card still has not arrived."
was misclassified as a declined card payment during routing checks. Review
status follows the saved confidence rule, so a confident mistake can still be
accepted automatically. See [checkpoint 2 verification](docs/milestone-02.txt).
