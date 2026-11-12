# Third-party sources

**BANKING77** — PolyAI, Casanueva et al. (2020), *Efficient Intent Detection with
Dual Sentence Encoders*, ACL ConvAI. Source:
https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data.
License: Creative Commons Attribution 4.0 International. The original dataset
license is retained in `banking_data/LICENSE`. Texts and category labels remain
unchanged; derived files add IDs, splits, numeric features and model predictions.

**all-MiniLM-L6-v2** — Sentence Transformers, Apache-2.0. Fixed model revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`:
https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41.
Downloaded weights are excluded from Git. The original model card is included
in the download. Used frozen for normalized 384-dimensional features.

Software dependencies retain their respective licenses; tested versions are
pinned in `requirements.txt`.
