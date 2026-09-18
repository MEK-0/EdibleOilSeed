# Website assets and provenance

## Official logo — required

Provide the official, unmodified TÜBİTAK logo as `docs/assets/tubitak-logo.png`
(preferred) or `docs/assets/tubitak-logo.svg`. The page tries these local assets
in that order and retains a translated text placeholder when neither exists.
No approximate logo or third-party hotlink is used. The absence of these two
optional assets is expected until the official file is supplied.

## Reused assets

- `figures/comparison_accuracy.png`: unchanged copy of `results/plots/comparison_accuracy.png`.
- `figures/comparison_f1_score.png`: unchanged copy of `results/plots/comparison_f1_score.png`.
- `results/vgg16_comparison.csv` and `results/inceptionv3_comparison.csv`: unchanged copies of the corresponding original-project result files.

These are historical project results, **not** the publication-stage result set.
Original plot labels are retained; captions and alternative text are bilingual.
The workflow near the top is a plain HTML summary, not a manuscript figure.

## Publication resources missing from this checkout

The website's publication-stage metrics, class counts, OOF summary, and final
protocol follow the manuscript-supported summary supplied in the website task.
No final publication CSV or final research manuscript was located in this checkout.
Earlier CSV metrics and training settings differ; the page explicitly separates them.

To add the original publication assets later, provide:

- `docs/assets/publication/workflow.png`
- `docs/assets/publication/class-distribution.png`
- `docs/assets/publication/accuracy-comparison.png`
- `docs/assets/publication/macro-f1-comparison.png`
- `docs/assets/publication/vgg16-mlp-oof-confusion.png`
- `docs/assets/publication/vgg16-mlp-oof-normalized.png`
- `docs/assets/publication/manuscript.pdf`
- `docs/assets/publication/results.csv`

Then update the corresponding HTML sections and both dictionaries in
`lang-data.js`, verifying provenance before replacing any historical figure.
There are deliberately no links or image elements pointing to these missing files.
Do not relabel `results/confusion_matrices/cm_*_svm.png` as OOF plots:
`results/visualize_results.py` generates these with an SVM on a 20% holdout.
No PDF screenshots, CNN execution, feature extraction, or training was used.
