# Edible Oil Seed Classification

**Classification of Edible Oil Seeds Using Hybrid Deep Learning and Machine Learning Methods**

Mahmut Esat Kolay · Murat Köklü

Selçuk University, Faculty of Technology, Department of Computer Engineering, Konya, Türkiye

TÜBİTAK 2209-A University Students Research Projects Support Program · May 2026

## Technical report website

The bilingual English/Turkish static report is in [docs/index.html](docs/index.html).
Expected URL after activation: https://mek-0.github.io/EdibleOilSeed/

Activate GitHub Pages under **Settings → Pages → Deploy from a branch → main → /docs → Save**.
The website has no build step or framework dependency. This change does not publish or push the site.

Supply the official TÜBİTAK logo as `docs/assets/tubitak-logo.png` or
`docs/assets/tubitak-logo.svg`; see [asset notes](docs/assets/README.md).
No verified project number is included.

## Results and provenance

The original project preprocesses images, extracts fixed VGG16/InceptionV3
features, saves matrices under `models/dl_features/`, and evaluates ML classifiers.
The publication-stage evaluation described in the supplied manuscript summary
reuses those saved matrices without repeating preprocessing or CNN extraction.

The supplied publication summary reports VGG16–MLP mean stratified 10-fold
cross-validation accuracy of **99.8318%** (SD **0.1716 percentage points**),
macro F1 of **99.8340%** (SD **0.1687 percentage points**), and **17 / 10,111** OOF errors.
These are internal cross-validation results, not production accuracy estimates.

This checkout contains earlier CSVs in `results/`, whose values differ from the
publication summary, and earlier classifier scripts with different settings.
The website keeps those historical results separate. The final publication
manuscript, publication result CSVs, and VGG16–MLP OOF figures are not present;
the existing PDFs are project documents. Available SVM confusion matrices use a
20% holdout split, not OOF predictions. No research workloads were rerun for the site.

Dataset: [Edible Oil Seed Dataset (V2), Mendeley Data](https://doi.org/10.17632/x7h34tkwcp.2).
