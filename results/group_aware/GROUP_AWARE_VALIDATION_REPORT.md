# Group-Aware Validation Report

## 1. Amaç

Bu analiz, mevcut VGG16 ve InceptionV3 feature vektörleri üzerinde near-duplicate leakage riskini inceleyen bağımsız bir validation stress-testidir. CNN yeniden eğitilmemiş, feature extraction tekrarlanmamıştır.

## 2. Neden standart image-level CV iyimser olabilir?

Aynı fiziksel tohumun veya aynı capture oturumunun çok benzer görüntüleri train ve test fold'larına ayrılırsa image-level CV bağımsız genelleme performansını iyimser gösterebilir.

## 3. Mevcut dataset kısıtı

Gerçek physical-seed ID ve capture-session ID bulunmadığı için örnekler feature-space pseudo-group'ları ile gruplanmıştır. Bu sonuçlar gerçek sample/session grouping yerine geçmez.

## 4. Pseudo-group yöntemi

Feature vektörleri L2 normalize edildi. Her sınıf kendi içinde cosine distance kullanan brute-force NearestNeighbors ile tarandı; 0.9999, 0.999, 0.995, 0.990 ve 0.985 similarity eşikleri test edildi. Eşik grafı Union-Find connected components ile kuruldu. Sınıflar arası örnekler aynı gruba alınmadı ve her grup için label purity assert edildi.

## 5. Similarity threshold analizi

| DL model | Threshold | Groups | Singleton | Multi groups | Grouped samples | Grouped % | Largest | Mean multi size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VGG16 | 0.9999 | 9232 | 8926 | 306 | 1185 | 11.720 | 15 | 3.873 |
| VGG16 | 0.999 | 3260 | 2661 | 599 | 7450 | 73.682 | 1275 | 12.437 |
| VGG16 | 0.995 | 85 | 59 | 26 | 10052 | 99.416 | 2996 | 386.615 |
| VGG16 | 0.99 | 13 | 6 | 7 | 10105 | 99.941 | 3000 | 1443.571 |
| VGG16 | 0.985 | 7 | 1 | 6 | 10110 | 99.990 | 3000 | 1685.000 |
| InceptionV3 | 0.9999 | 10103 | 10095 | 8 | 16 | 0.158 | 2 | 2.000 |
| InceptionV3 | 0.999 | 10103 | 10095 | 8 | 16 | 0.158 | 2 | 2.000 |
| InceptionV3 | 0.995 | 10082 | 10059 | 23 | 52 | 0.514 | 6 | 2.261 |
| InceptionV3 | 0.99 | 9928 | 9831 | 97 | 280 | 2.769 | 7 | 2.887 |
| InceptionV3 | 0.985 | 9713 | 9545 | 168 | 566 | 5.598 | 7 | 3.369 |

## 6. Seçilen threshold ve gerekçesi

Kod, 0.995'i önceleyen ve toplam örneklerin en fazla %50'sini multi-sample gruplara almayan, en büyük grubu toplamın %5'inden küçük tutan en yüksek-pratik eşiği seçer. Bu otomatik seçim agresif component collapse riskini sınırlar; seçilen eşik aşağıdadır.

- **VGG16:** 0.9999 (groups=9232, multi groups=306, grouped samples=1185)
- **InceptionV3:** 0.995 (groups=10082, multi groups=23, grouped samples=52)

## 7. VGG16 group statistics

See `group_statistics.csv` for all thresholds and `groups/vgg16_groups.npy` for selected labels.

## 8. InceptionV3 group statistics

See `group_statistics.csv` for all thresholds and `groups/inceptionv3_groups.npy` for selected labels.

## 9. Standard CV results

Existing result CSVs were read as Standard 10-Fold CV. Their standard deviations were unavailable and remain blank.

## 10. Group-aware CV results

The new evaluation uses StratifiedGroupKFold with shared groups, shuffle=True, random_state=42, and 5 folds. 10-fold is added only when every class has at least 10 independent groups. See `validation_comparison.csv` and `fold_results.csv.

This run intentionally used the requested 5-fold protocol only (`--no-10fold`).

The following requested model was not completed in this run because its exact fit exceeded the lightweight runtime budget: Gradient Boosting.

## 11. Performance drop analysis

`performance_delta.csv` reports Group-Aware 5-Fold minus Standard 10-Fold. A negative delta is a stricter estimate under this pseudo-group stress-test; it is not proof of a true production drop.

## 12. Confusion matrix evaluation

Matrices are aggregated out-of-fold predictions: each sample is predicted only in its held-out group-aware fold. SVM and the best Group-Aware F1 model are rendered. These are not random 80/20 holdouts.

## 13. Scientific interpretation

Interpret the output as a group-aware estimate versus a standard CV estimate. A large decrease suggests possible optimistic bias from near-duplicate structure; a small decrease suggests strong feature-space separation; no decrease would mean this pseudo-grouping did not materially stress the protocol. None of these outcomes establishes external validity.

## 14. Limitations

"Feature-space pseudo-grouping is a leakage stress-test and does not replace true sample-level or acquisition-session-level grouping."

The grouping threshold is feature-space dependent; approximate similarity may join distinct seeds with similar appearance. No physical IDs, capture sessions, external test set, or raw-image rerun were available. Existing standard CSVs do not contain fold-level standard deviations.

## 15. Future validation recommendation

Preserve stable sample and acquisition-session identifiers during collection, split by those identifiers before training, and evaluate a held-out external acquisition set. Report both group-aware and standard estimates with fold-level predictions and confidence intervals.
