#!/usr/bin/env python3
"""Leakage stress-test using feature-space pseudo-groups and grouped CV.

This is intentionally independent of the original training scripts.  It reads
only the saved feature/label arrays under models/dl_features and writes all
artifacts below results/group_aware.
"""
from __future__ import annotations

import argparse
import csv
import math
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "models" / "dl_features"
OUT_DIR = ROOT / "results" / "group_aware"
GROUP_DIR = OUT_DIR / "groups"
CM_DIR = OUT_DIR / "confusion_matrices"
PLOT_DIR = OUT_DIR / "plots"
THRESHOLDS = (0.9999, 0.999, 0.995, 0.990, 0.985)


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = int(self.parent[x])
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def require_sklearn():
    try:
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
        from sklearn.model_selection import StratifiedGroupKFold
        from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
        from sklearn.neural_network import MLPClassifier
        from sklearn.svm import SVC
    except ImportError as exc:
        raise RuntimeError(
            "This analysis requires numpy and scikit-learn. Install them in the "
            "project environment; no feature extraction or CNN work is required."
        ) from exc
    return locals()


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Zero-norm feature vector found; cosine grouping is undefined.")
    return x / norms


def basic_stats(name: str, x: np.ndarray, y: np.ndarray) -> dict:
    unique_rows = np.unique(x, axis=0).shape[0]
    values = {
        "DL_Model": name,
        "Feature_Shape": str(tuple(x.shape)),
        "Total_Samples": int(x.shape[0]),
        "Feature_Dimension": int(x.shape[1]),
        "Class_Count": int(np.unique(y).size),
        "NaN_Count": int(np.isnan(x).sum()),
        "Inf_Count": int(np.isinf(x).sum()),
        "Exact_Duplicate_Features": int(x.shape[0] - unique_rows),
        "Unique_Features": int(unique_rows),
    }
    print(f"{name}: shape={x.shape}, samples={len(y)}, dimensions={x.shape[1]}, classes={values['Class_Count']}")
    print("  class counts:", ", ".join(f"{c}={n}" for c, n in zip(*np.unique(y, return_counts=True))))
    print("  NaN:", values["NaN_Count"], "Inf:", values["Inf_Count"],
          "exact duplicates:", values["Exact_Duplicate_Features"])
    return values


def pairs_for_class(x: np.ndarray, threshold: float, nn_cls) -> Iterable[tuple[int, int]]:
    """Yield only upper-triangle neighbours, avoiding an NxN similarity matrix."""
    if len(x) < 2:
        return
    nn = nn_cls(metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(x)
    distances, indices = nn.radius_neighbors(x, radius=1.0 - threshold, return_distance=True)
    for i, (ds, js) in enumerate(zip(distances, indices)):
        for d, j in zip(ds, js):
            j = int(j)
            if j > i and float(d) <= 1.0 - threshold + 1e-7:
                yield i, j


def build_groups(x: np.ndarray, y: np.ndarray, threshold: float, nn_cls) -> tuple[np.ndarray, dict]:
    z = normalize_rows(x)
    uf = UnionFind(len(z))
    pair_count = 0
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        for a, b in pairs_for_class(z[indices], threshold, nn_cls):
            uf.union(int(indices[a]), int(indices[b]))
            pair_count += 1
    roots = np.array([uf.find(i) for i in range(len(z))], dtype=np.int64)
    _, groups = np.unique(roots, return_inverse=True)
    sizes = np.bincount(groups)
    multi = sizes[sizes > 1]
    # A grouping bug must never silently permit cross-label groups.
    for group_id in range(len(sizes)):
        if np.unique(y[groups == group_id]).size != 1:
            raise AssertionError(f"Group {group_id} contains multiple labels")
    stats = {
        "Similarity_Threshold": threshold,
        "Total_Samples": len(y),
        "Total_Groups": len(sizes),
        "Singleton_Groups": int(np.sum(sizes == 1)),
        "Multi_Sample_Groups": int(np.sum(sizes > 1)),
        "Samples_In_Multi_Groups": int(multi.sum()) if len(multi) else 0,
        "Grouped_Sample_Percentage": 100.0 * (float(multi.sum()) if len(multi) else 0.0) / len(y),
        "Largest_Group": int(sizes.max()) if len(sizes) else 0,
        "Mean_Multi_Group_Size": float(multi.mean()) if len(multi) else 0.0,
        "Near_Duplicate_Pairs": pair_count,
    }
    return groups.astype(np.int64), stats


def choose_threshold(analyses: list[tuple[float, np.ndarray, dict]]) -> tuple[float, np.ndarray, dict]:
    """Prefer 0.995, but reject thresholds that collapse too much of a class.

    The guard is deliberately conservative: at most half of all samples may be
    in multi-sample groups and no largest component may exceed 5% of the data.
    If no threshold passes, the least aggressive (highest) threshold is used.
    """
    preferred = next((item for item in analyses if math.isclose(item[0], 0.995)), None)
    if preferred is not None:
        _, _, stats = preferred
        if stats["Grouped_Sample_Percentage"] <= 50.0 and stats["Largest_Group"] <= max(50, int(0.05 * stats["Total_Samples"])):
            return preferred
    for item in analyses:
        _, _, stats = item
        if stats["Grouped_Sample_Percentage"] <= 50.0 and stats["Largest_Group"] <= max(50, int(0.05 * stats["Total_Samples"])):
            return item
    return analyses[0]


def classifier_factories(sklearn):
    return {
        "SVM": lambda: sklearn["SVC"](kernel="linear"),
        "Random Forest": lambda: sklearn["RandomForestClassifier"](n_estimators=100, random_state=42, n_jobs=-1),
        "KNN (k=3)": lambda: sklearn["KNeighborsClassifier"](n_neighbors=3, n_jobs=-1),
        "Gradient Boosting": lambda: sklearn["GradientBoostingClassifier"](n_estimators=50, random_state=42),
        "MLP": lambda: sklearn["MLPClassifier"](max_iter=1000, random_state=42),
    }


def grouped_cv(name, x, y, groups, folds, sklearn, factories, write_predictions=False):
    splitter = sklearn["StratifiedGroupKFold"](n_splits=folds, shuffle=True, random_state=42)
    rows, predictions = [], {}
    for model_name, make_model in factories.items():
        pred = np.empty(len(y), dtype=y.dtype)
        for fold, (train, test) in enumerate(splitter.split(x, y, groups), 1):
            model = make_model()
            model.fit(x[train], y[train])
            fold_pred = model.predict(x[test])
            pred[test] = fold_pred
            rows.append({
                "DL_Model": name, "ML_Model": model_name, "Validation_Type": f"Group-Aware {folds}-Fold CV", "Fold": fold,
                "Accuracy": sklearn["accuracy_score"](y[test], fold_pred),
                "Precision": sklearn["precision_score"](y[test], fold_pred, average="macro", zero_division=0),
                "Recall": sklearn["recall_score"](y[test], fold_pred, average="macro", zero_division=0),
                "F1": sklearn["f1_score"](y[test], fold_pred, average="macro", zero_division=0),
            })
        predictions[model_name] = pred
    return rows, predictions


def aggregate(rows):
    out = []
    keys = {(r["DL_Model"], r["ML_Model"], r["Validation_Type"]) for r in rows}
    for dl, ml, validation in sorted(keys):
        rs = [r for r in rows if (r["DL_Model"], r["ML_Model"], r["Validation_Type"]) == (dl, ml, validation)]
        out.append({"DL_Model": dl, "ML_Model": ml, "Validation_Type": validation, **{
            f"{metric}_Mean": float(np.mean([r[metric] for r in rs])) for metric in ("Accuracy", "Precision", "Recall", "F1")
        }, **{f"{metric}_Std": float(np.std([r[metric] for r in rs], ddof=1)) for metric in ("Accuracy", "Precision", "Recall", "F1")}})
    return out


def write_csv(path: Path, rows: list[dict], fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)


def load_standard():
    rows = []
    for dl, path in (("VGG16", ROOT / "results" / "vgg16_comparison.csv"), ("InceptionV3", ROOT / "results" / "inceptionv3_comparison.csv")):
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                ml = r.get("ML_Modeli", r.get("ML_Model", "")).replace("Yapay Sinir Aglari (YSA)", "MLP")
                rows.append({"DL_Model": dl, "ML_Model": ml, "Validation_Method": "Standard 10-Fold CV",
                             "Accuracy": float(r["Accuracy"]), "Precision": float(r.get("Precision", "nan")),
                             "Recall": float(r.get("Recall", "nan")), "F1": float(r["F1_Score"]),
                             "Accuracy_Std": "", "Precision_Std": "", "Recall_Std": "", "F1_Std": ""})
    return rows


def write_confusion(path: Path, y_true, y_pred, labels, title, sklearn):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        warnings.warn("Pillow unavailable; confusion matrices were not rendered")
        return
    cm = sklearn["confusion_matrix"](y_true, y_pred, labels=labels)
    size, margin = 760, 150; cell = (size - margin - 30) // len(labels)
    image = Image.new("RGB", (size, size), "white"); draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(); max_value = max(1, int(cm.max()))
    draw.text((20, 18), title, fill="black", font=font)
    for i, label in enumerate(labels):
        draw.text((margin + i * cell + 3, margin - 20), str(label)[:12], fill="black", font=font)
        draw.text((20, margin + i * cell + cell // 2), str(label)[:12], fill="black", font=font)
        for j in range(len(labels)):
            value = int(cm[i, j]); shade = 255 - int(180 * value / max_value)
            x0, y0 = margin + j * cell, margin + i * cell
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=(shade, shade + 10 if shade < 245 else 255, 255), outline="black")
            draw.text((x0 + cell // 2 - 8, y0 + cell // 2 - 5), str(value), fill="black", font=font)
    draw.text((size // 2 - 30, size - 24), "Predicted", fill="black", font=font)
    image.save(path)


def write_plots(comparison, aggregate_rows):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        warnings.warn("Pillow unavailable; summary plots were not rendered")
        return
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    labels = [f"{r['DL_Model']}\n{r['ML_Model']}" for r in aggregate_rows]
    def chart(filename, title, series, errors=None):
        image = Image.new("RGB", (1400, 620), "white"); draw = ImageDraw.Draw(image); font = ImageFont.load_default()
        draw.text((20, 18), title, fill="black", font=font); left, bottom, top, width = 70, 540, 55, 1260
        draw.line((left, top, left, bottom), fill="black"); draw.line((left, bottom, left + width, bottom), fill="black")
        for i, label in enumerate(labels):
            x = left + (i + .5) * width / len(labels); value = series[i]; y = bottom - value * (bottom - top)
            draw.rectangle((x - 20, y, x + 20, bottom), fill=(50, 100, 170), outline="black")
            if errors: draw.line((x, bottom - (value + errors[i]) * (bottom - top), x, bottom - (value - errors[i]) * (bottom - top)), fill="black", width=2)
            draw.text((x - 28, bottom + 8), label.replace("\n", " ")[:18], fill="black", font=font)
        image.save(PLOT_DIR / filename)
    for metric, filename, title in (("Accuracy", "groupaware_accuracy_mean_std.png", "Group-Aware Accuracy mean +/- SD"), ("F1", "groupaware_f1_mean_std.png", "Group-Aware F1 mean +/- SD")):
        chart(filename, title, [r[f"{metric}_Mean"] for r in aggregate_rows], [r[f"{metric}_Std"] for r in aggregate_rows])
    def compare_chart(filename, title, standard, grouped):
        image = Image.new("RGB", (1400, 620), "white"); draw = ImageDraw.Draw(image); font = ImageFont.load_default()
        draw.text((20, 18), title, fill="black", font=font); left, bottom, top, width = 70, 540, 55, 1260; draw.line((left, top, left, bottom), fill="black"); draw.line((left, bottom, left + width, bottom), fill="black")
        for i, label in enumerate(labels):
            center = left + (i + .5) * width / len(labels); step = max(6, width / len(labels) / 4)
            for x, value, color in ((center - step, standard[i], (130, 130, 130)), (center + step, grouped[i], (50, 100, 170))):
                y = bottom - value * (bottom - top); draw.rectangle((x - step / 2, y, x + step / 2, bottom), fill=color, outline="black")
            draw.text((center - 28, bottom + 8), label.replace("\n", " ")[:18], fill="black", font=font)
        draw.rectangle((left + width - 240, top + 10, left + width - 225, top + 25), fill=(130, 130, 130)); draw.text((left + width - 215, top + 10), "Standard", fill="black", font=font)
        draw.rectangle((left + width - 240, top + 30, left + width - 225, top + 45), fill=(50, 100, 170)); draw.text((left + width - 215, top + 30), "Group-aware", fill="black", font=font)
        image.save(PLOT_DIR / filename)
    for metric, filename, title in (("Accuracy", "standard_vs_groupaware_accuracy.png", "Standard CV vs Group-Aware Accuracy"), ("F1", "standard_vs_groupaware_f1.png", "Standard CV vs Group-Aware F1")):
        standard = [next((r[metric] for r in comparison if r["DL_Model"] == a["DL_Model"] and r["ML_Model"] == a["ML_Model"] and r["Validation_Method"] == "Standard 10-Fold CV"), np.nan) for a in aggregate_rows]
        grouped = [r[f"{metric}_Mean"] for r in aggregate_rows]
        compare_chart(filename, title, standard, grouped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-only", action="store_true", help="stop after grouping/statistics; do not fit classifiers")
    parser.add_argument("--skip-gradient-boosting", action="store_true", help="skip the very slow exact GradientBoostingClassifier run")
    parser.add_argument("--no-10fold", action="store_true", help="run the required 5-fold analysis only")
    args = parser.parse_args()
    sklearn = require_sklearn()
    OUT_DIR.mkdir(parents=True, exist_ok=True); GROUP_DIR.mkdir(exist_ok=True); CM_DIR.mkdir(exist_ok=True)
    datasets = {}
    basic = []
    threshold_rows = []
    selected = {}
    for name, stem in (("VGG16", "vgg16"), ("InceptionV3", "inceptionv3")):
        x = np.load(FEATURE_DIR / f"{stem}_features.npy")
        y = np.load(FEATURE_DIR / f"{stem}_labels.npy")
        if x.ndim != 2 or len(x) != len(y): raise ValueError(f"Invalid {name} feature/label shapes: {x.shape}, {y.shape}")
        basic.append(basic_stats(name, x, y)); datasets[name] = (x, y)
        analyses = []
        for threshold in THRESHOLDS:
            groups, stats = build_groups(x, y, threshold, sklearn["NearestNeighbors"])
            stats["DL_Model"] = name; threshold_rows.append(stats); analyses.append((threshold, groups, stats))
        selected[name] = choose_threshold(analyses)
        threshold, groups, stats = selected[name]
        np.save(GROUP_DIR / f"{stem}_groups.npy", groups)
        print(f"{name}: selected threshold={threshold}, groups={stats['Total_Groups']}, multi-groups={stats['Multi_Sample_Groups']}, grouped samples={stats['Samples_In_Multi_Groups']}")
    write_csv(OUT_DIR / "feature_statistics.csv", basic)
    write_csv(OUT_DIR / "group_statistics.csv", threshold_rows)
    if args.analysis_only:
        return
    factories = classifier_factories(sklearn)
    skipped_models = []
    if args.skip_gradient_boosting:
        factories.pop("Gradient Boosting")
        skipped_models.append("Gradient Boosting")
    fold_rows = []; aggregate_rows = []; predictions = {}
    for name, (x, y) in datasets.items():
        groups = selected[name][1]
        rows5, pred5 = grouped_cv(name, x, y, groups, 5, sklearn, factories); fold_rows.extend(rows5); predictions[(name, 5)] = pred5
        aggregate_rows.extend(aggregate(rows5))
        min_groups = min(np.unique(groups[y == c]).size for c in np.unique(y))
        if min_groups >= 10 and not args.no_10fold:
            rows10, pred10 = grouped_cv(name, x, y, groups, 10, sklearn, factories); fold_rows.extend(rows10); aggregate_rows.extend(aggregate(rows10)); predictions[(name, 10)] = pred10
        else:
            warnings.warn(f"{name}: only {min_groups} groups in the least represented class; 10-fold skipped")
    write_csv(OUT_DIR / "fold_results.csv", fold_rows)
    comparison = load_standard()
    for r in aggregate_rows:
        comparison.append({"DL_Model": r["DL_Model"], "ML_Model": r["ML_Model"], "Validation_Method": r["Validation_Type"], "Accuracy": r["Accuracy_Mean"], "Precision": r["Precision_Mean"], "Recall": r["Recall_Mean"], "F1": r["F1_Mean"], "Accuracy_Std": r["Accuracy_Std"], "Precision_Std": r["Precision_Std"], "Recall_Std": r["Recall_Std"], "F1_Std": r["F1_Std"]})
    write_csv(OUT_DIR / "validation_comparison.csv", comparison, ["DL_Model","ML_Model","Validation_Method","Accuracy","Precision","Recall","F1","Accuracy_Std","Precision_Std","Recall_Std","F1_Std"])
    deltas = []
    for dl in datasets:
        for ml in factories:
            std = next((r for r in comparison if r["DL_Model"] == dl and r["ML_Model"] == ml and r["Validation_Method"] == "Standard 10-Fold CV"), None)
            ga = next((r for r in comparison if r["DL_Model"] == dl and r["ML_Model"] == ml and r["Validation_Method"] == "Group-Aware 5-Fold CV"), None)
            if std and ga:
                deltas.append({"DL_Model": dl, "ML_Model": ml, "Standard_Accuracy": std["Accuracy"], "GroupAware_Accuracy": ga["Accuracy"], "Accuracy_Delta": ga["Accuracy"]-std["Accuracy"], "Accuracy_Delta_Percentage_Points": 100*(ga["Accuracy"]-std["Accuracy"]), "Standard_F1": std["F1"], "GroupAware_F1": ga["F1"], "F1_Delta": ga["F1"]-std["F1"], "F1_Delta_Percentage_Points": 100*(ga["F1"]-std["F1"])})
    write_csv(OUT_DIR / "performance_delta.csv", deltas)
    for name, (x, y) in datasets.items():
        rows = [r for r in aggregate_rows if r["DL_Model"] == name and r["Validation_Type"] == "Group-Aware 5-Fold CV"]
        best = max(rows, key=lambda r: r["F1_Mean"]); labels = np.unique(y)
        write_confusion(CM_DIR / f"cm_{name.lower()}_{best['ML_Model'].lower().replace(' ', '_').replace('(', '').replace(')', '')}_groupaware.png", y, predictions[(name, 5)][best["ML_Model"]], labels, f"{name} + {best['ML_Model']} Group-Aware OOF", sklearn)
        write_confusion(CM_DIR / f"cm_{name.lower()}_svm_groupaware.png", y, predictions[(name, 5)]["SVM"], labels, f"{name} + SVM Group-Aware OOF", sklearn)
    write_plots(comparison, [r for r in aggregate_rows if r["Validation_Type"] == "Group-Aware 5-Fold CV"])
    report = OUT_DIR / "GROUP_AWARE_VALIDATION_REPORT.md"
    report.write_text(make_report(threshold_rows, selected, basic, comparison, deltas, skipped_models, args.no_10fold), encoding="utf-8")
    print_summary(selected, comparison, deltas)


def make_report(threshold_rows, selected, basic, comparison, deltas, skipped_models=None, no_10fold=False):
    lines = ["# Group-Aware Validation Report", "", "## 1. Amaç", "", "Bu analiz, mevcut VGG16 ve InceptionV3 feature vektörleri üzerinde near-duplicate leakage riskini inceleyen bağımsız bir validation stress-testidir. CNN yeniden eğitilmemiş, feature extraction tekrarlanmamıştır.", "", "## 2. Neden standart image-level CV iyimser olabilir?", "", "Aynı fiziksel tohumun veya aynı capture oturumunun çok benzer görüntüleri train ve test fold'larına ayrılırsa image-level CV bağımsız genelleme performansını iyimser gösterebilir.", "", "## 3. Mevcut dataset kısıtı", "", "Gerçek physical-seed ID ve capture-session ID bulunmadığı için örnekler feature-space pseudo-group'ları ile gruplanmıştır. Bu sonuçlar gerçek sample/session grouping yerine geçmez.", "", "## 4. Pseudo-group yöntemi", "", "Feature vektörleri L2 normalize edildi. Her sınıf kendi içinde cosine distance kullanan brute-force NearestNeighbors ile tarandı; 0.9999, 0.999, 0.995, 0.990 ve 0.985 similarity eşikleri test edildi. Eşik grafı Union-Find connected components ile kuruldu. Sınıflar arası örnekler aynı gruba alınmadı ve her grup için label purity assert edildi.", "", "## 5. Similarity threshold analizi", "", "| DL model | Threshold | Groups | Singleton | Multi groups | Grouped samples | Grouped % | Largest | Mean multi size |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in threshold_rows: lines.append(f"| {r['DL_Model']} | {r['Similarity_Threshold']} | {r['Total_Groups']} | {r['Singleton_Groups']} | {r['Multi_Sample_Groups']} | {r['Samples_In_Multi_Groups']} | {r['Grouped_Sample_Percentage']:.3f} | {r['Largest_Group']} | {r['Mean_Multi_Group_Size']:.3f} |")
    lines += ["", "## 6. Seçilen threshold ve gerekçesi", "", "Kod, 0.995'i önceleyen ve toplam örneklerin en fazla %50'sini multi-sample gruplara almayan, en büyük grubu toplamın %5'inden küçük tutan en yüksek-pratik eşiği seçer. Bu otomatik seçim agresif component collapse riskini sınırlar; seçilen eşik aşağıdadır.", ""]
    for dl, item in selected.items(): lines.append(f"- **{dl}:** {item[0]} (groups={item[2]['Total_Groups']}, multi groups={item[2]['Multi_Sample_Groups']}, grouped samples={item[2]['Samples_In_Multi_Groups']})")
    lines += ["", "## 7. VGG16 group statistics", "", "See `group_statistics.csv` for all thresholds and `groups/vgg16_groups.npy` for selected labels.", "", "## 8. InceptionV3 group statistics", "", "See `group_statistics.csv` for all thresholds and `groups/inceptionv3_groups.npy` for selected labels.", "", "## 9. Standard CV results", "", "Existing result CSVs were read as Standard 10-Fold CV. Their standard deviations were unavailable and remain blank.", "", "## 10. Group-aware CV results", "", "The new evaluation uses StratifiedGroupKFold with shared groups, shuffle=True, random_state=42, and 5 folds. 10-fold is added only when every class has at least 10 independent groups. See `validation_comparison.csv` and `fold_results.csv."]
    if no_10fold:
        lines += ["", "This run intentionally used the requested 5-fold protocol only (`--no-10fold`)."]
    if skipped_models:
        lines += ["", "The following requested model was not completed in this run because its exact fit exceeded the lightweight runtime budget: " + ", ".join(skipped_models) + "."]
    lines += ["", "## 11. Performance drop analysis", "", "`performance_delta.csv` reports Group-Aware 5-Fold minus Standard 10-Fold. A negative delta is a stricter estimate under this pseudo-group stress-test; it is not proof of a true production drop.", "", "## 12. Confusion matrix evaluation", "", "Matrices are aggregated out-of-fold predictions: each sample is predicted only in its held-out group-aware fold. SVM and the best Group-Aware F1 model are rendered. These are not random 80/20 holdouts.", "", "## 13. Scientific interpretation", "", "Interpret the output as a group-aware estimate versus a standard CV estimate. A large decrease suggests possible optimistic bias from near-duplicate structure; a small decrease suggests strong feature-space separation; no decrease would mean this pseudo-grouping did not materially stress the protocol. None of these outcomes establishes external validity.", "", "## 14. Limitations", "", '"Feature-space pseudo-grouping is a leakage stress-test and does not replace true sample-level or acquisition-session-level grouping."', "", "The grouping threshold is feature-space dependent; approximate similarity may join distinct seeds with similar appearance. No physical IDs, capture sessions, external test set, or raw-image rerun were available. Existing standard CSVs do not contain fold-level standard deviations.", "", "## 15. Future validation recommendation", "", "Preserve stable sample and acquisition-session identifiers during collection, split by those identifiers before training, and evaluate a held-out external acquisition set. Report both group-aware and standard estimates with fold-level predictions and confidence intervals.", ""]
    return "\n".join(lines)


def print_summary(selected, comparison, deltas):
    print("\n------------------------------------\nGROUP-AWARE VALIDATION SUMMARY\n------------------------------------")
    for dl in selected:
        ds = [r for r in deltas if r["DL_Model"] == dl]
        best = max((r for r in comparison if r["DL_Model"] == dl and r["Validation_Method"] == "Group-Aware 5-Fold CV"), key=lambda r: r["F1"])
        std = next((r for r in comparison if r["DL_Model"] == dl and r["ML_Model"] == best["ML_Model"] and r["Validation_Method"] == "Standard 10-Fold CV"), None)
        delta = next((r for r in ds if r["ML_Model"] == best["ML_Model"]), None)
        delta_text = f"{delta['Accuracy_Delta']:.6f}" if delta else "n/a"
        print(f"\n{dl}:\nBest model: {best['ML_Model']}\nStandard CV Accuracy: {std['Accuracy'] if std else 'n/a'}\nGroup-Aware Accuracy: {best['Accuracy']:.6f}\nDelta: {delta_text}\nGroup-Aware F1: {best['F1']:.6f}")
    print("\nSelected similarity thresholds:", ", ".join(f"{k}={v[0]}" for k, v in selected.items()))
    print("Total multi-sample groups:", ", ".join(f"{k}={v[2]['Multi_Sample_Groups']}" for k, v in selected.items()))
    print("Samples included in duplicate groups:", ", ".join(f"{k}={v[2]['Samples_In_Multi_Groups']}" for k, v in selected.items()))


if __name__ == "__main__":
    main()
