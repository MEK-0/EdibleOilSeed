import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def detailed_visualization():
    # Klasör yollarını belirle (Hata payını sıfırlamak için dosya konumundan türetildi)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    results_path = os.path.join(base_dir, 'results')
    cm_path = os.path.join(results_path, 'confusion_matrices')
    plots_path = os.path.join(results_path, 'plots')
    features_path = os.path.join(base_dir, 'models', 'dl_features')

    # Klasörler yoksa oluştur
    for p in [cm_path, plots_path]:
        if not os.path.exists(p): os.makedirs(p)

    # 1. GENEL PERFORMANS KARŞILAŞTIRMA PLOTLARI
    vgg_file = os.path.join(results_path, 'vgg16_comparison.csv')
    inc_file = os.path.join(results_path, 'inceptionv3_comparison.csv')

    if os.path.exists(vgg_file) and os.path.exists(inc_file):
        vgg_df = pd.read_csv(vgg_file)
        inc_df = pd.read_csv(inc_file)
        vgg_df['DL_Modeli'] = 'VGG16'
        inc_df['DL_Modeli'] = 'InceptionV3'
        final_df = pd.concat([vgg_df, inc_df], ignore_index=True)

        # Accuracy, F1_Score vb. metrikler için grafikler
        # CSV sütun isimlerinin tam eşleştiğinden emin olun (Accuracy, F1_Score vb.)
        metrics = ['Accuracy', 'F1_Score', 'Recall', 'Precision']
        for metric in metrics:
            if metric in final_df.columns:
                plt.figure(figsize=(12, 6))
                sns.barplot(x='ML_Modeli', y=metric, hue='DL_Modeli', data=final_df)
                plt.title(f'Modeller Arası {metric} Karşılaştırması', fontsize=15)
                plt.ylim(0.7, 1.0)
                plt.axhline(0.90, color='red', linestyle='--', label='Hedef %90')
                plt.legend(loc='lower right')
                plt.savefig(os.path.join(plots_path, f'comparison_{metric.lower()}.png'))
                plt.close()

    # 2. HATA MATRİSİ (CONFUSION MATRIX) ÇİZİMİ
    dl_models = ['vgg16', 'inceptionv3']
    for dl in dl_models:
        feat_path = os.path.join(features_path, f'{dl}_features.npy')
        lab_path = os.path.join(features_path, f'{dl}_labels.npy')

        if os.path.exists(feat_path) and os.path.exists(lab_path):
            X = np.load(feat_path)
            y = np.load(lab_path)
            classes = np.unique(y)

            # HATANIN DÜZELTİLDİĞİ YER: test_test_split -> test_size
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Sunum için en stabil model olan Linear SVM kullanıyoruz
            clf = SVC(kernel='linear')
            print(f"{dl.upper()} için Hata Matrisi hesaplanıyor...")
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=classes, yticklabels=classes)
            plt.title(f'Hata Matrisi: {dl.upper()} + SVM (Hibrit)', fontsize=14)
            plt.ylabel('Gerçek Sınıf')
            plt.xlabel('Tahmin Edilen Sınıf')
            plt.tight_layout()
            plt.savefig(os.path.join(cm_path, f'cm_{dl}_svm.png'))
            plt.close()

    print(f"İşlem tamam! Grafikler ve matrisler kaydedildi: \n1. {plots_path} \n2. {cm_path}")


if __name__ == "__main__":
    detailed_visualization()