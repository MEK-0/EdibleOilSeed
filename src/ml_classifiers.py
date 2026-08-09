import numpy as np
import os
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


class MLTrainer:
    def __init__(self, features_path, results_path):
        self.features_path = features_path
        self.results_path = results_path

        # Sonuç klasörünü oluştur
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

    def train_and_evaluate(self, model_name_dl):
        # 1. Özellikleri ve etiketleri yükle
        feat_file = os.path.join(self.features_path, f'{model_name_dl}_features.npy')
        label_file = os.path.join(self.features_path, f'{model_name_dl}_labels.npy')

        if not os.path.exists(feat_file):
            print(f"Hata: {feat_file} bulunamadı!")
            return None

        X = np.load(feat_file)
        y = np.load(label_file)

        # 2. ML Modellerini tanımla (Proje önerindeki liste [cite: 24])
        models = {
            'SVM': SVC(kernel='linear'),  # [cite: 108, 134]
            'Random Forest': RandomForestClassifier(n_estimators=100),  # [cite: 115, 124]
            'KNN (k=3)': KNeighborsClassifier(n_neighbors=3),  # Proje önerisindeki spesifik k değeri [cite: 140, 368]
            'Gradient Boosting': GradientBoostingClassifier(),  # [cite: 388]
            'Yapay Sinir Aglari (YSA)': MLPClassifier(max_iter=1000)  # [cite: 399]
        }

        results = []

        print(f"\n--- {model_name_dl.upper()} Özellikleri ile Hibrit Eğitim Başlıyor ---")

        for name, model in models.items():
            print(f"{name} modeli 10-Katli Çapraz Doğrulama ile test ediliyor...")

            # 3. 10-Katlı Çapraz Doğrulama ve Metrikler [cite: 413, 449]
            scoring = {
                'accuracy': 'accuracy',
                'precision': 'precision_macro',
                'recall': 'recall_macro',
                'f1': 'f1_macro'
            }

            cv_results = cross_validate(model, X, y, cv=10, scoring=scoring, n_jobs=-1)

            # Ortalama sonuçları kaydet
            results.append({
                'DL_Modeli': model_name_dl,
                'ML_Modeli': name,
                'Accuracy': np.mean(cv_results['test_accuracy']),
                'Precision': np.mean(cv_results['test_precision']),
                'Recall': np.mean(cv_results['test_recall']),
                'F1_Score': np.mean(cv_results['test_f1'])
            })

        # 4. Sonuçları CSV olarak kaydet [cite: 502]
        df = pd.DataFrame(results)
        output_file = os.path.join(self.results_path, f'{model_name_dl}_comparison.csv')
        df.to_csv(output_file, index=False)
        print(f"Bitti! {model_name_dl} sonuçları kaydedildi: {output_file}")
        return df


if __name__ == "__main__":
    # Klasör yollarını ayarla
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)

    trainer = MLTrainer(
        features_path=os.path.join(base_dir, 'models/dl_features'),
        results_path=os.path.join(base_dir, 'results')
    )

    # Her iki DL yöntemi için ML sonuçlarını hesapla
    for dl in ['vgg16', 'inceptionv3']:
        trainer.train_and_evaluate(dl)
