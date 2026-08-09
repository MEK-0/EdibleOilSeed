import numpy as np
import os
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


class IncTrainer:
    def __init__(self, features_path, results_path):
        self.features_path = features_path
        self.results_path = results_path

    def run_inc_only(self):
        # 1. Sadece InceptionV3 özelliklerini yükle
        X = np.load(os.path.join(self.features_path, 'inceptionv3_features.npy'))
        y = np.load(os.path.join(self.features_path, 'inceptionv3_labels.npy'))

        # 2. Modeller (Proje önerindeki liste) [cite: 173, 182]
        models = {
            'SVM': SVC(kernel='linear'),
            'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1),
            'KNN (k=3)': KNeighborsClassifier(n_neighbors=3, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=50),  # Süre için ağaç sayısını 50 yaptık
            'Yapay Sinir Aglari (YSA)': MLPClassifier(max_iter=500)
        }

        results = []
        print(f"\n--- INCEPTIONV3 Hibrit Analizi Başlatıldı (Veri Boyutu: {X.shape}) ---")

        for name, model in models.items():
            print(f"{name} eğitiliyor ve 10-Katlı test ediliyor...")

            scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
            # n_jobs=-1 ile tüm çekirdekleri zorluyoruz
            cv_results = cross_validate(model, X, y, cv=10, scoring=scoring, n_jobs=-1)

            results.append({
                'ML_Modeli': name,
                'Accuracy': np.mean(cv_results['test_accuracy']),
                'F1_Score': np.mean(cv_results['test_f1_macro']),
                'Recall': np.mean(cv_results['test_recall_macro']),
                'Precision': np.mean(cv_results['test_precision_macro'])
            })

        # 3. Kaydet
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(self.results_path, 'inceptionv3_comparison.csv'), index=False)
        print("\nİşlem Tamamlandı! InceptionV3 sonuçları 'results' klasörüne kaydedildi.")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)

    trainer = IncTrainer(
        features_path=os.path.join(base_dir, 'models/dl_features'),
        results_path=os.path.join(base_dir, 'results')
    )
    trainer.run_inc_only()