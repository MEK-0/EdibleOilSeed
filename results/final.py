import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def final_report():
    # Mevcut dosyanın konumuna göre yolları belirle
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    results_path = os.path.join(base_dir, 'results')

    vgg_file = os.path.join(results_path, 'vgg16_comparison.csv')
    inc_file = os.path.join(results_path, 'inceptionv3_comparison.csv')

    # Dosya varlık kontrolü
    if not os.path.exists(vgg_file) or not os.path.exists(inc_file):
        print(f"HATA: CSV dosyaları {results_path} içinde bulunamadı!")
        return

    # Verileri oku
    vgg_df = pd.read_csv(vgg_file)
    inc_df = pd.read_csv(inc_file)

    # DL Modeli isimlerini ekle
    vgg_df['DL_Modeli'] = 'VGG16'
    inc_df['DL_Modeli'] = 'InceptionV3'

    # Tüm sonuçları birleştir
    final_df = pd.concat([vgg_df, inc_df], ignore_index=True)

    # --- GÖRSELLEŞTİRME ---
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(1, 2, figsize=(18, 7))

    # 1. Grafik: Accuracy Karşılaştırması
    sns.barplot(x='ML_Modeli', y='Accuracy', hue='DL_Modeli', data=final_df, ax=ax[0])
    ax[0].axhline(0.90, color='red', linestyle='--', label='Hedef %90')
    ax[0].set_title('Hibrit Modellerin Doğruluk (Accuracy) Oranları', fontsize=14)
    ax[0].set_ylim(0.7, 1.0)
    ax[0].legend()

    # 2. Grafik: F1-Score Karşılaştırması (Bilimsel Hassasiyet)
    sns.barplot(x='ML_Modeli', y='F1_Score', hue='DL_Modeli', data=final_df, ax=ax[1])
    ax[1].set_title('Hibrit Modellerin F1-Skor Karşılaştırması', fontsize=14)
    ax[1].set_ylim(0.7, 1.0)

    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'proje_final_analizi.png'))
    plt.show()

    # --- HOCANA SUNUM İÇİN ÖZET ---
    best_row = final_df.loc[final_df['Accuracy'].idxmax()]
    print("\n" + "=" * 40)
    print("      TÜBİTAK 2209-A PROJE SONUCU")
    print("=" * 40)
    print(f"En Başarılı Model: {best_row['DL_Modeli']} + {best_row['ML_Modeli']}")
    print(f"Doğruluk (Accuracy): %{best_row['Accuracy'] * 100:.2f}")
    print(f"F1-Skoru: {best_row['F1_Score']:.4f}")
    print("-" * 40)
    print(f"Sonuçlar {results_path} klasörüne kaydedildi.")


if __name__ == "__main__":
    final_report()