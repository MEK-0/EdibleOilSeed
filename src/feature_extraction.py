import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tqdm import tqdm
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class FeatureExtractor:
    def __init__(self, processed_path, features_path):
        self.processed_path = processed_path
        self.features_path = features_path

        if not os.path.exists(self.features_path):
            os.makedirs(self.features_path)

    def get_model(self, model_name):
        # Proje önerisindeki 224x224 giriş boyutu [cite: 273]
        base_model = None
        if model_name == 'vgg16':
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        elif model_name == 'inceptionv3':
            base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        # SqueezeNet için özel bir yükleme veya kütüphane gerekebilir, şimdilik bu ikisiyle başlayalım.

        # Global Average Pooling ekleyerek boyutu sabitliyoruz
        model = Model(inputs=base_model.input, outputs=tf.keras.layers.GlobalAveragePooling2D()(base_model.output))
        return model

    def extract_and_save(self, model_name):
        print(f"\n{model_name.upper()} ile özellik çıkarımı başlatılıyor...")
        model = self.get_model(model_name)

        categories = os.listdir(self.processed_path)
        features_list = []
        labels_list = []

        for idx, category in enumerate(categories):
            class_path = os.path.join(self.processed_path, category)
            for file_name in tqdm(os.listdir(class_path), desc=f"Sınıf: {category}"):
                img_data = np.load(os.path.join(class_path, file_name))
                img_data = np.expand_dims(img_data, axis=0)  # Batch boyutu ekle

                # Derin özellik çıkarımı [cite: 133, 136]
                feature = model.predict(img_data, verbose=0)
                features_list.append(feature.flatten())
                labels_list.append(category)

        # Sonuçları kaydet (ML modelleri için girdi olacak)
        np.save(os.path.join(self.features_path, f'{model_name}_features.npy'), np.array(features_list))
        np.save(os.path.join(self.features_path, f'{model_name}_labels.npy'), np.array(labels_list))
        print(f"{model_name} özellikleri başarıyla kaydedildi.")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)

    extractor = FeatureExtractor(
        processed_path=os.path.join(base_dir, 'data/processed'),
        features_path=os.path.join(base_dir, 'models/dl_features')
    )

    # Sırasıyla modellerden özellikleri çıkaralım
    for m in ['vgg16', 'inceptionv3']:
        extractor.extract_and_save(m)