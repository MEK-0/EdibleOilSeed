import cv2
import os
import numpy as np
from tqdm import tqdm


class Preprocessor:
    def __init__(self, raw_path, processed_path, img_size=(224, 224)):
        
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.img_size = img_size

        if not os.path.exists(self.processed_path):
            os.makedirs(self.processed_path)

    def process_images(self):
        
        categories = [f for f in os.listdir(self.raw_path) if os.path.isdir(os.path.join(self.raw_path, f))]
        print(f"Bulunan Tohum Türleri: {categories}")

        for category in categories:
            class_path = os.path.join(self.raw_path, category)
            save_path = os.path.join(self.processed_path, category)

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            images = os.listdir(class_path)
            
            for img_name in tqdm(images, desc=f"İşleniyor: {category}"):
                try:
                    img_path = os.path.join(class_path, img_name)
                    img = cv2.imread(img_path)

                    if img is None: continue  # Bozuk dosya kontrolü

                    img = cv2.resize(img, self.img_size) 
                    img = cv2.GaussianBlur(img, (3, 3), 0)  
                    img = img.astype('float32') / 255.0  

                    np_name = os.path.splitext(img_name)[0] + ".npy"
                    np.save(os.path.join(save_path, np_name), img)

                except Exception as e:
                    print(f"Hata: {img_name} -> {e}")


# --- ÇALIŞTIRMA KOMUTLARI ---
if __name__ == "__main__":
    
    
    raw_veri_yolu = 'data/raw'
    islenmis_veri_yolu = 'data/processed'

    preprocessor = Preprocessor(raw_path=raw_veri_yolu, processed_path=islenmis_veri_yolu)
    preprocessor.process_images()
