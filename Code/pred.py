import joblib
import numpy as np
import cv2
from skimage.feature import hog
from rembg import remove
from PIL import Image
import matplotlib.pyplot as plt

def predict_helmet(image_path, model_path='helmet_detection_gb_model.pkl', scaler_path='scaler.pkl'):
    # Memuat model dan scaler yang sudah dilatih
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Memuat dan memproses gambar
    img = Image.open(image_path)
    img = img.resize((128, 128))  
    img_no_bg = remove(img) 
    img_no_bg = np.array(img_no_bg)

    # Mengubah gambar ke grayscale
    gray = cv2.cvtColor(img_no_bg, cv2.COLOR_BGR2GRAY)

    # Ekstraksi fitur HOG
    features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys',
                   visualize=False)

    # Melakukan scaling pada fitur
    features = scaler.transform([features])

    # Memprediksi menggunakan model yang sudah dilatih
    prediction = model.predict(features)
    
    # Mengubah gambar kembali ke RGB untuk ditampilkan
    img_rgb = cv2.cvtColor(img_no_bg, cv2.COLOR_BGR2RGB)

    # Menampilkan gambar 
    plt.imshow(img_rgb)
    plt.title("Pakai Helm" if prediction == 1 else "Tidak Pakai Helm")
    plt.axis('off')
    plt.show()

    # Mengembalikan hasil prediksi
    return "Pakai Helm" if prediction == 1 else "Tidak Pakai Helm"

# Contoh penggunaan fungsi
image_path ='./helm-137.jpg'
result = predict_helmet(image_path)
print(f'Hasil prediksi: {result}')
