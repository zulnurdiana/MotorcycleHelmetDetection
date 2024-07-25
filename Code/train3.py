import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import imgaug.augmenters as iaa
from rembg import remove
from PIL import Image
from scipy.stats import uniform

# Fungsi untuk membaca gambar, mengubah ukurannya, dan menghapus latar belakang
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path)
            img = img.resize((50, 50))  
            img_no_bg = remove(img)  
            img_no_bg = np.array(img_no_bg)
            if img_no_bg is not None:
                images.append(img_no_bg)
                labels.append(label)  
    return images, labels

# Fungsi untuk melakukan segmentasi gambar menggunakan Canny Edge Detection
def segment_image(image):
    edges = cv2.Canny(image, 100, 200)
    return edges

# Fungsi untuk ekstraksi fitur HOG
def extract_hog_features(images):
    hog_features = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        segmented_img = segment_image(gray)  # Segmentasi gambar
        features = hog(segmented_img, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys',
                       visualize=False)
        hog_features.append(features)
    return hog_features

# Augmentasi gambar untuk setiap gambar pada dataset
def augment_images(images, labels, augmentation_factor=5):
    seq = iaa.Sequential([
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # Skala
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # Translasi
            rotate=(-45, 45)  # Rotasi
        ),
        iaa.Fliplr(0.5),  # Flip gambar secara horizontal
        iaa.Crop(percent=(0, 0.1)),  # Crop gambar secara acak
        iaa.GaussianBlur(sigma=(0, 1.0))  # Menambahkan blur gaussian
    ])

    augmented_images = []
    augmented_labels = []

    for img, label in zip(images, labels):
        for _ in range(augmentation_factor):
            augmented_img = seq(image=img)
            augmented_images.append(augmented_img)
            augmented_labels.append(label)
    
    return augmented_images, augmented_labels

# Memuat dataset
images_with_helm, labels_with_helm = load_images_from_folder('dataset/Helm', 1)  # Folder dengan helm
images_without_helm, labels_without_helm = load_images_from_folder('dataset/Tanpa', 0)  # Folder tanpa helm

# Augmentasi data
augmented_images_with_helm, augmented_labels_with_helm = augment_images(images_with_helm, labels_with_helm, augmentation_factor=5)
augmented_images_without_helm, augmented_labels_without_helm = augment_images(images_without_helm, labels_without_helm, augmentation_factor=5)

# Gabungkan data asli dan data augmentasi
images = images_with_helm + images_without_helm + augmented_images_with_helm + augmented_images_without_helm
labels = labels_with_helm + labels_without_helm + augmented_labels_with_helm + augmented_labels_without_helm

# Ekstraksi fitur HOG
hog_features = extract_hog_features(images)

# Konversi list ke numpy array
X = np.array(hog_features)
y = np.array(labels)

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Membagi data menjadi data pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter Tuning dengan SVC linear menggunakan RandomizedSearchCV
param_dist = {'C': uniform(0.1, 100)}
random_search = RandomizedSearchCV(SVC(kernel='linear'), param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
random_search.fit(X_train, y_train)

print("Parameter terbaik untuk SVC:", random_search.best_params_)

# Melatih model dengan parameter terbaik
svc_model = random_search.best_estimator_
svc_model.fit(X_train, y_train)

# Memprediksi pada set pengujian
y_pred_svc = svc_model.predict(X_test)

# Evaluasi model SVC
print("Akurasi SVC:", accuracy_score(y_test, y_pred_svc))
print(classification_report(y_test, y_pred_svc))

# Menyimpan model dan scaler ke direktori root
joblib.dump(svc_model, 'svc_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
