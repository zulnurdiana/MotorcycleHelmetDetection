import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.feature import hog
from skimage import exposure

# Fungsi untuk membaca gambar dari folder
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                # img = cv2.resize(img, (100, 100)) 
                images.append(img)  
                labels.append(label)  
    return images, labels

# segmentasi motor dan helm dari gambar menggunakan deteksi tepi Canny
def segment_motor_helm(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    edges = cv2.Canny(gray, 100, 200)  
    return edges

# ekstraksi fitur HOG dan visualisasinya
def extract_hog_features_and_visualize(images):
    hog_features = [] 
    hog_images = []  
    segmented_images = []  
    for img in images:
        segmented_img = segment_motor_helm(img)  
        segmented_images.append(segmented_img)  
        features, hog_image = hog(segmented_img, orientations=9, pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2), block_norm='L2-Hys',
                                  visualize=True)  
        hog_features.append(features) 
        hog_images.append(exposure.rescale_intensity(hog_image, in_range=(0, 10)))  
    return hog_features, hog_images, segmented_images

# Load dataset dengan helm dan tanpa helm
images_with_helm, labels_with_helm = load_images_from_folder('dataset/Helm', 1)
images_without_helm, labels_without_helm = load_images_from_folder('dataset/Tanpa', 0)

# Gabungkan data gambar dan label
images = images_with_helm + images_without_helm
labels = labels_with_helm + labels_without_helm

# Ekstraksi fitur HOG dan visualisasi
hog_features, hog_images, segmented_images = extract_hog_features_and_visualize(images)

# Menampilkan beberapa contoh gambar dan fitur HOG secara acak
num_images_to_show = 5
random_indices = random.sample(range(len(images)), num_images_to_show)

for idx in random_indices:
    fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    ax[0].imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB))  # Tampilkan gambar asli
    ax[0].set_title('Gambar Asli')
    ax[1].imshow(segmented_images[idx], cmap='gray')  # Tampilkan gambar hasil deteksi tepi
    ax[1].set_title('Gambar Segmentasi')
    ax[2].imshow(hog_images[idx], cmap='gray')  # Tampilkan gambar HOG
    ax[2].set_title('Gambar HOG')
    plt.show()
    # Print nilai fitur HOG
    print(f'Nilai fitur HOG untuk gambar {idx+1}:\n', hog_features[idx].reshape(-1, 9))

plt.close()
