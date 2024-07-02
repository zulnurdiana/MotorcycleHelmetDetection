import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.feature import hog
from skimage import exposure

def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                # img = cv2.resize(img, (128, 128))
                images.append(img)
                labels.append(label)
    return images, labels

def segment_motor_helm(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_color = np.array([0, 0, 0])
    upper_color = np.array([179, 255, 50])
    mask = cv2.inRange(hsv, lower_color, upper_color)
    segmented_image = cv2.bitwise_and(image, image, mask=mask)
    return segmented_image

def canny_edge_detection(image):
    edges = cv2.Canny(image, 100, 200)
    return edges

def extract_hog_features_and_visualize(images):
    hog_features = []
    hog_images = []
    segmented_images = []
    for img in images:
        segmented_img = segment_motor_helm(img)
        segmented_images.append(segmented_img)
        gray = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
        edges = canny_edge_detection(gray)
        features, hog_image = hog(edges, orientations=9, pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2), block_norm='L2-Hys',
                                  visualize=True)
        hog_features.append(features)
        hog_images.append(exposure.rescale_intensity(hog_image, in_range=(0, 10)))
    return hog_features, hog_images, segmented_images

images_with_helm, labels_with_helm = load_images_from_folder('dataset/Helm', 1)
images_without_helm, labels_without_helm = load_images_from_folder('dataset/Tanpa', 0)

images = images_with_helm + images_without_helm
labels = labels_with_helm + labels_without_helm

hog_features, hog_images, segmented_images = extract_hog_features_and_visualize(images)

num_images_to_show = 5
random_indices = random.sample(range(len(images)), num_images_to_show)



# denoised = cv2.GaussianBlur(gray, (5, 5), 0)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# contrast_adjusted = clahe.apply(denoised)
# edges = cv2.Canny(contrast_adjusted, 100, 200)
# cv2.imshow('Original Image', image)
# cv2.imshow('Grayscale', gray)
# cv2.imshow('Denoised', denoised)
# cv2.imshow('Contrast Adjusted', contrast_adjusted)
# cv2.imshow('Edges', edges)


for idx in random_indices:
    fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    ax[0].imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB))
    ax[0].set_title('Gambar Asli')
    ax[1].imshow(cv2.cvtColor(segmented_images[idx], cv2.COLOR_BGR2RGB))
    ax[1].set_title('Gambar Segmentasi')
    ax[2].imshow(hog_images[idx], cmap='gray')
    ax[2].set_title('Gambar HOG')
    plt.show()
    print(f'Nilai fitur HOG untuk gambar {idx+1}:\n', hog_features[idx].reshape(-1, 9))

plt.close()
