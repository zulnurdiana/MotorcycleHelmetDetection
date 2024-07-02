import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (128, 128))
                images.append(img)
                labels.append(label)
                print(f'Memproses gambar {filename}')
    return images, labels

def segment_motor_helm(image):
    _, segmented_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return segmented_image

def compute_lbp(image):
    radius = 1
    num_points = 8 * radius
    lbp = np.zeros_like(image)
    
    x, y = np.meshgrid(range(-radius, radius + 1), range(-radius, radius + 1))
    theta = np.arctan2(y, x)
    theta[theta < 0] += 2 * np.pi
    
    for k in range(num_points):
        x_shift = np.round(radius * np.cos(2 * np.pi * k / num_points)).astype(int)
        y_shift = -np.round(radius * np.sin(2 * np.pi * k / num_points)).astype(int)
        shifted_image = np.roll(image, (y_shift, x_shift), axis=(0, 1))
        lbp += (shifted_image > image)
    
    return lbp.astype(np.uint8)

images_with_helm, labels_with_helm = load_images_from_folder('dataset/Helm', 1)
images_without_helm, labels_without_helm = load_images_from_folder('dataset/Tanpa', 0)

images = images_with_helm + images_without_helm
labels = labels_with_helm + labels_without_helm

num_images_to_show = min(5, len(images))
random_indices = random.sample(range(len(images)), num_images_to_show)

for idx in random_indices:
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    ax[0].imshow(images[idx], cmap='gray')
    ax[0].set_title('Gambar Asli')
    
    segmented_img = segment_motor_helm(images[idx])
    ax[1].imshow(segmented_img, cmap='gray')
    ax[1].set_title('Gambar Setelah Segmentasi')
    
    lbp = compute_lbp(segmented_img)
    ax[2].imshow(lbp, cmap='gray')
    ax[2].set_title('Gambar LBP')
    
    plt.tight_layout()
    plt.show()
    
    hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-7)
    print(f'Nilai fitur LBP untuk gambar {idx+1}:\n', hist.flatten())

plt.close()
