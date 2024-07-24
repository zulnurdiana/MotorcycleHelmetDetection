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

def canny_edge_detection(image):
    edges = cv2.Canny(image, 100, 200)
    return edges

# def segment_motor(image):
#     img=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#     vectorized = img.reshape((-1,3))
#     vectorized = np.float32(vectorized)
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#     K = 2
#     attempts=10
#     ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
#     center = np.uint8(center)
#     res = center[label.flatten()]
#     mask = res.reshape((img.shape))
#     return mask

def segment_motor(image):
    orb = cv2.ORB_create()
    orb.setEdgeThreshold(4)
    kp = orb.detect(image, None)
    kp, des = orb.compute(image, kp)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for point in kp:
        x, y = point.pt 
        x, y = int(x), int(y)
        mask = cv2.circle(mask, (x, y), 1, (255, 255, 255), thickness=-1)
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.erode(mask, None, iterations=2)
    return mask

def extract_hog_features_and_visualize(images):
    hog_features = []
    hog_images = []
    segmented_images = []
    for img in images:
        segmented_img = cv2.bitwise_and(img, img, mask=segment_motor(img))
        segmented_images.append(segmented_img)
        features, hog_image = hog(cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY), orientations=9, pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2), block_norm='L2-Hys', transform_sqrt=True, visualize=True)
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

for idx in random_indices:
    fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    ax[0].imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB))
    ax[0].set_title('Gambar Asli')
    ax[1].imshow(cv2.cvtColor(segmented_images[idx], cv2.COLOR_BGR2RGB))
    ax[1].set_title('Gambar Segmentasi')
    ax[2].imshow(hog_images[idx], cmap='gray')
    ax[2].set_title('Gambar HOG')
    plt.show()

plt.close()
