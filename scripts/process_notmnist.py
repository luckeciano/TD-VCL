import os
import cv2
import numpy as np
import gzip
import random
import pickle

def process_images(root_dir):
    labels = []
    images = []
    for dir_name in os.listdir(root_dir):
        print(dir_name)
        dir_path = os.path.join(root_dir, dir_name)
        if os.path.isdir(dir_path):
            for file_name in os.listdir(dir_path):
                if file_name.endswith('.png'):
                    # Read the image as an array
                    img_path = os.path.join(dir_path, file_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                    if img is None:
                        print(f'Skipping image {img_path}')
                        continue
                    
                    # Normalize and flatten the image
                    img = img / 255.0
                    img = img.flatten()
                    
                    # Append the image and label to the respective lists
                    images.append(img)
                    labels.append(dir_name)
    
    # Convert the lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

def shuffle_arrays(a, b):
    # Combine the two arrays
    combined = list(zip(a, b))
    # Shuffle the combined array
    random.shuffle(combined)
    # Split the shuffled combined array back into two arrays
    a[:], b[:] = zip(*combined)

dataset = 'notMNIST_small'
images, labels = process_images(f'data/{dataset}')

shuffle_arrays(images, labels)

size = len(images)
train_idx = int(size * 0.925)
val_idx = int(size * 0.95)

train_x, train_y = images[:train_idx], labels[:train_idx]
train = [train_x, train_y]
valid_x, valid_y = images[train_idx:val_idx], labels[train_idx:val_idx]
valid = [valid_x, valid_y]
test_x, test_y = images[val_idx:], labels[val_idx:]
test = [test_x, test_y]


with gzip.open(f'data/{dataset}.pkl.gz', 'wb') as f:
    pickle.dump((train, valid, test), f)