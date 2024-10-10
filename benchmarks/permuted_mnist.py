import numpy as np
import gzip
import pickle
from copy import deepcopy
from PIL import Image
import cv2


class PermutedMNIST():
    def __init__(self, max_iter=10, seed=42):
        f = gzip.open('data/mnist.pkl.gz', 'rb')
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        train_set, valid_set, test_set = u.load()
        f.close()

        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.Y_train = np.hstack((train_set[1], valid_set[1]))
        self.X_test = test_set[0]
        self.Y_test = test_set[1]
        self.max_iter = max_iter
        self.seed = seed
        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], 10
    
    def reset_env(self):
        self.cur_iter = 0

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            np.random.seed(self.cur_iter * self.seed)
            perm_inds = list(range(self.X_train.shape[1]))
            np.random.shuffle(perm_inds)

            # Retrieve train data
            next_x_train = deepcopy(self.X_train)
            next_x_train = next_x_train[:,perm_inds]
            next_y_train = np.eye(10)[self.Y_train]

            # Retrieve test data
            next_x_test = deepcopy(self.X_test)
            next_x_test = next_x_test[:,perm_inds]
            next_y_test = np.eye(10)[self.Y_test]

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

    def generate_image(self, index, save_path):
        """
        Generates and saves an image from the MNIST-style dataset.

        Parameters:
        index (int): The index of the image to be saved.
        save_path (str): The path where the image will be saved.
        """
        # Ensure the index is within the range of the dataset
        if index < 0 or index >= len(self.X_train):
            raise ValueError("Index out of range")

        # Get the image data from X_train
        image_data = self.X_train[index].reshape(28, 28)

        # Convert the image data to a PIL Image object
        image = Image.fromarray(np.uint8(image_data * 255), 'L')

        # Save the image
        image.save(save_path)
        print(f"Image saved to {save_path}")

        # np.random.seed(768)
        # perm_inds = list(range(self.X_train.shape[1]))
        # np.random.shuffle(perm_inds)
        # permuted_image_data = self.X_train[index][perm_inds].reshape(28, 28)
        # permuted_image = Image.fromarray(np.uint8(permuted_image_data * 255), 'L')

        # # Save the image
        # permuted_image.save("data/permuted_mnist_pattern_3_third.png")
        # print(f"Image saved to {save_path}")

    def generate_filtered_image(self, label, index, save_path):
        """
        Filters X_train by label, selects the image at the given index from the filtered data, and saves it.

        Parameters:
        label (int): The label to filter X_train by.
        index (int): The index of the image to be saved within the filtered data.
        save_path (str): The path where the image will be saved.
        """
        # Filter X_train by the given label
        filtered_indices = np.where(self.Y_train == label)[0]
        if index < 0 or index >= len(filtered_indices):
            raise ValueError("Index out of range in filtered data")

        # Get the image data from the filtered X_train
        image_data = self.X_train[filtered_indices[index]].reshape(28, 28)

        # Convert the image data to a PIL Image object
        image = Image.fromarray(np.uint8(image_data * 255), 'L')

        # Save the image
        image.save(save_path)
        print(f"Filtered image saved to {save_path}")

# # Example usage
# # Assuming you have an instance of PermutedMNIST
# permuted_mnist = PermutedMNIST()
# permuted_mnist.generate_image(2, 'data/mnist_image_third.png')
# permuted_mnist.generate_filtered_image(0, 2, 'data/mnist_image_0_second.png')
# permuted_mnist.generate_filtered_image(5, 2, 'data/mnist_image_5_second.png')