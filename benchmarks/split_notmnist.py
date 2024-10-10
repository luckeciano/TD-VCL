import numpy as np
import gzip
import pickle
from scipy.io import loadmat
from PIL import Image

class SplitNotMNIST():
    def __init__(self):
        f = gzip.open('data/notMNIST_large.pkl.gz', 'rb')
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        train_set, valid_set, test_set = u.load()
        f.close()

        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.X_test = test_set[0]
        self.train_label = np.hstack((train_set[1], valid_set[1]))
        self.test_label = test_set[1]

        self.sets_0 = ['A', 'B', 'C', 'D', 'E']
        self.sets_1 = ['F', 'G', 'H', 'I', 'J']
        self.max_iter = len(self.sets_0)
        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], 2

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            # Retrieve train data
            train_0_id = np.where(self.train_label == self.sets_0[self.cur_iter])[0]
            train_1_id = np.where(self.train_label == self.sets_1[self.cur_iter])[0]
            next_x_train = np.vstack((self.X_train[train_0_id], self.X_train[train_1_id]))

            next_y_train = np.vstack((np.ones((train_0_id.shape[0], 1)), np.zeros((train_1_id.shape[0], 1))))
            next_y_train = np.hstack((next_y_train, 1-next_y_train))

            # Retrieve test data
            test_0_id = np.where(self.test_label == self.sets_0[self.cur_iter])[0]
            test_1_id = np.where(self.test_label == self.sets_1[self.cur_iter])[0]
            next_x_test = np.vstack((self.X_test[test_0_id], self.X_test[test_1_id]))

            next_y_test = np.vstack((np.ones((test_0_id.shape[0], 1)), np.zeros((test_1_id.shape[0], 1))))
            next_y_test = np.hstack((next_y_test, 1-next_y_test))

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_train, next_y_train
        
    def generate_image(self, label, index, save_path):
        """
        Filters X_train by label, selects the image at the given index from the filtered data, and saves it.

        Parameters:
        label (str): The label to filter X_train by (a letter from 'A' to 'G').
        index (int): The index of the image to be saved within the filtered data.
        save_path (str): The path where the image will be saved.
        """
        # Ensure the label is valid
        if label not in self.sets_0 + self.sets_1:
            raise ValueError("Invalid label")

        # Filter X_train by the given label
        filtered_indices = np.where(self.train_label == label)[0]
        if index < 0 or index >= len(filtered_indices):
            raise ValueError("Index out of range in filtered data")

        # Get the image data from the filtered X_train
        image_data = self.X_train[filtered_indices[index]].reshape(28, 28)

        # Convert the image data to a PIL Image object
        image = Image.fromarray(np.uint8(image_data * 255), 'L')

        # Save the image
        image.save(save_path)
        print(f"Image saved to {save_path}")

# Example usage
# Assuming you have an instance of SplitNotMNIST
# split_not_mnist = SplitNotMNIST()

# for i in range(10):
#     idx = i
#     split_not_mnist.generate_image('F', idx, f'data/notmnist_image_F_{idx}.png')
#     split_not_mnist.generate_image('B', idx, f'data/notmnist_image_B_{idx}.png')
#     split_not_mnist.generate_image('G', idx, f'data/notmnist_image_G_{idx}.png')
#     split_not_mnist.generate_image('C', idx, f'data/notmnist_image_C_{idx}.png')
#     split_not_mnist.generate_image('H', idx, f'data/notmnist_image_H_{idx}.png')
#     # split_not_mnist.generate_image('A', 1, 'data/notmnist_image_A_second.png')