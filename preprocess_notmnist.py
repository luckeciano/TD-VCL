import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import gzip
import pickle
import utils

# Load the pre-trained ResNet model
model = models.resnet50(pretrained=True)
model.eval()

# Define the transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = 'notMNIST_small'

# Load the MNIST dataset
f = gzip.open(f'data/{dataset}.pkl.gz', 'rb')
u = pickle._Unpickler(f)
u.encoding = 'latin1'
train_set, valid_set, test_set = u.load()
f.close()

final_sets = []
for set in [train_set, valid_set, test_set]:
    x = set[0]
    y = np.array([ord(char) - 65 for char in set[1]])
    dataloader = utils.get_dataloader(x, y, batch_size=256, shuffle=True)

    # Run inference and save embeddings
    embeddings = []
    labels = []
    with torch.no_grad():
        for images, ys in dataloader:
            # If your model is on GPU, you may need to move your images to GPU
            # images = images.to(device)
            images = images.view(-1, 1, 28, 28)
            images = images.repeat(1, 3, 1, 1)
            output = model(images)
            embeddings.append(output.cpu().numpy())
            labels.append(ys.cpu().numpy())
            

    # Save the embeddings
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    char_labels = np.array([chr(int(num) + 65) for num in labels])
    final_set = [embeddings, char_labels]
    final_sets.append(final_set)

with gzip.open(f'data/{dataset}_restnet.pkl.gz', 'wb') as f:
    pickle.dump(final_sets, f)
