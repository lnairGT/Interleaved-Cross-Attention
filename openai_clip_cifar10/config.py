import torch

debug = True
batch_size = 256
num_workers = 0
lr = 1e-4
weight_decay = 1e-1
patience = 2
factor = 0.5
epochs = 75
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_length = 200

# image size
size = 32

# for projection head; used for both image and text encoders
projection_dim = 256