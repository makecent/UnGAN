import os.path
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from my_codes import UnGAN, FcGenerator, FcDiscriminator, FcClassifier, get_logger

# %% Init
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
custom_name = 'mnist_fc_mean=-1to1_std=1.0'
if custom_name is None:
    version = len(list(Path('logs').iterdir())) + 1
    log_dir = f'logs/version_{version}'
else:
    log_dir = f'logs/{custom_name}'
    assert not os.path.exists(log_dir)
Path(log_dir).mkdir()
(Path(log_dir) / 'checkpoints').mkdir()
logger = get_logger(log_dir)

# %% Hyper-parameters
display_step = 1
image_shape = (1, 28, 28)
num_classes = 10
embedding_channels = 0
noise_channels = 224
n_epoch = 200
lr = (0.0002, 0.0002, 0.0002)
batch_size = 100

# %% Build Dataloader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# %% Build Model
G = nn.DataParallel(FcGenerator(num_classes, embedding_channels, noise_channels, image_shape).to(device))
D = nn.DataParallel(FcDiscriminator(image_shape).to(device))
C = nn.DataParallel(FcClassifier(num_classes, image_shape).to(device))
# optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr[0])
D_optimizer = optim.Adam(D.parameters(), lr=lr[1])
C_optimizer = optim.Adam(C.parameters(), lr=lr[2])
# loss
adv_criterion = nn.BCEWithLogitsLoss()
cls_criterion = nn.CrossEntropyLoss()
# UnGAN
ungan = UnGAN(generator=G,
              discriminator=D,
              classifier=C,
              gen_optimizer=G_optimizer,
              dsc_optimizer=D_optimizer,
              cls_optimizer=C_optimizer,
              adv_loss=adv_criterion,
              cls_loss=cls_criterion)
# %% Train
ungan.fit(train_loader, n_epoch, device, log_dir=log_dir, logger=logger)


