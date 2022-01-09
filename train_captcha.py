from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from my_codes import UnGAN, UpGenerator, DownDiscriminator, DownClassifier, get_logger, CaptchaDataset

# %% Init
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
custom_name = None
if custom_name is None:
    version = len(list(Path('logs').iterdir())) + 1
    log_dir = f'logs/version_{version}'
else:
    log_dir = f'logs/{custom_name}'
Path(log_dir).mkdir()
(Path(log_dir) / 'checkpoints').mkdir()
logger = get_logger(log_dir)

# %% Hyper-parameters
display_step = 1
image_shape = (3, 64, 64)
num_classes = 10000
embedding_channels = 50
noise_channel = 1024
n_epoch = 1000
lr = (0.0002, 0.0002, 0.0002)
batch_size = 100

# %% Build Dataloader
transform = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
train_dataset = CaptchaDataset(img_dir='./captcha_data/', transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# %% Build Model
G = UpGenerator(num_classes, noise_channel, embedding_channels, image_shape).to(device)
D = DownDiscriminator(image_shape).to(device)
C = DownClassifier(num_classes, image_shape).to(device)
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
              cls_loss=cls_criterion,
              num_classes=num_classes)
# %% Train
ungan.fit(train_loader, n_epoch, device, log_dir=log_dir, logger=logger)
