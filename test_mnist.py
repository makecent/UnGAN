import numpy as np
import torch
from torchvision import datasets, transforms
from my_codes import FcClassifier, FcGenerator, display_results, get_mappings
from torch.nn import DataParallel
# %% Init
dir_name = 'mnist_fc_2vs224'
display_step = 1
batch_size = 64
embedding_channels = 2
noise_channels = 224
image_shape = (1, 28, 28)
num_classes = 10

# %% Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
C = DataParallel(FcClassifier(num_classes, image_shape).to(device))
C.load_state_dict(torch.load(f'logs/{dir_name}/checkpoints/C_epoch200.pth'))
C.eval()
G = DataParallel(FcGenerator(num_classes, embedding_channels, noise_channels, image_shape).to(device))
G.load_state_dict(torch.load(f'logs/{dir_name}/checkpoints/G_epoch200.pth'))
G.eval()

# %% Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))])
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transform, download=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# %%Main
# ========================== Test Classifier ===================================== #
# pred
preds, labels = [], []
for batch_idx, (real_images, class_labels) in enumerate(test_loader):
    pred = C(real_images.to(device)).argmax(-1).detach().cpu().tolist()
    labels.extend(class_labels.tolist())
    preds.extend(pred)
labels = np.array(labels)
preds = np.array(preds)
acc, mapping = get_mappings(labels, preds)
print(f'Best Accuracy: {acc:.2%}')

# plot
itered = iter(test_loader)
real_images, _ = next(itered)
pred = C(real_images.to(device)).argmax(-1).cpu().numpy()
display_results(pred, real_images, title='Classified Test Images')
display_results(np.array([mapping[p] for p in pred]), real_images, title='Classified Test Images (Mapped)')

# =========================== Test Generator ========================================== #
class_labels = torch.randint(10, size=[64])
fake_images = G(class_labels.to(device)).cpu()
class_labels = class_labels.numpy()

display_results(class_labels, fake_images, title='Generated Conditional Images')
