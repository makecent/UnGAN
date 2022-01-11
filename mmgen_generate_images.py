from mmgen.apis import init_model
from matplotlib import pyplot as plt
from torchvision.utils import make_grid, save_image
import torch
from mmcv import mkdir_or_exist
from pathlib import Path

config_file = 'mmgen_styleganv2_mnist_conditional.py'
# you can download this checkpoint in advance and use a local file path.
checkpoint_file = 'work_dirs/mmgen_styleganv2_mnist_conditional_embeddingmapfixed_batchincreased/ckpt/mmgen_styleganv2_mnist_conditional/iter_150000.pth'
device = 'cuda:0'
# init a generatvie
model = init_model(config_file, checkpoint_file, device=device)
batch_size = 16
num_images = 60000
batches = num_images // batch_size // 2
for d in range(10):
    mkdir_or_exist(f'datasets/generated_mnist/{d}')

# sample images
# eval mode
model.eval()
# no grad in sampling
with torch.no_grad():
    for i in range(batches):
        digits = torch.randint(10, [batch_size])
        kwargs = {'digits': digits}
        outputs_dict = model(
            None,
            return_loss=False,
            num_batches=batch_size,
            return_noise=True,
            **kwargs)
        fake_imgs = outputs_dict['fake_img']

        for j, img in enumerate(fake_imgs):
            save_image(img, f'datasets/generated_mnist/{digits[j%batch_size].item()}/img_{i*batches+j+1}.jpg')

folder = Path.cwd() / 'datasets/generated_mnist'
with open('annotations.txt', 'w') as f:
    for cf in folder.iterdir():
        for img in cf.iterdir():
            f.write(f'{cf.name}/{img.name} {cf.name[-1]}\n')
