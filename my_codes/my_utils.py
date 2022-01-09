from torchvision import utils
from matplotlib import pyplot as plt
import os
import logging
import os.path as osp

import mmcv
import torch
from mmcv.runner import HOOKS, Hook
from mmcv.runner.dist_utils import master_only
import numpy as np
from torchvision.utils import save_image


def display_progress(epoch_idx, cond, fake, real, figsize=(10, 5), save_fig=None):
    img_height, img_width = real.shape[-2:]
    padding = 14

    real = utils.make_grid(real.detach().cpu(), padding=padding, pad_value=0, normalize=True).permute(1, 2, 0)
    fake = utils.make_grid(fake.detach().cpu(), padding=padding, pad_value=0, normalize=True).permute(1, 2, 0)
    cond = cond.detach().cpu().tolist()

    fig, ax = plt.subplots(2, 1, figsize=figsize)
    fig.suptitle(f'Epoch {epoch_idx}')
    ax[0].imshow(real)
    ax[0].set_title("Real Images")
    ax[0].axis("off")
    ax[1].imshow(fake)
    ax[1].set_title("Fake Images")
    ax[1].axis("off")
    for i in range(1, len(cond) + 1):
        ax[1].annotate(f'{cond[i - 1]:04d}', xy=[padding * i + img_width * (i - 0.5), padding / 2], ha='center', color='w')
    if save_fig is not None:
        if not os.path.exists(save_fig):
            os.makedirs(save_fig)
        fig.savefig(f'{save_fig}/epoch_{epoch_idx}.jpg')
    plt.show()
    # if writer is not None:
    #     writer.add_figure(f'epoch_{epoch_idx}', fig)
    return fig


def display_results(cond, images, title=None, save_fig=None):
    img_height, img_width = images.shape[-2:]
    padding = img_width // 2
    cond = cond.reshape(8, 8)

    images = utils.make_grid(images, nrow=8, padding=padding, pad_value=0, normalize=True).permute(1, 2, 0).cpu()

    plt.figure(figsize=(10, 10))
    plt.imshow(images)
    if title is not None:
        plt.title(title)
    for i in range(1, 9):
        for j in range(1, 9):
            x = padding * i + img_width * (i - 0.5)
            y = padding * j + img_height * (j - 1.0)
            plt.annotate(f'{cond[j - 1, i - 1]}', xy=[x, y], ha='center', color='w')
    if save_fig is not None:
        plt.savefig(save_fig)
    plt.show()
    return plt.gcf()


def get_logger(log_dir):
    # 创建logger实例
    logger = logging.getLogger('ungan')
    # 设置logger实例的等级
    logger.setLevel(logging.INFO)
    # 创建formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s')

    # 创建控制台handler
    cons_handler = logging.StreamHandler()
    cons_handler.setLevel(logging.INFO)
    cons_handler.setFormatter(formatter)

    # 创建文件handler
    file_handler = logging.FileHandler(f'{log_dir}/logs.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # 添加handler到logger
    logger.addHandler(cons_handler)
    logger.addHandler(file_handler)

    return logger


def get_mappings(labels, preds):
    mapping = np.arange(10)
    accuracies = np.zeros([10, 10])
    for i in range(10):
        for j in range(10):
            true_positive = (preds[labels == j] == i)
            recall = sum(true_positive) / len(true_positive)
            accuracies[i, j] = recall
    for k in range(10):
        ki, kj = np.unravel_index(accuracies.argmax(), accuracies.shape)
        accuracies[ki, :] = 0
        accuracies[:, kj] = 0
        mapping[ki] = kj
    mapped_preds = np.array([mapping[p] for p in preds])
    acc = sum(mapped_preds == labels) / len(labels)
    return acc, mapping


@HOOKS.register_module()
class VisualizeConditionalSamples(Hook):
    """Visualization hook for unconditional GANs.

    In this hook, we use the official api `save_image` in torchvision to save
    the visualization results.

    Args:
        output_dir (str): The file path to store visualizations.
        fixed_noise (bool, optional): Whether to use fixed noises in sampling.
            Defaults to True.
        num_samples (int, optional): The number of samples to show in
            visualization. Defaults to 16.
        interval (int): The interval of calling this hook. If set to -1,
            the visualization hook will not be called. Default: -1.
        filename_tmpl (str): Format string used to save images. The output file
            name will be formatted as this args. Default: 'iter_{}.png'.
        rerange (bool): Whether to rerange the output value from [-1, 1] to
            [0, 1]. We highly recommend users should preprocess the
            visualization results on their own. Here, we just provide a simple
            interface. Default: True.
        bgr2rgb (bool): Whether to reformat the channel dimension from BGR to
            RGB. The final image we will save is following RGB style.
            Default: True.
        nrow (int): The number of samples in a row. Default: 1.
        padding (int): The number of padding pixels between each samples.
            Default: 4.
        kwargs (dict | None, optional): Key-word arguments for sampling
            function. Defaults to None.
    """

    def __init__(self,
                 output_dir,
                 fixed_noise=True,
                 fixed_digits=True,
                 num_samples=16,
                 interval=-1,
                 filename_tmpl='iter_{}.png',
                 rerange=True,
                 bgr2rgb=True,
                 nrow=4,
                 padding=0,
                 kwargs=None):
        self.output_dir = output_dir
        self.fixed_noise = fixed_noise
        self.fixed_digits = fixed_digits
        self.num_samples = num_samples
        self.interval = interval
        self.filename_tmpl = filename_tmpl
        self.bgr2rgb = bgr2rgb
        self.rerange = rerange
        self.nrow = nrow
        self.padding = padding

        # the sampling noise will be initialized by the first sampling.
        self.sampling_noise = None
        self.digits = None

        self.kwargs = kwargs if kwargs is not None else dict()

    @master_only
    def after_train_iter(self, runner):

        if not self.every_n_iters(runner, self.interval):
            return
        if self.digits is None and self.fixed_digits:
            self.digits = torch.cat([torch.arange(10,), torch.randint(10, [self.num_samples - 10])])
            runner.logger.info(f"The fixed digits are {self.digits}")
            self.kwargs.update({'digits': self.digits})
        # eval mode
        runner.model.eval()
        # no grad in sampling
        with torch.no_grad():
            outputs_dict = runner.model(
                self.sampling_noise,
                return_loss=False,
                num_batches=self.num_samples,
                return_noise=True,
                **self.kwargs)
            imgs = outputs_dict['fake_img']
            noise_ = outputs_dict['noise_batch']
        # initialize samling noise with the first returned noise
        if self.sampling_noise is None and self.fixed_noise:
            self.sampling_noise = noise_

        # train mode
        runner.model.train()

        filename = self.filename_tmpl.format(runner.iter + 1)
        if self.rerange:
            imgs = ((imgs + 1) / 2)
        if self.bgr2rgb and imgs.size(1) == 3:
            imgs = imgs[:, [2, 1, 0], ...]
        if imgs.size(1) == 1:
            imgs = torch.cat([imgs, imgs, imgs], dim=1)
        imgs = imgs.clamp_(0, 1)

        mmcv.mkdir_or_exist(osp.join(runner.work_dir, self.output_dir))
        save_image(
            imgs,
            osp.join(runner.work_dir, self.output_dir, filename),
            nrow=self.nrow,
            padding=self.padding)


# @HOOKS.register_module()
# class VisualizeConditionalSamples(Hook):
#
#     def __init__(self,
#                  output_dir,
#                  fixed_noise=True,
#                  num_samples=16,
#                  interval=-1,
#                  filename_tmpl='iter_{}.png',
#                  rerange=True,
#                  bgr2rgb=True,
#                  nrow=4,
#                  padding=20,
#                  kwargs=None):
#         self.output_dir = output_dir
#         self.fixed_noise = fixed_noise
#         self.num_samples = num_samples
#         self.interval = interval
#         self.filename_tmpl = filename_tmpl
#         self.bgr2rgb = bgr2rgb
#         self.rerange = rerange
#         self.nrow = nrow
#         self.padding = padding
#
#         # the sampling noise will be initialized by the first sampling.
#         self.sampling_noise = None
#
#         self.kwargs = kwargs if kwargs is not None else dict()
#
#     @master_only
#     def after_train_iter(self, runner):
#         if not self.every_n_iters(runner, self.interval):
#             return
#         # eval mode
#         runner.model.eval()
#         # no grad in sampling
#         with torch.no_grad():
#             outputs_dict = runner.model(
#                 self.sampling_noise,
#                 return_loss=False,
#                 num_batches=self.num_samples,
#                 return_noise=True,
#                 **self.kwargs)
#             imgs = outputs_dict['fake_img']
#             noise_ = outputs_dict['noise_batch']
#         # initialize samling noise with the first returned noise
#         if self.sampling_noise is None and self.fixed_noise:
#             self.sampling_noise = noise_
#
#         # train mode
#         runner.model.train()
#
#         filename = self.filename_tmpl.format(runner.iter + 1)
#         if self.rerange:
#             imgs = ((imgs + 1) / 2)
#         if self.bgr2rgb and imgs.size(1) == 3:
#             imgs = imgs[:, [2, 1, 0], ...]
#         if imgs.size(1) == 1:
#             imgs = torch.cat([imgs, imgs, imgs], dim=1)
#         imgs = imgs.clamp_(0, 1)
#
#         mmcv.mkdir_or_exist(osp.join(runner.work_dir, self.output_dir))
#         images = utils.make_grid(imgs, nrow=self.nrow, padding=self.padding, pad_value=0).permute(1, 2, 0)
#
#         plt.figure(figsize=(10, 10))
#         plt.imshow(images)
#         plt.title(f'Generated fake images {runner.iter}')
#         for i in range(1, 5):
#             for j in range(1, 5):
#                 x = self.padding * i + images.shape[-1] * (i - 0.5)
#                 y = self.padding * j + images.shape[-2] * (j - 1.0)
#                 plt.annotate(f'{noise_[j - 1, i - 1]}', xy=[x, y], ha='center', color='w')
#
#         mmcv.mkdir_or_exist(osp.join(runner.work_dir, self.output_dir))
#         plt.savefig(osp.join(runner.work_dir, self.output_dir, filename))
#         plt.show()

