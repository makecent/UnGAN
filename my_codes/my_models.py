import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from .my_utils import display_progress, display_results


class UnGAN(nn.Module):

    def __init__(self,
                 generator,
                 discriminator,
                 classifier,
                 gen_optimizer,
                 dsc_optimizer,
                 cls_optimizer,
                 adv_loss,
                 cls_loss,
                 num_classes=10,
                 display_step=1):

        super().__init__()
        self.gen = generator
        self.dsc = discriminator
        self.cls = classifier

        self.gen_opt = gen_optimizer
        self.dsc_opt = dsc_optimizer
        self.cls_opt = cls_optimizer
        self.adversarial_criterion = adv_loss
        self.cls_criterion = cls_loss

        self.display_step = display_step
        self.num_classes = num_classes

    def dsc_step(self, real_images, conditions):
        self.gen.eval()
        self.dsc.train()
        self.dsc.zero_grad()
        bs = real_images.size(0)

        # train discriminator on real
        real_labels = torch.ones(bs, 1).type_as(real_images)
        real_logits = self.dsc(real_images)
        D_real_loss = self.adversarial_criterion(real_logits, real_labels)

        # train discriminator on facke
        fake_labels = torch.zeros(bs, 1).type_as(real_images)
        fake_images = self.gen(conditions)
        fake_logits = self.dsc(fake_images)
        D_fake_loss = self.adversarial_criterion(fake_logits, fake_labels)

        # gradient backprop & optimize ONLY self.dsc's parameters
        D_loss = D_real_loss + D_fake_loss
        D_loss.backward()
        self.dsc_opt.step()

        # return D_loss.data.item(), D_fake_loss.data.item(), D_real_loss.data.item()
        return D_loss.data.item()

    def gen_step(self, conditions):
        self.dsc.eval()
        self.gen.train()
        self.gen.zero_grad()
        bs = conditions.size(0)

        # train generator to fool discriminator
        fake_labels = torch.ones(bs, 1).cuda()
        fake_images = self.gen(conditions)
        fake_logits = self.dsc(fake_images)
        G_loss = self.adversarial_criterion(fake_logits, fake_labels)

        # gradient backprop & optimize ONLY self.gen's parameters
        G_loss.backward()
        self.gen_opt.step()

        return G_loss.data.item()

    def cls_step(self, conditions):
        self.gen.eval()
        # self.gen.train()
        # self.gen.zero_grad()
        self.cls.train()
        self.cls.zero_grad()

        # train generator to fool discriminator
        fake_images = self.gen(conditions, drop=True)
        fake_predicted_cls = self.cls(fake_images)
        C_loss = self.cls_criterion(fake_predicted_cls, conditions)
        C_loss.backward()
        self.cls_opt.step()
        # self.gen_opt.step()

        return C_loss.data.item()

    def get_acc(self, conditions):
        self.gen.eval()
        self.cls.eval()
        bs = conditions.size(0)
        # train generator to fool discriminator
        fake_images = self.gen(conditions)
        fake_predicted_cls = self.cls(fake_images)
        acc = torch.eq(fake_predicted_cls.argmax(dim=-1), conditions).sum() / bs
        return acc.item()

    def fit(self, dataloader, n_epoch, device, log_dir, logger):
        # Init
        writer = SummaryWriter(f'{log_dir}/tensorboard')
        n_iters = 0
        # Train
        figs = []
        for epoch in range(1, n_epoch + 1):
            for batch_index, (real_images, _) in enumerate(dataloader):
                real_images = real_images.to(device)
                class_labels = torch.randint(self.num_classes, size=[real_images.size(0)]).to(device)

                loss_d = self.dsc_step(real_images, class_labels)
                loss_g = self.gen_step(class_labels)
                writer.add_scalars('adv_loss', {'loss_d': loss_d, 'loss_g': loss_g}, n_iters)

                if batch_index % 10 == 0:
                    loss_c = self.cls_step(class_labels.to(device))
                    writer.add_scalar('cls_loss', loss_c, n_iters)
                acc_c = self.get_acc(class_labels)
                writer.add_scalar('cls_acc', acc_c, n_iters)
                n_iters += 1
            logger.info(f'[{epoch:03d}/{n_epoch}]: loss_d: {loss_d:.3f}, loss_g: {loss_g:.3f}, acc_c: {acc_c:.2%}')

            if epoch % self.display_step == 0:
                fake_images = self.gen(class_labels.to(device))
                fig = display_progress(epoch, class_labels[:4], fake_images[:4], real_images[:4],
                                       save_fig=f'{log_dir}/figures')
                figs.append(fig)
            torch.save(self.gen.state_dict(), f'{log_dir}/checkpoints/G_epoch{epoch}.pth')
            torch.save(self.dsc.state_dict(), f'{log_dir}/checkpoints/D_epoch{epoch}.pth')
            torch.save(self.cls.state_dict(), f'{log_dir}/checkpoints/C_epoch{epoch}.pth')

        # Save results
        writer.add_figure('figures', figs)
        fake_images = self.gen(class_labels[:64].to(device))
        display_results(class_labels[:64], fake_images, title='Generated Conditional Images',
                        save_fig=f'{log_dir}/figures/gen_result.jpg')

        pred = self.cls(real_images[:64].to(device)).argmax(-1)
        display_results(pred, real_images[:64], title='Classified Test Images',
                        save_fig=f'{log_dir}/figures/cls_result.jpg')


class DownSampleConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=4, strides=2, padding=1, activation=True, batchnorm=True):
        """
        Paper details:
        - C64-C128-C256-C512-C512-C512-C512-C512
        - All convolutions are 4×4 spatial filters applied with stride 2
        - Convolutions in the encoder downsample by a factor of 2
        """
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm

        self.conv = nn.Conv2d(in_channels, out_channels, kernel, strides, padding)

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)

        if activation:
            self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        if self.batchnorm:
            x = self.bn(x)
        if self.activation:
            x = self.act(x)
        return x


class UpSampleConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=4, strides=2, padding=1, activation=True, batchnorm=True,
                 dropout=False):
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm
        self.dropout = dropout

        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel, strides, padding)

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)

        if activation:
            self.act = nn.ReLU(True)

        if dropout:
            self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.deconv(x)
        if self.batchnorm:
            x = self.bn(x)

        if self.dropout:
            x = self.drop(x)
        return x


class UnetGenerator(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # encoder/donwsample convs
        self.encoders = [DownSampleConv(in_channels, 64, batchnorm=False),  # bs x 128 x 64 x 64
                         DownSampleConv(64, 128),  # bs x 256 x 32 x 32
                         DownSampleConv(128, 256),  # bs x 512 x 16 x 16
                         DownSampleConv(256, 512),  # bs x 512 x 8 x 8
                         DownSampleConv(512, 512),  # bs x 512 x 4 x 4
                         DownSampleConv(512, 512),  # bs x 512 x 2 x 2
                         DownSampleConv(512, 512, batchnorm=False),  # bs x 512 x 1 x 1
                         ]

        # decoder/upsample convs
        self.decoders = [UpSampleConv(512, 512, dropout=True),  # bs x 512 x 2 x 2
                         UpSampleConv(1024, 512, dropout=True),  # bs x 512 x 4 x 4
                         UpSampleConv(1024, 512, dropout=True),  # bs x 512 x 8 x 8
                         UpSampleConv(1024, 256),  # bs x 512 x 16 x 16
                         UpSampleConv(512, 128),  # bs x 256 x 32 x 32
                         UpSampleConv(256, 64),  # bs x 128 x 64 x 64
                         ]
        self.decoder_channels = [512, 512, 512, 512, 256, 128, 64]
        self.final_conv = nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

    def forward(self, x):
        skips_cons = []
        for encoder in self.encoders:
            x = encoder(x)

            skips_cons.append(x)

        skips_cons = list(reversed(skips_cons[:-1]))
        decoders = self.decoders[:-1]

        for decoder, skip in zip(decoders, skips_cons):
            x = decoder(x)
            # print(x.shape, skip.shape)
            x = torch.cat((x, skip), axis=1)

        x = self.decoders[-1](x)
        # print(x.shape)
        x = self.final_conv(x)
        return self.tanh(x)


# class CA_NET(nn.Module):
#     # some code is modified from vae examples
#     # (https://github.com/pytorch/examples/blob/master/vae/main.py)
#     def __init__(self):
#         super(CA_NET, self).__init__()
#         self.t_dim = 100
#         self.c_dim = 100
#         self.fc = nn.Linear(self.t_dim, self.c_dim * 2, bias=True)
#         self.relu = nn.ReLU()
#
#     def encode(self, text_embedding):
#         x = self.relu(self.fc(text_embedding))
#         mu = x[:, :self.c_dim]
#         logvar = x[:, self.c_dim:]
#         return mu, logvar
#
#     def reparametrize(self, mu, logvar):
#         std = logvar.mul(0.5).exp_()
#         eps = torch.cuda.FloatTensor(std.size()).normal_()
#         eps = Variable(eps)
#         return eps.mul(std).add_(mu)
#
#     def forward(self, text_embedding):
#         mu, logvar = self.encode(text_embedding)
#         c_code = self.reparametrize(mu, logvar)
#         return c_code, mu, logvar

class UpGenerator(nn.Module):

    @staticmethod
    def conv3x3(in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    def upBlock(self, in_planes, out_planes):
        block = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), self.conv3x3(in_planes, out_planes),
                              nn.InstanceNorm2d(out_planes), nn.LeakyReLU(0.2, True))
        return block

    def __init__(self, num_classes, noise_channels, embedding_channels, image_shape, latent_channels=128):
        super().__init__()
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.noise_channels = noise_channels
        self.latent_channels = latent_channels
        image_channels = image_shape[0]

        self.embedding = nn.Embedding(num_classes, embedding_channels)
        self.fc = nn.Linear(embedding_channels + noise_channels, latent_channels * 4 * 4)  # 128 x 4 x 4
        # self.norm = nn.InstanceNorm1d(latent_channels * 4 * 4)

        self.upsample1 = self.upBlock(latent_channels, latent_channels // 2)  # 64 x 8 x 8
        self.upsample2 = self.upBlock(latent_channels // 2, latent_channels // 4)  # 32 x 16 x 16
        self.upsample3 = self.upBlock(latent_channels // 4, latent_channels // 8)  # 16 x 32 x 32
        self.upsample4 = self.upBlock(latent_channels // 8, latent_channels // 16)  # 8 x 64 x 64
        self.img = nn.Sequential(self.conv3x3(latent_channels // 16, image_channels), nn.Tanh())  # 3 x 64 x 64

    def forward(self, x):
        x = self.embedding(x)
        noises = torch.randn(x.size(0), self.noise_channels).type_as(x)
        x = torch.cat((x, noises), dim=1)
        x = F.leaky_relu(self.fc(x), 0.2)
        x = x.view(-1, self.latent_channels, 4, 4)

        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)
        fake_img = self.img(x)
        return fake_img


# class PatchGAN(nn.Module):
#
#     def __init__(self, input_channels):
#         super().__init__()
#         self.d1 = DownSampleConv(input_channels, 64, batchnorm=False)
#         self.d2 = DownSampleConv(64, 128)
#         self.d3 = DownSampleConv(128, 256)
#         self.d4 = DownSampleConv(256, 512)
#         self.final = nn.Conv2d(512, 1, kernel_size=1)
#
#     def forward(self, x):
#         x = self.d1(x)
#         x = self.d2(x)
#         x = self.d3(x)
#         x = self.d4(x)
#         x = self.final(x)
#         return x


class DownClassifier(nn.Module):
    def __init__(self, n_classes, image_shape):
        super().__init__()
        image_channels = image_shape[0]
        self.d1 = DownSampleConv(image_channels, 64, batchnorm=False)
        self.d2 = DownSampleConv(64, 128)
        self.d3 = DownSampleConv(128, 256)
        self.d4 = DownSampleConv(256, 256)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.pool(x).flatten(start_dim=1)
        x = self.fc(x)
        return x


class DownDiscriminator(nn.Module):
    def __init__(self, image_shape):
        super().__init__()
        image_channels = image_shape[0]
        self.d1 = DownSampleConv(image_channels, 64, batchnorm=False)
        self.d2 = DownSampleConv(64, 128)
        self.d3 = DownSampleConv(128, 256)
        self.d4 = DownSampleConv(256, 256)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.pool(x).flatten(start_dim=1)
        x = self.fc(x)
        return x


class FcGenerator(nn.Module):

    def __init__(self, num_embeddings=10, embedding_channels=10, noise_channels=224, image_shape=(1, 28, 28)):
        super().__init__()
        self.image_shape = image_shape
        self.embedding_channels = embedding_channels
        self.noise_channels = noise_channels
        if embedding_channels > 0:
            self.embedding = nn.Embedding(num_embeddings, embedding_channels)
        # normalized_means = [(2*i/(num_embeddings-1)) - 1 for i in range(num_embeddings)]
        # self.distributions = [torch.distributions.Normal(torch.tensor([mean]), torch.tensor([1.0])) for mean in normalized_means]

        image_dim = np.prod(image_shape).item()
        self.fc1 = nn.Linear(embedding_channels + noise_channels, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, image_dim)
        self.drop = nn.Dropout(0.5)

    def forward(self, x, drop=False):
        if self.embedding_channels > 0:
            x = self.embedding(x)
            noise = torch.randn(x.size(0), self.noise_channels).type_as(x)
            x = torch.cat([x, noise], dim=-1)
        else:
            x = torch.randn(x.size(0), self.noise_channels, device=x.device, dtype=torch.float)
        noises = [torch.normal(mean=1 * x.float(), std=torch.ones(x.size(0)).cuda()) for k in
                  range(self.noise_channels)]
        # noises = [self.distributions[d].sample((self.noise_channels,)).cuda() for d in x]
        x = torch.stack(noises).squeeze()
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = torch.tanh(self.fc4(x))
        if drop:
            x = self.drop(x)
        return x.view(-1, *self.image_shape)


class FcDiscriminator(nn.Module):

    def __init__(self, image_shape=(1, 28, 28)):
        super().__init__()
        image_dim = np.prod(image_shape).item()
        self.fc1 = nn.Linear(image_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    # forward method
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return self.fc4(x)


class FcClassifier(FcDiscriminator):

    def __init__(self, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc4 = nn.Linear(self.fc3.out_features, num_classes)
