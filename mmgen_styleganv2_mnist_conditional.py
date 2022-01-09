custom_imports = dict(imports=['my_codes'], allow_failed_imports=False)

d_reg_interval = 16
g_reg_interval = 4

g_reg_ratio = g_reg_interval / (g_reg_interval + 1)
d_reg_ratio = d_reg_interval / (d_reg_interval + 1)

model = dict(
    type='StaticUnconditionalGAN',
    generator=dict(
        type='ConditionalStyleGANv2Generator',
        out_size=64,
        style_channels=512,
    ),
    discriminator=dict(
        # type='ConditionalStyleGAN2Discriminator',
        # in_channels=512,
        type='StyleGAN2Discriminator',
        in_size=64,
    ),
    gan_loss=dict(type='GANLoss', gan_type='wgan-logistic-ns'),
    disc_auxiliary_loss=dict(
        type='R1GradientPenalty',
        loss_weight=10. / 2. * d_reg_interval,
        interval=d_reg_interval,
        norm_mode='HWC',
        data_info=dict(real_data='real_imgs', discriminator='disc')),
    gen_auxiliary_loss=dict(
        type='GeneratorPathRegularizer',
        loss_weight=2. * g_reg_interval,
        pl_batch_shrink=2,
        interval=g_reg_interval,
        data_info=dict(generator='gen', num_batches='batch_size'))
)

train_cfg = dict(use_ema=True)
test_cfg = None

# Style-based GANs do not perform any augmentation for the LSUN datasets
dataset_type = 'UnconditionalImageDataset'

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='real_img',
        io_backend='disk',
    ),
    dict(type='Resize', keys=['real_img'], scale=(64, 64), keep_ratio=False),
    dict(
        type='Normalize',
        keys=['real_img'],
        mean=[127.5] * 3,
        std=[127.5] * 3,
        to_rgb=False),
    dict(type='ImageToTensor', keys=['real_img']),
    dict(type='Collect', keys=['real_img'], meta_keys=['real_img_path'])
]

val_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='real_img',
        io_backend='disk',
    ),
    dict(type='Resize', keys=['real_img'], scale=(64, 64), keep_ratio=False),
    dict(
        type='Normalize',
        keys=['real_img'],
        mean=[127.5] * 3,
        std=[127.5] * 3,
        to_rgb=True),
    dict(type='ImageToTensor', keys=['real_img']),
    dict(type='Collect', keys=['real_img'], meta_keys=['real_img_path'])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=100,
        dataset=dict(
            type=dataset_type, imgs_root='./mnist_test', pipeline=train_pipeline)),
    val=dict(type=dataset_type, imgs_root='./mnist_test', pipeline=val_pipeline))


ema_half_life = 10.
custom_hooks = [
    dict(
        type='VisualizeConditionalSamples',
        output_dir='training_samples',
        interval=500),
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('generator_ema', ),
        interval=1,
        interp_cfg=dict(momentum=0.5**(32. / (ema_half_life * 1000.))),
        priority='VERY_HIGH')
]

# define optimizer
optimizer = dict(
    generator=dict(
        type='Adam', lr=0.002 * g_reg_ratio, betas=(0, 0.99**g_reg_ratio)),
    discriminator=dict(
        type='Adam', lr=0.002 * d_reg_ratio, betas=(0, 0.99**d_reg_ratio)))

lr_config = None
total_iters = 800002  # need to modify

metrics = dict(
    fid50k=dict(
        type='FID', num_images=50000, inception_pkl=None, bgr2rgb=True),
    pr50k3=dict(type='PR', num_images=50000, k=3),
    ppl_wend=dict(type='PPL', space='W', sampling='end', num_images=50000))

# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
# yapf:enable
checkpoint_config = dict(interval=10000, by_epoch=False, max_keep_ckpts=30)
# use dynamic runner
runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=True,
    pass_training_status=True)

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 10000)]
find_unused_parameters = True
cudnn_benchmark = True


