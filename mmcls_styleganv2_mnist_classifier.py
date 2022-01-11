custom_imports = dict(imports=['my_codes'], allow_failed_imports=False)

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        in_channels=1,
        depth=18,
        num_stages=4,
        out_indices=(3,),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

# dataset setting
dataset_type = 'CustomMNIST'
img_norm_cfg = dict(mean=[128], std=[128], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(type='Resize', size=(32, 32)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(type=dataset_type,
               data_prefix='./datasets/generated_mnist',
               ann_file='./datasets/generated_mnist/annotations.txt',
               pipeline=train_pipeline),
    val=dict(type=dataset_type,
             data_prefix='./datasets/mnist_test',
             ann_file='./datasets/mnist_test/annotations.txt',
             pipeline=test_pipeline,
             test_mode=True),
    test=dict(type=dataset_type,
              data_prefix='./datasets/mnist_test',
              ann_file='./datasets/mnist_test/annotations.txt',
              pipeline=test_pipeline,
              test_mode=True))

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[10, 15])
runner = dict(type='EpochBasedRunner', max_epochs=20)

# checkpoint saving
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
