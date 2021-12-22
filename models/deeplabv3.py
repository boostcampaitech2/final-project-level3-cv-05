norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', stride=(200, 200), crop_size=(512, 512)))
dataset_type = 'CustomDataset'
data_root = '/opt/ml/data/math/background/'
classes = ['Background', 'printing', 'handwriting']
palette = [[0, 0, 0], [192, 0, 128], [0, 128, 192]]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
albu_train_transforms = [
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='OpticalDistortion',
                distort_limit=0.25,
                shift_limit=0.25,
                border_mode=0,
                value=[255, 255, 255],
                p=0.5),
            dict(
                type='ElasticTransform',
                alpha=0,
                sigma=0,
                alpha_affine=30,
                border_mode=0,
                value=[255, 255, 255],
                p=0.5)
        ],
        p=0.5),
    dict(
        type='OneOf',
        transforms=[
            dict(type='RandomBrightnessContrast', p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.2)
        ],
        p=0.5),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='CLAHE', clip_limit=0.4, p=1.0)
        ],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(type='RandomGamma', p=1.0),
            dict(type='GaussNoise', p=1.0)
        ],
        p=0.3)
]
train_pipeline = [
    dict(type='DrawImageAnnFromFile', pos='train'),
    dict(type='RandomCrop', crop_size=(384, 384)),
    dict(
        type='Albu',
        transforms=[
            dict(
                type='OneOf',
                transforms=[
                    dict(
                        type='OpticalDistortion',
                        distort_limit=0.25,
                        shift_limit=0.25,
                        border_mode=0,
                        value=[255, 255, 255],
                        p=0.5),
                    dict(
                        type='ElasticTransform',
                        alpha=0,
                        sigma=0,
                        alpha_affine=30,
                        border_mode=0,
                        value=[255, 255, 255],
                        p=0.5)
                ],
                p=0.5),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='RandomBrightnessContrast', p=1.0),
                    dict(
                        type='HueSaturationValue',
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        val_shift_limit=20,
                        p=0.2)
                ],
                p=0.5),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='CLAHE', clip_limit=0.4, p=1.0)
                ],
                p=0.2),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='RandomGamma', p=1.0),
                    dict(type='GaussNoise', p=1.0)
                ],
                p=0.3)
        ],
        keymap=dict(img='image', gt_semantic_seg='mask'),
        update_pad_shape=False),
    dict(type='RandomFlip', prob=0),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=None,
        img_ratios=[2.0],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=6,
    train=dict(
        classes=['Background', 'printing', 'handwriting'],
        palette=[[0, 0, 0], [192, 0, 128], [0, 128, 192]],
        type='CustomDataset',
        reduce_zero_label=False,
        img_dir='/opt/ml/data/math/background/background_all',
        img_suffix='.png',
        ann_dir='/opt/ml/data/math/background/background_all',
        pipeline=[
            dict(type='DrawImageAnnFromFile', pos='train'),
            dict(type='RandomCrop', crop_size=(384, 384)),
            dict(
                type='Albu',
                transforms=[
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(
                                type='OpticalDistortion',
                                distort_limit=0.25,
                                shift_limit=0.25,
                                border_mode=0,
                                value=[255, 255, 255],
                                p=0.5),
                            dict(
                                type='ElasticTransform',
                                alpha=0,
                                sigma=0,
                                alpha_affine=30,
                                border_mode=0,
                                value=[255, 255, 255],
                                p=0.5)
                        ],
                        p=0.5),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(type='RandomBrightnessContrast', p=1.0),
                            dict(
                                type='HueSaturationValue',
                                hue_shift_limit=20,
                                sat_shift_limit=30,
                                val_shift_limit=20,
                                p=0.2)
                        ],
                        p=0.5),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(type='Blur', blur_limit=3, p=1.0),
                            dict(type='CLAHE', clip_limit=0.4, p=1.0)
                        ],
                        p=0.2),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(type='RandomGamma', p=1.0),
                            dict(type='GaussNoise', p=1.0)
                        ],
                        p=0.3)
                ],
                keymap=dict(img='image', gt_semantic_seg='mask'),
                update_pad_shape=False),
            dict(type='RandomFlip', prob=0),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        classes=['Background', 'printing', 'handwriting'],
        palette=[[0, 0, 0], [192, 0, 128], [0, 128, 192]],
        type='CustomDataset',
        reduce_zero_label=False,
        img_dir='/opt/ml/data/mmseg_data/syn_valid/images',
        img_suffix='.png',
        ann_dir='/opt/ml/data/mmseg_data/syn_valid/annotations',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=None,
                img_ratios=[2.0],
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        classes=['Background', 'printing', 'handwriting'],
        palette=[[0, 0, 0], [192, 0, 128], [0, 128, 192]],
        type='CustomDataset',
        reduce_zero_label=False,
        img_dir='/opt/ml/data/math/background/background_valid',
        ann_dir='/opt/ml/data/math/background/background_valid',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=None,
                img_ratios=[2.0],
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0))))
optimizer_config = dict()
lr_config = dict(
    policy='CosineRestart',
    periods=[14, 10],
    restart_weights=[1, 0.01],
    min_lr_ratio=0.01,
    by_epoch=True,
    warmup='linear',
    warmup_iters=102,
    warmup_ratio=0.001)
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(max_keep_ckpts=1, interval=1)
evaluation = dict(
    interval=1, metric='mIoU', pre_eval=True, classwise=True, save_best='mIoU')
NAME = '16_Optical_with_14'
work_dir = 'work_dirs/16_Optical_with_14'
gpu_ids = range(0, 1)