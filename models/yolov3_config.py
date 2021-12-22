checkpoint_config = dict(interval=1, max_keep_ckpts=3)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook')
    ])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
model = dict(
    type='YOLOV3',
    backbone=dict(
        type='MobileNetV2',
        out_indices=(2, 4, 6),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://mmdet/mobilenet_v2')),
    neck=dict(
        type='YOLOV3Neck',
        num_scales=3,
        in_channels=[320, 96, 32],
        out_channels=[96, 96, 96]),
    bbox_head=dict(
        type='YOLOV3Head',
        num_classes=2,
        in_channels=[96, 96, 96],
        out_channels=[96, 96, 96],
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(220, 125), (128, 222), (264, 266)],
                        [(35, 87), (102, 96), (60, 170)],
                        [(10, 15), (24, 36), (72, 42)]],
            strides=[32, 16, 8]),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[32, 16, 8],
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_conf=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_xy=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=2.0,
            reduction='sum'),
        loss_wh=dict(type='MSELoss', loss_weight=2.0, reduction='sum')),
    train_cfg=dict(
        assigner=dict(
            type='GridAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        conf_thr=0.005,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=100))
dataset_type = 'CocoDataset'
data_root = '/opt/ml/data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMyImageAnnoFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='MinIoURandomCrop', min_ious=(0.8, 0.9), min_crop_size=0.3),
    dict(
        type='Resize',
        img_scale=[(1024, 1024)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='Albu',
        transforms=[
            dict(
                type='OpticalDistortion',
                distort_limit=0.3,
                shift_limit=0.0,
                interpolation=1,
                border_mode=0,
                value=(255, 255, 255),
                p=1),
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0,
                scale_limit=0.0,
                rotate_limit=5,
                interpolation=1,
                border_mode=0,
                value=(255, 255, 255),
                p=0.4),
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
        keymap=dict(img='image', gt_semantic_seg='mask')),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=24,
    workers_per_gpu=4,
    train=dict(
        type='CocoDataset',
        classes=['correct', 'uncorrect'],
        ann_file='/opt/ml/data/train_coco.json',
        img_prefix='/opt/ml/data/detection/',
        pipeline=[
            dict(type='LoadMyImageAnnoFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='MinIoURandomCrop',
                min_ious=(0.8, 0.9),
                min_crop_size=0.3),
            dict(
                type='Resize',
                img_scale=[(1024, 1024)],
                multiscale_mode='range',
                keep_ratio=True),
            dict(
                type='Albu',
                transforms=[
                    dict(
                        type='OpticalDistortion',
                        distort_limit=0.3,
                        shift_limit=0.0,
                        interpolation=1,
                        border_mode=0,
                        value=(255, 255, 255),
                        p=1),
                    dict(
                        type='ShiftScaleRotate',
                        shift_limit=0.0,
                        scale_limit=0.0,
                        rotate_limit=5,
                        interpolation=1,
                        border_mode=0,
                        value=(255, 255, 255),
                        p=0.4),
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
                keymap=dict(img='image', gt_semantic_seg='mask')),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='CocoDataset',
        classes=['correct', 'uncorrect'],
        ann_file='/opt/ml/data/valid_coco.json',
        img_prefix='/opt/ml/data/detection/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        classes=['correct', 'uncorrect'],
        ann_file='/opt/ml/data/valid_coco.json',
        img_prefix='/opt/ml/data/detection/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
optimizer = dict(type='SGD', lr=0.003, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=80,
    warmup_ratio=0.0001,
    step=[50, 58])
runner = dict(type='EpochBasedRunner', max_epochs=66)
evaluation = dict(interval=1, metric=['bbox'])
find_unused_parameters = True
NAME = 'yolov3_handwriting_albu_opticalvalue255_randomhwn'
CLASSES = ['correct', 'uncorrect']
albu_train_transforms = [
    dict(
        type='OpticalDistortion',
        distort_limit=0.3,
        shift_limit=0.0,
        interpolation=1,
        border_mode=0,
        value=(255, 255, 255),
        p=1),
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0,
        scale_limit=0.0,
        rotate_limit=5,
        interpolation=1,
        border_mode=0,
        value=(255, 255, 255),
        p=0.4),
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
work_dir = './work_dirs/yolov3_handwriting_albu_opticalvalue255_randomhwn'
gpu_ids = range(0, 1)