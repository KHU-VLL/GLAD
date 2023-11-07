model = dict(
    type='DARecognizer3D',
    backbone=dict(
        type='ResNet3d',
        pretrained2d=False,
        pretrained=None,
        depth=50,
        conv1_kernel=(5, 7, 7),
        conv1_stride_t=2,
        pool1_stride_t=2,
        conv_cfg=dict(type='Conv3d'),
        norm_eval=False,
        inflate=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
        zero_init_residual=False,
        non_local=((0, 0, 0), (0, 1, 0, 1), (0, 1, 0, 1, 0, 1), (0, 0, 0)),
        non_local_cfg=dict(
            sub_sample=True,
            use_scale=False,
            norm_cfg=dict(type='BN3d', requires_grad=True),
            mode='dot_product')),
    cls_head=dict(
        type='IdentityHead',
        num_classes=12,
        in_channels=2048,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01),
    train_cfg=None,
    test_cfg=dict(average_clips='prob'),
    samplings=dict(source='dense', target='dense'),
    neck=dict(
        type='TOLN',
        in_channels=2048,
        num_clips=3,
        backbone='TimeSformer',
        num_segments=8))
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(policy='step', step=[285])
total_epochs = 300
checkpoint_config = dict(interval=25)
log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
dataset_type = 'RawframeDataset'
data_root = 'data/k400/rawframes_resized/train'
data_root_val = 'data/k400/rawframes_resized/val'
data_root_test = 'data/babel'
ann_file_train = 'data/filelists/k400/filelist_k400_train_closed.txt'
ann_file_val = 'data/filelists/k400/filelist_k400_val_closed.txt'
ann_file_test = 'data/filelists/babel/filelist_babel_test_closed.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.8),
        random_crop=False,
        max_wh_scale_gap=0),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=32,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='TCNHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=18,
    workers_per_gpu=8,
    test_dataloader=dict(videos_per_gpu=1),
    train=[
        dict(
            type='RawframeDataset',
            filename_tmpl='img_{:05d}.jpg',
            start_index=0,
            data_prefix='data/k400/rawframes_resized/train',
            ann_file=
            'data/filelists/k400/filelist_k400_train_closed.txt',
            pipeline=[
                dict(
                    type='SampleFrames',
                    clip_len=8,
                    frame_interval=2,
                    num_clips=3),
                dict(type='RawFrameDecode'),
                dict(type='Resize', scale=(-1, 256)),
                dict(type='Resize', scale=(224, 224), keep_ratio=False),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_bgr=False),
                dict(type='FormatShape', input_format='NCTHW'),
                dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
                dict(type='ToTensor', keys=['imgs', 'label'])
            ]),
        dict(
            type='RawframeDataset',
            filename_tmpl='img_{:05d}.jpg',
            with_offset=True,
            start_index=1,
            data_prefix='data/babel',
            ann_file=
            'data/filelists/babel/filelist_babel_train_closed.txt',
            pipeline=[
                dict(
                    type='SampleFrames',
                    clip_len=8,
                    frame_interval=1,
                    num_clips=3),
                dict(type='RawFrameDecode'),
                dict(type='Resize', scale=(224, 224), keep_ratio=False),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_bgr=False),
                dict(type='FormatShape', input_format='NCTHW'),
                dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
                dict(type='ToTensor', keys=['imgs', 'label'])
            ])
    ],
    val=dict(
        type='RawframeDataset',
        start_index=0,
        ann_file='data/filelists/k400/filelist_k400_val_closed.txt',
        data_prefix='data/k400/rawframes_resized/val',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=32,
                frame_interval=2,
                num_clips=1,
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]),
    test=dict(
        type='RawframeDataset',
        start_index=1,
        ann_file=
        'data/filelists/babel/filelist_babel_test_closed.txt',
        with_offset=True,
        data_prefix='data/babel',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=1,
                frame_interval=1,
                num_clips=32,
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='TCNHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]))
evaluation = dict(
    interval=5,
    metrics=['top_k_accuracy', 'mean_class_accuracy', 'confusion_matrix'],
    save_best='mean_class_accuracy')
work_dir = 'work_dirs/tol'
load_from = 'https://download.openmmlab.com/mmaction/v1.0/recognition/i3d/i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_k400-rgb/i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_k400-rgb_20220812-8e1f2148.pth'
domain_adaptation = True
datasets = dict(
    BABEL=dict(
        type='RawframeDataset',
        filename_tmpl='img_{:05d}.jpg',
        with_offset=True,
        start_index=1),
    K400=dict(
        type='RawframeDataset', filename_tmpl='img_{:05d}.jpg', start_index=0))
dataset_settings = dict(
    source=dict(
        train=dict(
            type='RawframeDataset',
            filename_tmpl='img_{:05d}.jpg',
            start_index=0,
            data_prefix='data/k400/rawframes_resized/train',
            ann_file=
            'data/filelists/k400/filelist_k400_train_closed.txt')),
    target=dict(
        train=dict(
            type='RawframeDataset',
            filename_tmpl='img_{:05d}.jpg',
            with_offset=True,
            start_index=1,
            data_prefix='data/babel',
            ann_file=
            'data/filelists/babel/filelist_babel_train_closed.txt'))
)
pipelines = dict(
    source=dict(train=[
        dict(type='SampleFrames', clip_len=8, frame_interval=2, num_clips=3),
        dict(type='RawFrameDecode'),
        dict(type='Resize', scale=(-1, 256)),
        dict(type='Resize', scale=(224, 224), keep_ratio=False),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_bgr=False),
        dict(type='FormatShape', input_format='NCTHW'),
        dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs', 'label'])
    ]),
    target=dict(train=[
        dict(type='SampleFrames', clip_len=8, frame_interval=1, num_clips=3),
        dict(type='RawFrameDecode'),
        dict(type='Resize', scale=(224, 224), keep_ratio=False),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_bgr=False),
        dict(type='FormatShape', input_format='NCTHW'),
        dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs', 'label'])
    ]))
gpu_ids = range(0, 8)
omnisource = False
module_hooks = []
