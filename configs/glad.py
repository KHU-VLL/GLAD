model = dict(
    type='TemporallyPyramidicDARecognizer',
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
        type='I3DHead',
        num_classes=12,
        in_channels=2048,
        spatial_type=None,
        dropout_ratio=0.5,
        init_std=0.01),
    train_cfg=None,
    test_cfg=dict(average_clips='prob'),
    dim='3d',
    sampler_name='lngn',
    sampler_index=dict(l=[4, 5], g=3),
    neck=[
        dict(
            type='TemporallyPyramidicDomainClassifier',
            temporal_locality='local',
            fusion_method='mean',
            loss_weight=1.0,
            in_channels=2048,
            hidden_dim=2048,
            nickname='l',
            num_layers=4,
            dropout_ratio=0.5,
            backbone='TimeSformer'),
        dict(
            type='TemporallyPyramidicDomainClassifier',
            temporal_locality='global',
            fusion_method='mean',
            loss_weight=1,
            in_channels=2048,
            hidden_dim=2048,
            nickname='g',
            num_layers=4,
            dropout_ratio=0.5,
            backbone='TimeSformer'),
        dict(
            type='TemporallyPyramidicDomainClassifier',
            temporal_locality='cross',
            loss_weight=1.0,
            in_channels=2048,
            hidden_dim=2048,
            nickname='x',
            num_layers=4,
            dropout_ratio=0.5,
            backbone='TimeSformer')
    ])
optimizer = dict(
    type='SGD',
    lr=0.000875,
    momentum=0.9,
    weight_decay=0.0001,
    constructor='TSMOptimizerConstructor',
    paramwise_cfg=dict(fc_lr5=True))
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(policy='step', step=[8, 16])
total_epochs = 3
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
domain_adaptation = True
find_unused_parameters = False
domain_classifier_common_options = dict(
    num_layers=4, dropout_ratio=0.5, backbone='TimeSformer')
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
            'data/filelists/k400/filelist_k400_train_closed.txt'),
        test=dict(
            type='RawframeDataset',
            filename_tmpl='img_{:05d}.jpg',
            start_index=0,
            test_mode=True,
            data_prefix='data/k400/rawframes_resized/val',
            ann_file=
            'data/filelists/k400/filelist_k400_test_closed.txt')),
    target=dict(
        train=dict(
            type='RawframeDataset',
            filename_tmpl='img_{:05d}.jpg',
            with_offset=True,
            start_index=1,
            data_prefix='data/babel',
            ann_file=
            'data/filelists/babel/filelist_babel_train_closed.txt'),
        valid=dict(
            type='RawframeDataset',
            filename_tmpl='img_{:05d}.jpg',
            with_offset=True,
            start_index=1,
            test_mode=True,
            data_prefix='data/babel',
            ann_file=
            'data/filelists/babel/filelist_babel_val_closed.txt'),
        test=dict(
            type='RawframeDataset',
            filename_tmpl='img_{:05d}.jpg',
            with_offset=True,
            start_index=1,
            test_mode=True,
            data_prefix='data/babel',
            ann_file=
            'data/filelists/babel/filelist_babel_test_closed.txt')))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
pipelines = dict(
    source=dict(train=[
        dict(type='SampleFrames', clip_len=8, frame_interval=2, num_clips=8),
        dict(type='RawFrameDecode'),
        dict(type='Resize', scale=(-1, 256)),
        dict(
            type='MultiScaleCrop',
            input_size=224,
            scales=(1, 0.875, 0.66),
            random_crop=False,
            max_wh_scale_gap=1,
            num_fixed_crops=13),
        dict(type='Resize', scale=(224, 224), keep_ratio=False),
        dict(type='Flip', flip_ratio=0.5),
        dict(
            type='BackgroundBlend',
            p=0.25,
            alpha=0.75,
            resize_h=256,
            crop_size=224,
            ann_files=[
                'data/filelists/k400/filelist_k400_train_closed.txt'
            ],
            data_prefixes=['data/median/k400/train'],
            blend_label=False),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_bgr=False),
        dict(type='FormatShape', input_format='NCTHW'),
        dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs', 'label'])
    ]),
    target=dict(
        train=[
            dict(
                type='SampleFrames',
                clip_len=8,
                frame_interval=2,
                num_clips=8,
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(
                type='MultiScaleCrop',
                input_size=224,
                scales=(1, 0.875, 0.66),
                random_crop=False,
                max_wh_scale_gap=1,
                num_fixed_crops=13),
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
        ],
        valid=[
            dict(
                type='SampleFrames',
                clip_len=8,
                frame_interval=2,
                num_clips=8,
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
            dict(type='ToTensor', keys=['imgs', 'label'])
        ],
        test=[
            dict(
                type='SampleFrames',
                clip_len=8,
                frame_interval=2,
                num_clips=8,
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
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]))
data = dict(
    videos_per_gpu=4,
    workers_per_gpu=8,
    val_dataloader=dict(videos_per_gpu=4),
    test_dataloader=dict(videos_per_gpu=4),
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
                    num_clips=8),
                dict(type='RawFrameDecode'),
                dict(type='Resize', scale=(-1, 256)),
                dict(
                    type='MultiScaleCrop',
                    input_size=224,
                    scales=(1, 0.875, 0.66),
                    random_crop=False,
                    max_wh_scale_gap=1,
                    num_fixed_crops=13),
                dict(type='Resize', scale=(224, 224), keep_ratio=False),
                dict(type='Flip', flip_ratio=0.5),
                dict(
                    type='BackgroundBlend',
                    p=0.25,
                    alpha=0.75,
                    resize_h=256,
                    crop_size=224,
                    ann_files=[
                        'data/filelists/k400/filelist_k400_train_closed.txt'
                    ],
                    data_prefixes=['data/median/k400/train'],
                    blend_label=False),
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
                    frame_interval=2,
                    num_clips=8,
                    test_mode=True),
                dict(type='RawFrameDecode'),
                dict(type='Resize', scale=(-1, 256)),
                dict(
                    type='MultiScaleCrop',
                    input_size=224,
                    scales=(1, 0.875, 0.66),
                    random_crop=False,
                    max_wh_scale_gap=1,
                    num_fixed_crops=13),
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
            ])
    ],
    val=dict(
        type='RawframeDataset',
        filename_tmpl='img_{:05d}.jpg',
        with_offset=True,
        start_index=1,
        test_mode=True,
        data_prefix='data/babel',
        ann_file=
        'data/filelists/babel/filelist_babel_val_closed.txt',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=8,
                frame_interval=2,
                num_clips=8,
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
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]),
    test=dict(
        type='RawframeDataset',
        filename_tmpl='img_{:05d}.jpg',
        with_offset=True,
        start_index=1,
        test_mode=True,
        data_prefix='data/babel',
        ann_file=
        'data/filelists/babel/filelist_babel_test_closed.txt',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=8,
                frame_interval=2,
                num_clips=8,
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
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]))
evaluation = dict(
    interval=1,
    metrics=['top_k_accuracy', 'mean_class_accuracy', 'confusion_matrix'],
    save_best='mean_class_accuracy')
lr = 0.001
work_dir = 'work_dirs/glad'
load_from = 'work_dirs/tol/epoch_300.pth'
blend_options = dict(
    p=0.25,
    alpha=0.75,
    resize_h=256,
    crop_size=224,
    ann_files=[
        'data/filelists/k400/filelist_k400_train_closed.txt'
    ],
    data_prefixes=['data/median/k400/train'],
    blend_label=False)
ckpt_revise_keys = ''
gpu_ids = range(0, 7)
omnisource = False
module_hooks = []
