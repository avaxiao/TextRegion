_base_ = '../base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_context59.txt',
)

# dataset settings
dataset_type = 'PascalContext59Dataset'
data_root = '/shared/nas2/yaox11/data/datasets/VOCdevkit/VOC2010'


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2016, 672), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]


test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClassContext'),
        ann_file='ImageSets/SegmentationContext/val.txt',
        pipeline=test_pipeline))