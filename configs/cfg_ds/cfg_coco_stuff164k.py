_base_ = '../base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_coco_stuff.txt',
)

# dataset settings
dataset_type = 'COCOStuffDataset'
data_root = '/shared/nas2/yaox11/data/datasets/coco_stuff164k'


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1334, 896), keep_ratio=True),
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
            img_path='images/val2017', seg_map_path='annotations/val2017'),
        pipeline=test_pipeline))