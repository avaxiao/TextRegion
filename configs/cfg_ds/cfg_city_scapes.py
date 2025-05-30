_base_ = '../base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_city_scapes.txt',
    sam2_cfg=dict(
            points_per_side=36,
        ),
)

# dataset settings
dataset_type = 'CityscapesDataset'
data_root = '/shared/nas2/yaox11/data/datasets/cityscapes'


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2688, 1344), keep_ratio=True),

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
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        pipeline=test_pipeline))