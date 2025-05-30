_base_ = '../base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_voc21.txt',
    prob_thd=0.4,

)

# dataset settings
dataset_type = 'PascalVOCDataset'
data_root = '/shared/nas2/yaox11/data/datasets/VOCdevkit/VOC2012'


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1008, 672), keep_ratio=True),
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
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='ImageSets/Segmentation/val.txt',
        pipeline=test_pipeline))