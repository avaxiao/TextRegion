# base configurations


model = dict(
    type='CLIPWithSAM2ForSegmentation',
    sam2_checkpoint="YOUR_SAM2_CHECKPOINT_PATH",
    clip_download_root="YOUR_CLIP_DOWNLOAD_ROOT",
    sam2_architecture='hiera_large',
    model_cfg='configs/sam2.1/sam2.1_hiera_l.yaml',
    sam2_cfg=dict(
        points_per_side=16,
        pred_iou_thresh=0.6,
        stability_score_thresh=0.6,
        box_nms_thresh=0.9,
        fuse_mask=True,
        fuse_mask_threshold=0.8,
    ),
    crop_size=336,
    remove_global_patch=True,
    global_patch_threshold=0.07,
    resize_method='multi_resolution',

)



test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, alpha=1.0, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', interval=5))