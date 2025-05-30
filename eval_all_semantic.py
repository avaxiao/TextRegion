import os
import pytz
from datetime import datetime


local_rank = 0
clip_pretrained = 'laion2b_s32b_b79k'  # 'openai' or 'laion2b_s32b_b79k'
clip_architecture = 'ViT-H/14' # 'ViT-B/16', 'ViT-H/14'


chicagoTz = pytz.timezone("America/Chicago")
timeInChicago = datetime.now(chicagoTz)
flag = timeInChicago.strftime("%m_%d_%Y_%H_%M")

work_dir = f'./eval_results/semantic_{clip_pretrained}_{clip_architecture.replace("/", "-")}/{flag}'
os.makedirs(work_dir, exist_ok=True)


configs_list = [
    f'./configs/cfg_ds/cfg_voc21.py',
    f'./configs/cfg_ds/cfg_context60.py',
    f'./configs/cfg_ds/cfg_coco_object.py',
    f'./configs/cfg_ds/cfg_voc20.py',
    f'./configs/cfg_ds/cfg_context59.py',
    f'./configs/cfg_ds/cfg_coco_stuff164k.py',
    f'./configs/cfg_ds/cfg_city_scapes.py',
    f'./configs/cfg_ds/cfg_ade20k.py'
]


for config in configs_list:
    os.system(f"bash ./dist_eval_semantic.sh {local_rank} {config} {work_dir} {clip_pretrained} {clip_architecture}")
    print(f"Local_rank={local_rank}: Finish running {config} for {clip_pretrained} {clip_architecture}")
    print(f"Save results to {work_dir}")