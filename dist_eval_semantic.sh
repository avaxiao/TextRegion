LOCAL_RANK=$1
CONFIG=$2
WORK_DIR=$3

CLIP_PRETRAIN=$4
CLIP_ARCHITECTURE=$5


CUDA_VISIBLE_DEVICES=$LOCAL_RANK python eval_semantic.py --config $CONFIG --work-dir $WORK_DIR --local-rank $LOCAL_RANK  \
    --clip_pretrained $CLIP_PRETRAIN --clip_architecture $CLIP_ARCHITECTURE