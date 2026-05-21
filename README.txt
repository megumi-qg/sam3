CUDA_VISIBLE_DEVICES=2,3 nohup python sam3/train/train.py \
    -c configs/acdc/scribble_sam3_tracker_image_init_v1.yaml \
    --use-cluster 0 \
    --num-gpus 2 \
    > scribble_sam3_tracker_image_init_v1.log 2>&1 &