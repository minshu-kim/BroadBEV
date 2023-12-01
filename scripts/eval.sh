GPU=$1

CUDA_VISIBLE_DEVICES=$GPU torchpack dist-run -np 2 python tools/test.py \
 configs/nuscenes/seg/BroadBEV.yaml \
 pretrained/BroadBEV.pth \
 --eval map
# --night true
# --rainy true
