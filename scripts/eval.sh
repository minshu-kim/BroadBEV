CUDA_VISIBLE_DEVICES=0,1,2,3 torchpack dist-run -np 4 python tools/test.py \
 configs/nuscenes/seg/BroadBEV.yaml \
 pretrained/BroadBEV.pth \
 --eval map
# --night true
# --rainy true
