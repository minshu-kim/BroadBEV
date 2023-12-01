CUDA_VISIBLE_DEVICES=$1 torchpack dist-run -np 1 python tools/visualize.py \
 configs/nuscenes/seg/BroadBEV.yaml \
 --checkpoint pretrained/BroadBEV.pth \
  --out-dir viz/ --mode pred --bbox-score 0.1
