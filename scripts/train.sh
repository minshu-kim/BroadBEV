CUDA_VISIBLE_DEVICES=$1 torchpack dist-run -np 2 python tools/train.py \
 configs/nuscenes/seg/BroadBEV.yaml \
 --run_dir runs/BroadBEV/ \
 --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth
