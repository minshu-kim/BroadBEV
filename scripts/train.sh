CUDA_VISIBLE_DEVICES=0,1,2,3 torchpack dist-run -np 4 python tools/train.py \
 configs/nuscenes/seg/BroadBEV.yaml \
 --run_dir runs/BroadBEV/ \
 --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth
