# check pytorch installation: 
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

from detectron2.data.datasets import register_coco_instances
register_coco_instances("iSAID_train", {}, 
                        "/apps/local/shared/CV703/datasets/iSAID/iSAID_patches/train/instancesonly_filtered_train.json",
                        "/apps/local/shared/CV703/datasets/iSAID/iSAID_patches/train/images/")
register_coco_instances("iSAID_val", {}, 
                        "/apps/local/shared/CV703/datasets/iSAID/iSAID_patches/val/instancesonly_filtered_val.json",
                        "/apps/local/shared/CV703/datasets/iSAID/iSAID_patches/val/images/")
                        

cfg = get_cfg()
cfg.OUTPUT_DIR = 'output_fasterrcnn'
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("iSAID_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 100000 
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 15  


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
