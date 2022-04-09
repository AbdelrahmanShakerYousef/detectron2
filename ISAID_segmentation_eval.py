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
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

register_coco_instances("iSAID_train", {}, 
                        "/l/users/miriam.cristofoletti/CV703-project/iSAID/train/instancesonly_filtered_train.json",
                        "/l/users/miriam.cristofoletti/CV703-project/iSAID/train/images/")
register_coco_instances("iSAID_val", {}, 
                        "/l/users/miriam.cristofoletti/CV703-project/iSAID/val/instancesonly_filtered_val.json",
                        "/l/users/miriam.cristofoletti/CV703-project/iSAID/val/images/")
                        

cfg = get_cfg()
cfg.OUTPUT_DIR = 'output_maskrcnn_batch4_lr002'

print(cfg.OUTPUT_DIR)
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("iSAID_train")
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2


# Training from Scratch !!
#cfg.MODEL.WEIGHTS = ""
#cfg.MODEL.BACKBONE.FREEZE_AT = 0
#cfg.MODEL.RESNETS.NORM = "GN"
#cfg.MODEL.FPN.NORM = "GN"

 
#Comment this line to pretrain on ImageNet only
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo

cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.02  # pick a good LR
cfg.SOLVER.MAX_ITER = 100000 
cfg.SOLVER.STEPS = [60000, 80000]        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 15  


# Freezing stages: `1` means freezing the stem. `2` means freezing the stem and one residual stage, etc.
#cfg.MODEL.BACKBONE.FREEZE_AT = 3
#print("Freezing two layers")
#os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
#cfg.MODEL.BACKBONE.FREEZE_AT = 0

print(cfg)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained


cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold : Need to be studies!!!

predictor = DefaultPredictor(cfg)

evaluator = COCOEvaluator("iSAID_val")
val_loader = build_detection_test_loader(cfg, "iSAID_val")
print(inference_on_dataset(trainer.model, val_loader, evaluator))
# another equivalent way to evaluate the model is to use `trainer.test`