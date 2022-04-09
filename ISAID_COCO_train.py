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

register_coco_instances("coco_2017_train", {}, 
                        "/l/users/abdelrahman.youssief/COCO/annotations/instances_train2017.json",
                        "/l/users/abdelrahman.youssief/COCO/train2017/")
register_coco_instances("coco_2017_val", {}, 
                        "/l/users/abdelrahman.youssief/COCO/annotations/instances_val2017.json",
                        "/l/users/abdelrahman.youssief/COCO/train2017/")
                        

cfg = get_cfg()
cfg.OUTPUT_DIR = 'output_fasterrcnn_COCO_pretrain'


# TODO: load ConvNExt pretraining ImageNEt and train it on COCO 
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("coco_2017_train")
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2


#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo

cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.01  # pick a good LR
cfg.SOLVER.MAX_ITER = 100000 
cfg.SOLVER.STEPS = [60000,80000]        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80  


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

evaluator = COCOEvaluator("coco_2017_val")
val_loader = build_detection_test_loader(cfg, "coco_2017_val")
print(inference_on_dataset(trainer.model, val_loader, evaluator))
# another equivalent way to evaluate the model is to use `trainer.test`