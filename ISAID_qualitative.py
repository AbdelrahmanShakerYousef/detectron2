# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets import load_coco_json


image_id = "P1854_0_800_6600_7400.png" #"P0003_0_800_0_800.png"

# Read the testing image
im = cv2.imread("/l/users/miriam.cristofoletti/CV703-project/iSAID/val/images/" + image_id)

# Register COCO images
register_coco_instances("iSAID_train", {}, 
                        "/l/users/miriam.cristofoletti/CV703-project/iSAID/train/instancesonly_filtered_train.json",
                        "/l/users/miriam.cristofoletti/CV703-project/iSAID/train/images/")
register_coco_instances("iSAID_val", {}, 
                        "/l/users/miriam.cristofoletti/CV703-project/iSAID/val/instancesonly_filtered_val.json",
                        "/l/users/miriam.cristofoletti/CV703-project/iSAID/val/images/")

# ISAID Classes
MetadataCatalog.get("iSAID_train").thing_classes = ["ship", "storage_tank", "baseball_diamond", "tennis_court" ,"basketball_court",
                                                    "Ground_Track_Field" ,"Bridge", "Large_Vehicle" ,"Small_Vehicle", "Helicopter",
                                                    "Swimming_pool", "Roundabout" ,"Soccer_ball_field", "plane" , "Harbor" ]
               
cfg = get_cfg()

# The folder of the model
cfg.OUTPUT_DIR = 'output_maskrcnn_updated_hyperparameters_resnet_101'

cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 15

# Predict the masks
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

# Visualize the image
v = Visualizer(im, MetadataCatalog.get(cfg.DATASETS.TRAIN[1]), scale=1.5)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

print("The output is ",outputs)
cv2.imwrite(image_id,v.get_image())
