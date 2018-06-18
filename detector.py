from __future__ import division

import time
import os
import sys
import argparse
import pickle as pkl
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import pandas as pd

import cv2

from util import *
from darknet import Darknet


def parse_args():

    parser = argparse.ArgumentParser(description="YOLO v3 detection module")

    parser.add_argument("--images", dest="images", 
                        help="Image / Directory containing images to perform detection upon",
                        default="imgs", type=str)
    
    parser.add_argument("--det", dest = 'det', 
                        help="Image / Directory to store detections to",
                        default="det", type=str)

    parser.add_argument("--bs", dest="bs", 
                        help="Batch size", default=1)

    parser.add_argument("--confidence", dest = "confidence", 
                        help="Object Confidence to filter predictions", default = 0.5)
    
    parser.add_argument("--nms_thresh", dest="nms_thresh", 
                        help="NMS Threshhold", default = 0.4)
    
    parser.add_argument("--cfg", dest='cfgfile', 
                        help="Config file",
                        default="cfg/yolov3.cfg", type=str)
    
    parser.add_argument("--weights", dest = 'weightsfile', 
                        help="weightsfile",
                        default = "yolov3.weights", type = str)
    
    parser.add_argument("--reso", dest='reso', 
                        help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)

    return parser.parse_args()

def load_classes(namesfile):
    with open(namesfile, "r") as fp:
        names = fp.read().split("\n")[:-1]
        return names

args = parse_args()
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thresh = float(args.nms_thresh)
start = 0

# Device setup
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("\nUsing", device, "\n")

num_classes = 80  # For COCO
classes = load_classes("data/coco.names")

# NN setup
print("Loading net: ...")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Net loaded.")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32

# Give model to desired device
model.to(device)
# Evaluation mode since we are using pretrained weights
model.eval()

# Data loading time
read_dir = time.time()
# Detection phase
try:
    imlist = []

### TODO .............


