from __future__ import division

import time
import os
import sys
import argparse
import pickle as pkl
import random

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import pandas as pd

import cv2

from util import *
from darknet import Darknet


def parse_args():

    parser = argparse.ArgumentParser(description="YOLO v3 detection module")

    parser.add_argument("--gpu", dest="cuda_id", 
                        help="GPU id for single GPU inference. Checkout nvidia-smi first.",
                        default=False)

    parser.add_argument("--video", dest="video_file",
                        help="Specify the video file to use.",
                        default="video.avi", type=str)

    parser.add_argument("--bs", dest="bs", 
                        help="Batch size", 
                        default=1)

    parser.add_argument("--confidence", dest = "confidence", 
                        help="Object Confidence to filter predictions", 
                        default=0.5)
    
    parser.add_argument("--nms_thresh", dest="nms_thresh", 
                        help="NMS Threshhold", 
                        default=0.4)
    
    parser.add_argument("--cfg", dest='cfgfile', 
                        help="Config file",
                        default="cfg/yolov3.cfg", type=str)
    
    parser.add_argument("--weights", dest = 'weightsfile', 
                        help="weightsfile",
                        default="yolov3.weights", type = str)
    
    parser.add_argument("--reso", dest='reso', 
                        help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)

    return parser.parse_args()


def write(x, results):
	# Draw box & class
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results
    cls = int(x[-1])
    #color = random.choice(colors)  # Looks really bad on videos
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2, colors[cls], 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, colors[cls], -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)
    return img

# Main code
args = parse_args()
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thresh = float(args.nms_thresh)
start = 0

# Device setup
device = torch.device("cuda:" + args.cuda_id if args.cuda_id and torch.cuda.is_available() else "cpu")

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
video_file = args.video_file

if video_file == "webc":
	cap = cv2.VideoCapture(0)
else:
	cap = cv2.VideoCapture(video_file)


assert cap.isOpened(), "Cannot capture video/webcam feed."

frames = 0
start = time.time()
colors = pkl.load(open("pallete", "rb"))


while cap.isOpened():
	cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
	cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
	ret, frame = cap.read()

	if ret:
		img = prep_image(frame, inp_dim)
		im_dim = frame.shape[1], frame.shape[0]
		im_dim = torch.FloatTensor(im_dim).repeat(1, 2)

		im_dim = im_dim.to(device)
		img = img.to(device)

		with torch.no_grad():
		    output = model(img, device)
		output = write_results(output, confidence, num_classes, device=device, nms_conf=nms_thresh)

		if output.shape[0] == 0:
			continue

		if type(output) == int:
		    frames += 1
		    print("FPS of the video is {:5.4f}".format(frames / (time.time() - start)))
		    cv2.imshow("frame", frame)
		    key = cv2.waitKey(1)
		    if key & 0xFF == ord('q'):
		        break
		    continue

		im_dim = im_dim.repeat(output.size(0), 1)
		scaling_factor = torch.min(416 / im_dim, 1)[0].view(-1, 1)

		output[:, [1, 3]] -= (inp_dim - scaling_factor*im_dim[:, 0].view(-1, 1)) / 2
		output[:, [2, 4]] -= (inp_dim - scaling_factor*im_dim[:, 1].view(-1, 1)) / 2
		output[:, 1:5] /= scaling_factor

		for i in range(output.shape[0]):
		    output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
		    output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1]) 
		classes = load_classes("data/coco.names")
		
		list(map(lambda x: write(x, frame), output))

		cv2.imshow("frame", frame)
		key = cv2.waitKey(1)
		if key & 0xFF == ord('q'):
		    break
		frames += 1
		#print(time.time() - start)
		print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
		torch.cuda.synchronize()        
	else:
	    break


if device != "cpu":
    torch.cuda.empty_cache()
