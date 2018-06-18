from __future__ import division
from __future__ import print_function  # In case python 2.7 is being used

import sys
import os
from collections import OrderedDict as OD

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import cv2

from util import *



def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416, 416))           # Resize to the net's input dimension 
    img_ = img[:, :, ::-1].transpose((2, 0, 1)) # BGR2RGB | HWC2CHW
    img_ = img_[np.newaxis, :, :, :] / 255.0    # Add a batch dimension & normalize
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    return img_

def parse_cfg(cfgFname):
    if not os.path.exists(cfgFname):
        sys.exit("Config file path incorrect")
    with open(cfgFname, 'r') as f:
        lines = f.read().split("\n")
        lines = [x for x in lines if len(x) > 0]    # Get rid of empty lines (May want to run it @ the end)
        lines = [x for x in lines if x[0] != "#"]   # Get rid of comments
        lines = [x.rstrip().lstrip() for x in lines]    # Get rid of fringe whitespaces


    block = {}  # Temporary block dict
    blocks = [] # List of all blocks

    for line in lines:
        if line[0] == "[":                      # New block 
            if len(block) != 0:                 
                blocks.append(block)            # Add previous block to blocks list.
                block = {}                      # Init a new block dict
            block['type'] = line[1:-1].rstrip()     
        else: 
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()

    blocks.append(block)
    return blocks

def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3 # Initial input filters are 3 (RGB)
    output_filters = []
    #print("Net:\n{0}".format(net_info))
    print("Number of modules in the architecture: {}".format(len(blocks[1:])))
    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()        
        
        # CONVOLUTIONAL BLOCK
        # Add the typical additional modules to a conv layer
        # like relu, ...
        if x['type'] == "convolutional":
            # Get layer info
            activation = x['activation']
            try:
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x['filters'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])
            padding = int(x['pad'])
            
            if padding:
                pad = (kernel_size - 1) // 2  # Why? 
            else:
                pad = 0

            # Add conv layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(index), conv)
            #sys.exit("Testing")
       
            # Add batchnorm layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batchnorm_{0}".format(index), bn)
            
            # Check for the correct activation to add
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)            
                module.add_module("leaky_{0}".format(index), activn)
        
        # UPSAMPLING
        elif x['type'] == "upsample":
            stride = int(x['stride'])
            upsample = nn.Upsample(scale_factor=2, mode="bilinear")
            module.add_module("upsample_{0}".format(index), upsample)
        
        # Route layer / Shortcut layers
        elif x['type'] == "route":
            x['layers'] = x['layers'].split(",")
            
            # Start of  a route    
            start = int(x['layers'][0])            
            try: # Specify end if specified
                end = int(x['layers'][1])
            except:
                end = 0

            if start > 0:
                start = start - index  # No need to use
            if end > 0:
                end = end - index  # Find the gap between current index and the layers[1]'th layer
            
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)

            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]  # When concatenating maps
            else:
                filters = output_filters[index + start]  # No concatenation requirement
        
        # Shortcut layer / skip connections
        elif x['type'] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{0}".format(index), shortcut)

        # YOLO layer
        elif x['type'] == "yolo":
            mask = x['mask'].split(",")
            mask = [int(x)  for x in mask]
           
            anchors = x['anchors'].split(",") 
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{0}".format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        # Loop end
    return (net_info, module_list)
        
            
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

class Darknet(nn.Module):
    def __init__(self, cfg_filename):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfg_filename)
        self.net_info, self.module_list = create_modules(self.blocks)

    # Device agnostic code; CUDA param replaced with device param
    def forward(self, x, device):
        
        modules = self.blocks[1:]
        """
        Since route and shortcut layers need output maps from previous layers, 
        we cache the output feature maps of every layer in a dict outputs. 
        The keys are the the indices of the layers, and the values are the feature maps.
        """
        outputs = {}
        detections = torch.Tensor().to(device)
        for i, module in enumerate(modules):
            module_type = (module['type'])
            
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)
            # Re-read the following part
            elif module_type == "route":
                layers = module['layers']
                layers = [int(a) for a in layers]
                if layers[0] > 0:
                    layers[0] -= i  # Find the gap between current index and the layers[1]'th layer          
                   
                if len(layers) == 1: # Get outputs from previous layers
                    x = outputs[i + (layers[0])]
                else:
                    if layers[1] > 0:
                        layers[1] -= i  # Find the gap between current index and the layers[1]'th layer
                    map1 = outputs[i + layers[0]]  # Gets feature maps from orig layers[0]
                    map2 = outputs[i + layers[1]]  # Gets feature maps from orig layers[1]

                    x = torch.cat((map1, map2), 1)
            elif module_type == "shortcut":
                from_ = int(module['from'])
                x = outputs[i - 1] + outputs[i + from_]

            elif module_type == "yolo":
                anchors = self.module_list[i][0].anchors
                # Get input dimensions
                inp_dim = int(self.net_info['height'])
                
                # Get the number of classes
                num_classes = int(module['classes'])

                # Transform
                x = x.data  
                x = predict_transform(x, inp_dim, anchors, num_classes, device)                
                
                detections = torch.cat((detections, x), 1)  # concatenate detections @ different scales
            # Buffer the output
            outputs[i] = x
            # End of loop

        return detections                        

    def load_weights(self, weights_file):

        with open(weights_file, "rb") as wf:
            header = np.fromfile(wf, dtype=np.int32, count=5)
            self.header = torch.from_numpy(header)
            #print(self.header, self.header.type(), self.header.shape)
            self.seen = self.header[3]

            weights = np.fromfile(wf, dtype=np.float32)

            ptr = 0  # Tracking weights arrat
            for i in range(len(self.module_list)):
                module_type = self.blocks[i + 1]["type"]

                # If module_type is convolutional load weights
                # Else ignore
                if module_type == "convolutional":
                    model = self.module_list[i]
                    try:
                        batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                    except:
                        batch_normalize = 0
                    conv = model[0]
                    if batch_normalize:
                        bn = model[1]

                        # Get the number of biases of Batch Norm Layer
                        num_bn_biases = bn.bias.numel()

                        # Load the weights
                        bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                        ptr += num_bn_biases

                        bn_weights = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                        ptr += num_bn_biases

                        bn_running_mean = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                        ptr += num_bn_biases

                        bn_running_var = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                        ptr += num_bn_biases

                        # Cast the loaded weights into dims of model weights
                        bn_biases = bn_biases.view_as(bn.bias.data)
                        bn_weights = bn_weights.view_as(bn.weight.data)
                        bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                        bn_running_var = bn_running_var.view_as(bn.running_var)

                        # Copy the data to model
                        bn.bias.data.copy_(bn_biases)
                        bn.weight.data.copy_(bn_weights)
                        bn.running_mean.copy_(bn_running_mean)
                        bn.running_var.copy_(bn_running_var)
                    else:
                        # Number of biases
                        num_biases = conv.bias.numel()

                        # Load the biases
                        conv_biases = torch.from_numpy(weights[ptr:ptr + num_biases])
                        ptr = ptr + num_biases

                        # Reshape the loaded biases according to the dims of the model biases
                        conv_biases = conv_biases.view_as(conv.bias.data)

                        # Copy data
                        conv.bias.data.copy_(conv_biases)
                    
                    # Load the conv layer weights
                    num_wts = conv.weight.numel()

                    conv_weights = torch.from_numpy(weights[ptr:ptr + num_wts])
                    ptr += num_wts

                    conv_weights = conv_weights.view_as(conv.weight.data)
                    conv.weight.data.copy_(conv_weights)


'''
def main():
    if 2 > len(sys.argv) > 3:
        sys.exit("Invalid clargs") 
    
    cfg_filename = sys.argv[1]
    print("Config file: ()".format(cfg_filename))
    if len(sys.argv) == 3:
        wts_file = sys.argv[2]
        print("Weights file: {}".format(wts_file))
        

    # +++ Testing code 1 +++ #
    ''' 
    all_blocks = parse_cfg(cfg_filename)    
    print(len(all_blocks))
    print(create_modules(all_blocks))
    '''
    # Device setup
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")   
        
    print("\nUsing", device, "\n")
    model = Darknet(cfg_filename)
    model.to(device)
    inputs = get_test_input()
    inputs = inputs.to(device)
    '''
    pred = model(inputs, device)
    print(pred, pred.shape) 
    '''
    model.load_weights(wts_file)

if __name__ == "__main__":
    main()
'''













