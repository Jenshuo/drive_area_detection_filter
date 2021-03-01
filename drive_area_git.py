#!/usr/bin/env python
import os
import yaml
from collections import OrderedDict
import time

import torch

import argparse


import numpy as np

import torch.nn as nn
import torchvision


from ptsemseg.models import get_model


import cv2



from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header

import rospy
import rospkg

from rospy.numpy_msg import numpy_msg
from drive_area_detection.msg import DrivePredMax



def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

class Img_Sub():
    def __init__(self,cfg):
        self.bridge = CvBridge()
        self.image_raw_sub= rospy.Subscriber(cfg['image_src'], Image, self.callback)
        self.lane_mask_sub= rospy.Subscriber("/Lane/mask", Image, self.lane_callback)
        self.header = Header()
        self.image_ok = False
        self.lane_ok = False
        self.count = 0
    def callback(self, msg):
        image_raw  = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.header = msg.header
        self.image_raw = cv2.resize(image_raw,(1024,576))
        self.image_ok = True
        self.count += 1
    def lane_callback(self,msg):
        self.lane_mask = self.bridge.imgmsg_to_cv2(msg, "mono8")
        self.lane_ok = True


def demo(cfg,pkg_root,img_sub,drive_pred_max):
     # Setup device


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("============", device)
    # Setup Model
    n_classes = cfg["testing"]["n_classes"]
    img_size = (cfg["data"]["img_rows"], cfg["data"]["img_cols"])

    model = get_model(cfg["model"], n_classes)

    total_params = sum(p.numel() for p in model.parameters())
    print( 'Parameters:',total_params )

    if cfg["testing"]["resume"] is not None:
        resume_dir = os.path.join(pkg_root,cfg["testing"]["resume"])
        if os.path.isfile(resume_dir):
            print(
                "Loading model and optimizer from checkpoint '{}'".format(resume_dir)
            )
            checkpoint = torch.load(resume_dir, map_location='cuda:0')
            new_state_dict = OrderedDict()
            for k, v in checkpoint["model_state"].items():
                name = k[7:]
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            model = model.to(device)
            print(
                "Loaded checkpoint '{}' (iter {})".format(
                    resume_dir, checkpoint["epoch"]
                )
            )
        else:
            print("No checkpoint found at '{}'".format(resume_dir))
             
    # image = cv2.imread('pic/10.jpg')
    bridge = CvBridge()
    rate = rospy.Rate(cfg["testing"]["publish_rate"])

    # Color like mtsan

    with torch.no_grad():

        while not rospy.is_shutdown():
            
            image = img_sub.image_raw

            ToTensor = torchvision.transforms.ToTensor()
            image = ToTensor(image).unsqueeze(0).to(device)
            out = model(image) 
            
            out_max = torch.nn.functional.softmax(out[0],dim=0).argmax(0)
            out_max = out_max.detach().cpu().numpy()

            ''' Add '''
            # ===== DrivePredMax ===== #
            pred_max_numpy = DrivePredMax()
            pred_max_numpy.data = out_max.copy().astype(np.int8).flatten().tolist()
            drive_pred_max_numpy.publish(pred_max_numpy)
            # ===== DrivePredMax ===== #
            ''' Add '''

            # ===== Publish id of lane ===== #

            RGB = np.zeros((cfg['data']['img_rows'], cfg['data']['img_cols'], 3), dtype=np.uint8)
            overlay_flag = np.zeros((cfg['data']['img_rows'], cfg['data']['img_cols']))
            img_draw = img_sub.image_raw.copy().astype(np.uint8)
            alpha = 0.8
            
            out_max_color = np.zeros((cfg['data']['img_rows'],cfg['data']['img_cols'],3),dtype=np.uint8)
           
            for key in cfg["vis"]:
                if key == "background":
                    continue
                out_max_color[out_max==cfg["vis"][key]["id"]] = np.array(cfg["vis"][key]["color"])
                RGB[out_max==cfg["vis"][key]["id"]] = np.array(cfg["vis"][key]["color"])
                overlay_flag[out_max==cfg["vis"][key]["id"]] = 1

                

            drive_pred_max.publish(bridge.cv2_to_imgmsg(out_max_color.astype(np.uint8),'bgr8'))

            """ overlay """
            overlay = cv2.addWeighted(img_draw, alpha, RGB, (1-alpha), 0)
            img_draw[overlay_flag == 1] = overlay[overlay_flag == 1]

          
            # Publish msg
            drive_area_pred.publish(bridge.cv2_to_imgmsg(img_draw.astype(np.uint8),'bgr8'))

        
            rate.sleep()
    return


    

if __name__ == "__main__":
    rospy.init_node('DriveArea', anonymous=True)
    rospack = rospkg.RosPack()
    pkg_root = os.path.join(rospack.get_path('drive_area_detection'),'src','FCHarDNet')


    # Load basic config from config yaml file --------
    with open(os.path.join(pkg_root,"configs/demo.yml")) as fp:
        cfg = yaml.load(fp)
    # Load ROS param  --------
    poblished_rate = rospy.get_param("~det_rate")
    # poblished_rate = 100
    image_src = rospy.get_param('~image_src')
    cfg['image_src'] = image_src
    cfg["testing"]["publish_rate"] = poblished_rate

    img_sub = Img_Sub(cfg)

    # Publish node init  --------

    drive_pred_max = rospy.Publisher("Drive/pred_max", Image)
    drive_area_pred = rospy.Publisher("/pred_area_img", Image)
    ''' Add '''
    drive_pred_max_numpy = rospy.Publisher('Drive/pred_max/numpy', numpy_msg(DrivePredMax))
    ''' Add '''

    # drive_area_vritual = rospy.Publisher("/virtual_img", Image, queue_size=1)

    while not rospy.is_shutdown():
        if img_sub.image_ok:
            break
        print("drive_area image_src is not ready")
        time.sleep(0.5)
        
    print("drive_area is ok!!")
    demo(cfg,pkg_root,img_sub,drive_pred_max)

    
