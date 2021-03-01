#!/usr/bin/env python
from sensor_msgs.msg import Image,CameraInfo
from autoware_msgs.msg import PointsImage
from geometry_msgs.msg import PoseArray , Pose , PoseStamped
from cv_bridge import CvBridge
from std_msgs.msg import Header

import os
import rospkg
import rospy
import cv2
import numpy as np
import time
import yaml

# from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from drive_area_detection.msg import DrivePredMax


pred_max_sub_topic = "Drive_filter_no_ground/pred_max/numpy"

class Img_Sub():
    def __init__(self):
        self.bridge = CvBridge()
        self.image_raw_sub= rospy.Subscriber(cfg['image_src'], Image, self.callback)
        self.header = Header()
        self.image_ok = False
        self.count = 0
    def callback(self, msg):
        image_raw  = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.header = msg.header
        # self.image_raw = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.image_raw = cv2.resize(image_raw,(1024,576))
        self.image_ok = True

class Pred_Max_Numpy_Sub():
    def __init__(self):
        self.pred_max_sub = rospy.Subscriber(pred_max_sub_topic, numpy_msg(DrivePredMax), self.callback)
        self.pred_max = np.zeros((576, 1024),dtype=np.uint8)
        self.ok = False
    def callback(self, msg):
        self.pred_max = msg.data.reshape(576, 1024).astype(np.uint8)
        self.ok = True


def fill_color(filter_pred_2d, cfg, image_raw_sub):
    pred_max_color = np.zeros((cfg['data']['img_rows'],cfg['data']['img_cols'],3),dtype=np.uint8)
    RGB = np.zeros((cfg['data']['img_rows'], cfg['data']['img_cols'], 3), dtype=np.uint8)
    overlay_flag = np.zeros((cfg['data']['img_rows'], cfg['data']['img_cols']))
    img_draw = image_raw_sub.image_raw.copy().astype(np.uint8)
    alpha = 0.8

    for key in cfg["vis"]:
        if key == "background":
            continue
        pred_max_color[filter_pred_2d==cfg["vis"][key]["id"]] = np.array(cfg["vis"][key]["color"])
        RGB[filter_pred_2d==cfg["vis"][key]["id"]] = np.array(cfg["vis"][key]["color"])
        overlay_flag[filter_pred_2d==cfg["vis"][key]["id"]] = 1

    """ overlay """
    overlay = cv2.addWeighted(img_draw, alpha, RGB, (1-alpha), 0)
    img_draw[overlay_flag == 1] = overlay[overlay_flag == 1]

    return img_draw


if __name__ == "__main__":

    rospack = rospkg.RosPack()
    pkg_root_lane_detection = os.path.join(rospack.get_path('drive_area_detection'),'src','FCHarDNet')

    # Output video path
    # outpath = '/media/ivslab/f53ae25d-cb9f-4170-a0c3-ed7a6264afc5/NCTU_Round_lane_intersection_filtered_alter.avi'
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out_videos = cv2.VideoWriter(outpath, fourcc, 15.0, (1024, 576))

    with open(os.path.join(pkg_root_lane_detection,"configs/demo.yml")) as fp:
        cfg = yaml.load(fp)
    
    # ======= Init Node ======= #
    rospy.init_node('Draw_Filter_Lane', anonymous=True)
    rospack = rospkg.RosPack()
    pkg_root = os.path.join(rospack.get_path('points_raw_check'),'src')

    # ======= Publisher ======= #
    drive_area_image = rospy.Publisher("Drive_filter/pred_max/image", Image)

    # ======= Subscriber ======= #
    image_raw_sub = Img_Sub()
    pred_max_numpy_sub = Pred_Max_Numpy_Sub()

    while not (pred_max_numpy_sub.ok and image_raw_sub.image_ok):
        time.sleep(0.5)
    
    rate = rospy.Rate(15)

    while not rospy.is_shutdown():

        # ======= global name bridge ======= #
        bridge = CvBridge()

        # Get Drive_filter_no_ground/pred_max/numpy
        pred_max_numpy = pred_max_numpy_sub.pred_max.copy()

        # ===== Draw color on image ===== #
        pred_max_image = fill_color(pred_max_numpy, cfg, image_raw_sub)

        # Write to video
        # out_videos.write(pred_max_image)

        # ===== Publish msg ===== #
        drive_area_image.publish(bridge.cv2_to_imgmsg(pred_max_image.astype(np.uint8), 'bgr8'))

        rate.sleep()