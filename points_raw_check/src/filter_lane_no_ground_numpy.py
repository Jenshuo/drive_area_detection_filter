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

# ======= Global Variable ======= #
point_image_sub_topic = "/points_image"
pred_max_numpy_sub_topic = "Drive_filter_alter/pred_max/numpy"

# ======= Subscriber Class ======= #
class Img_Sub():
    def __init__(self):
        self.points_image_sub = rospy.Subscriber(point_image_sub_topic, PointsImage, self.callback)
        self.points_image_ok = False
        self.distance = np.zeros((576, 1024)).astype(np.float)
        self.intensity = np.ones((576, 1024)).astype(np.float)
    def callback(self, msg):
        self.header = msg.header    # Modified
        image_height,image_width = msg.image_height, msg.image_width
        self.distance = np.array(msg.distance).reshape((image_height,image_width))
        self.points_image_ok = True


class Pred_Max_Numpy_Sub():
    def __init__(self):
        self.pred_max_sub = rospy.Subscriber(pred_max_numpy_sub_topic, numpy_msg(DrivePredMax), self.callback)
        self.pred_max = np.zeros((576, 1024),dtype=np.uint8) # 576 * 1024
        self.ok = False
    def callback(self, msg):
        self.pred_max = msg.data.reshape(576, 1024).astype(np.uint8)
        self.ok = True


def filter_lane_no_ground(points_image_sub_2, filter_pred):

    # Get /points_image
    distance = points_image_sub_2.distance.copy()

    # Get the points image highest point
    distance_y, distance_x = np.where(distance > 0.0)
    distance_y_min = np.min(distance_y)

    # Filter the lane when its height is higher than points image
    filter_y_array = np.ones((filter_pred.shape[0], filter_pred.shape[1], filter_pred.shape[2]))
    filter_y_array[0:distance_y_min-1, :, :] = 0.0

    # Intersection filter_y_array and pred_max --> Filter the lane not on the ground
    filter_pred_no_ground = filter_pred * filter_y_array

    return filter_pred_no_ground

def filter_lane_no_ground_2d(points_image_sub_2, filter_pred):
    # Get /points_image
    distance = points_image_sub_2.distance.copy().astype(np.bool)

    # Get the points image highest point
    # distance_y, distance_x = np.where(distance > 0.0)
    distance_y, distance_x = np.where(distance == True)
    distance_y_min = np.min(distance_y)

    # Filter the lane when its height is higher than points image
    filter_y_array = np.ones((filter_pred.shape[0], filter_pred.shape[1]), dtype=bool)
    filter_y_array[0:distance_y_min-1, :] = False

    # Intersection filter_y_array and pred_max --> Filter the lane not on the ground
    filter_pred_no_ground = filter_pred * filter_y_array

    return filter_pred_no_ground
    

def fill_color(filter_pred_2d, cfg):

    pred_max_color = np.zeros((cfg['data']['img_rows'],cfg['data']['img_cols'],3),dtype=np.uint8)
    for key in cfg["vis"]:
        if key == "background":
            continue
        pred_max_color[filter_pred_2d==cfg["vis"][key]["id"]] = np.array(cfg["vis"][key]["color"])
    
    return pred_max_color


if __name__ == "__main__":

    rospack = rospkg.RosPack()
    pkg_root_lane_detection = os.path.join(rospack.get_path('drive_area_detection'),'src','FCHarDNet')

    with open(os.path.join(pkg_root_lane_detection,"configs/demo.yml")) as fp:
        cfg = yaml.load(fp)

    # ======= Init Node ======= #
    rospy.init_node('Filter_No_Ground_Lane', anonymous=True)
    rospack = rospkg.RosPack()
    pkg_root = os.path.join(rospack.get_path('points_raw_check'),'src')

    # ======= Publisher ======= #
    # drive_area_filter = rospy.Publisher("Drive_filter_no_ground/pred_max", Image)
    drive_area_filter_numpy = rospy.Publisher('Drive_filter_no_ground/pred_max/numpy', numpy_msg(DrivePredMax))

    # ======= Subscriber ======= #
    points_image_sub = Img_Sub()
    pred_max_numpy_sub = Pred_Max_Numpy_Sub()

    while not (points_image_sub.points_image_ok and pred_max_numpy_sub.ok):
        time.sleep(0.5)

    print("drive_area_no_ground_filter is ok!!")

    rate = rospy.Rate(15)

    while not rospy.is_shutdown():

        # ======= global name bridge ======= #
        bridge = CvBridge()

        # Get Drive/pred_max
        # pred_max = pred_max_sub.pred_max.copy()
        pred_max_numpy = pred_max_numpy_sub.pred_max.copy()

        # ===== Filter lane not on the road ===== #
        filter_pred_no_ground = filter_lane_no_ground_2d(points_image_sub, pred_max_numpy)

        # ===== Publish msg ===== #
        pred_max_msg = DrivePredMax()
        pred_max_msg.data = filter_pred_no_ground.copy().astype(np.int8).flatten().tolist()
        drive_area_filter_numpy.publish(pred_max_msg)
    

        rate.sleep()

