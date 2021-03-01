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
point_image_sub_topic_2 = "/points_image_2"
pred_max_sub_topic = "Drive/pred_max"
pred_max_numpy_sub_topic = "Drive/pred_max/numpy"

# ======= Subscriber Class ======= #
class Img_Sub2():
    def __init__(self):
        self.points_image_sub = rospy.Subscriber(point_image_sub_topic_2, PointsImage, self.callback)
        self.points_image_ok = False
        self.distance = np.zeros((576, 1024), dtype=bool)
    def callback(self, msg):
        self.header = msg.header    
        image_height,image_width = msg.image_height, msg.image_width
        self.distance = np.array(msg.distance).reshape((image_height,image_width)).astype(np.bool)
        self.points_image_ok = True

class Pred_Max_Sub():
    def __init__(self):
        self.pred_max_sub = rospy.Subscriber(pred_max_sub_topic, Image, self.callback)
        self.pred_max = np.zeros((576,1024,3),dtype=np.uint8)
        self.bridge = CvBridge()
        self.ok = False
    def callback(self, msg):
        self.pred_max = self.bridge.imgmsg_to_cv2(msg)
        self.ok = True


class Pred_Max_Numpy_Sub():
    def __init__(self):
        self.pred_max_sub = rospy.Subscriber(pred_max_numpy_sub_topic, numpy_msg(DrivePredMax), self.callback)
        self.pred_max = np.zeros((576, 1024), dtype=np.uint8) # 576 * 1024
        self.ok = False
    def callback(self, msg):
        self.pred_max = msg.data.reshape(576, 1024).astype(np.uint8)
        self.ok = True


def demo(points_image_sub, pred_max):

    # Get /points_image
    distance = points_image_sub.distance.copy()

    # ===== Extract alter lane ===== #
    pred_max_alt = np.all(pred_max == [0, 0, 255], axis = -1)   # Bottleneck --> sol: use numpy (576, 1024), 7 class(0 ~6)

    # distance[distance > 0] = 1
    alt_y_distance, alt_x_distance = np.nonzero(distance * pred_max_alt)
    # print(alt_x_distance)

    filter_pred = pred_max.copy()

    # ===== Point image not on alt lane ===== #
    if(len(alt_x_distance) == 0):
        return filter_pred
    
    # ===== Point image on alt lane ===== #
    else:
        # ===== Distinguish on left or right alt lane ===== #

        # Find Main area point index
        main_y, main_x = np.where(np.all(pred_max == [0, 255, 0], axis = -1))
        mean_main_x = np.mean(main_x)
        
        # Find Alter lane point index
        alt_y_idx, alt_x_idx = np.where(np.all(pred_max == [0, 0, 255], axis = -1))

        # Left and right alt lane index
        alt_x_idx_left_idx =  np.array(np.where(alt_x_idx < mean_main_x))
        alt_x_idx_left_idx = np.squeeze(alt_x_idx_left_idx)
        alt_y_idx_left_idx = alt_y_idx[alt_x_idx_left_idx]

        alt_x_idx_right_idx = np.array(np.where(alt_x_idx > mean_main_x))
        alt_x_idx_right_idx = np.squeeze(alt_x_idx_right_idx)
        alt_y_idx_right_idx = alt_y_idx[alt_x_idx_right_idx]

        # Left and right alt lane(with points image)
        alt_x_distance_left_idx = np.array(np.where(alt_x_distance < mean_main_x))
        alt_x_distance_left_idx = np.squeeze(alt_x_distance_left_idx, axis=0)
        alt_x_distance_right_idx =  np.array(np.where(alt_x_distance > mean_main_x))
        alt_x_distance_right_idx = np.squeeze(alt_x_distance_right_idx, axis=0)

        # Remove left alt lane
        if(len(alt_x_distance_left_idx) > 0):
            filter_pred[alt_y_idx[0:len(alt_y_idx_left_idx)], alt_x_idx[0:len(alt_x_idx_left_idx)]] = [0, 0, 0]

        # Remove right alt lane
        if(len(alt_x_distance_right_idx) > 0):
            filter_pred[alt_y_idx[0:len(alt_y_idx_right_idx)], alt_x_idx[0:len(alt_x_idx_right_idx)]] = [0, 0, 0]


        # drive_area_filter.publish(bridge.cv2_to_imgmsg(filter_pred.astype(np.uint8), 'bgr8'))
        return filter_pred


def demo2d(points_image_sub, pred_max_numpy):

    # Get /points_image
    distance = points_image_sub.distance.copy()
    # print(type(distance[0,0]))
    
    # ===== Extract alter lane ===== #
    pred_max_alt_numpy = np.where(pred_max_numpy == cfg["vis"]["alter_lane"]["id"], 1, 0).astype(np.bool)    # Extract alter lane from numpy
    # distance[distance > 0] = 1
    alt_y_distance, alt_x_distance = np.nonzero(distance * pred_max_alt_numpy)
    # print(alt_x_distance)

    filter_pred_2d = pred_max_numpy.copy()

    # ===== Point image not on alt lane ===== #
    if(len(alt_x_distance) == 0):
        return filter_pred_2d
    
    # ===== Point image on alt lane ===== #
    else:
        # ===== Distinguish on left or right alt lane ===== #

        # Find Main area point index
        main_y, main_x = np.where(pred_max_numpy == cfg["vis"]["main_lane"]["id"])
        mean_main_x = np.mean(main_x)
        
        # Find Alter lane point index
        alt_y_idx, alt_x_idx = np.where(pred_max_numpy == cfg["vis"]["alter_lane"]["id"])

        # Left and right alt lane index
        alt_x_idx_left_idx =  np.array(np.where(alt_x_idx < mean_main_x))
        alt_x_idx_left_idx = np.squeeze(alt_x_idx_left_idx)
        alt_y_idx_left_idx = alt_y_idx[alt_x_idx_left_idx]

        alt_x_idx_right_idx = np.array(np.where(alt_x_idx > mean_main_x))
        alt_x_idx_right_idx = np.squeeze(alt_x_idx_right_idx)
        alt_y_idx_right_idx = alt_y_idx[alt_x_idx_right_idx]

        # Left and right alt lane(with points image)
        alt_x_distance_left_idx = np.array(np.where(alt_x_distance < mean_main_x))
        alt_x_distance_left_idx = np.squeeze(alt_x_distance_left_idx, axis=0)
        alt_x_distance_right_idx =  np.array(np.where(alt_x_distance > mean_main_x))
        alt_x_distance_right_idx = np.squeeze(alt_x_distance_right_idx, axis=0)

        # Remove left alt lane
        if(len(alt_x_distance_left_idx) > 0):
            filter_pred_2d[alt_y_idx[0:len(alt_y_idx_left_idx)], alt_x_idx[0:len(alt_x_idx_left_idx)]] = 0

        # Remove right alt lane
        if(len(alt_x_distance_right_idx) > 0):
            filter_pred_2d[alt_y_idx[0:len(alt_y_idx_right_idx)], alt_x_idx[0:len(alt_x_idx_right_idx)]] = 0


        return filter_pred_2d



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
    rospy.init_node('Filter_Alter_Lane', anonymous=True)
    rospack = rospkg.RosPack()
    pkg_root = os.path.join(rospack.get_path('points_raw_check'),'src')

    # ======= Publisher ======= #
    drive_area_filter_numpy = rospy.Publisher('Drive_filter_alter/pred_max/numpy', numpy_msg(DrivePredMax))
    # drive_area_filter = rospy.Publisher("Drive_filter/pred_max", Image)

    # ======= Subscriber ======= #
    points_image_sub_2 = Img_Sub2()
    pred_max_numpy_sub = Pred_Max_Numpy_Sub()
    # pred_max_sub = Pred_Max_Sub()

    while not (points_image_sub_2.points_image_ok and pred_max_numpy_sub.ok):
        time.sleep(0.5)

    print("drive_area_filter is ok!!")

    rate = rospy.Rate(15)
    
    while not rospy.is_shutdown():

        # ======= global name bridge ======= #
        bridge = CvBridge()

        # Get Drive/pred_max
        pred_max_numpy = pred_max_numpy_sub.pred_max.copy()
        # pred_max = pred_max_sub.pred_max.copy()

        # ===== Filter alter lane on crosswalk ===== #
        filter_pred_numpy = demo2d(points_image_sub_2, pred_max_numpy)

        # ===== Publish msg ===== #
        pred_max_msg = DrivePredMax()
        pred_max_msg.data = filter_pred_numpy.copy().astype(np.int8).flatten().tolist()
        drive_area_filter_numpy.publish(pred_max_msg)
        # drive_area_filter.publish(bridge.cv2_to_imgmsg(filter_pred.astype(np.uint8), 'bgr8'))

        rate.sleep()

