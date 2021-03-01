#!/usr/bin/env python
from sensor_msgs.msg import Image,CameraInfo
from autoware_msgs.msg import PointsImage
from geometry_msgs.msg import PoseArray , Pose , PoseStamped
from cv_bridge import CvBridge
from std_msgs.msg import Header
import torch

import os
import rospkg
import rospy
import cv2
import numpy as np
import time
import yaml

from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg

# ======= Global Variable ======= #
point_image_sub_topic = "/points_image"
point_image_sub_topic_2 = "/points_image_2"
pred_max_sub_topic = "Drive/pred_max"
pred_max_numpy_sub_topic = "Drive/pred_max/numpy"


# ======= Subscriber Class ======= #
class Img_Sub():
    def __init__(self):
        self.points_image_sub = rospy.Subscriber(point_image_sub_topic, PointsImage, self.callback)
        self.points_image_ok = False
        self.distance = np.zeros((576, 1024)).astype(np.float)
        self.intensity = np.ones((576, 1024)).astype(np.float)
        # self.distance = np.zeros((360, 640)).astype(np.float)   # Modified
        # self.intensity = np.ones((360, 640)).astype(np.float)   # Modified
    def callback(self, msg):
        self.header = msg.header    # Modified
        image_height,image_width = msg.image_height, msg.image_width
        self.distance = np.array(msg.distance).reshape((image_height,image_width))
        self.points_image_ok = True

class Img_Sub2():
    def __init__(self):
        self.points_image_sub = rospy.Subscriber(point_image_sub_topic_2, PointsImage, self.callback)
        self.points_image_ok = False
        self.distance = np.zeros((576, 1024)).astype(np.float)
        self.intensity = np.ones((576, 1024)).astype(np.float)
        # self.distance = np.zeros((360, 640)).astype(np.float)   # Modified
        # self.intensity = np.ones((360, 640)).astype(np.float)   # Modified
    def callback(self, msg):
        self.header = msg.header    # Modified
        image_height,image_width = msg.image_height, msg.image_width
        self.distance = np.array(msg.distance).reshape((image_height,image_width))
        self.points_image_ok = True

class Pred_Max_Sub():
    def __init__(self):
        self.pred_max_sub = rospy.Subscriber(pred_max_sub_topic, Image, self.callback)
        self.pred_max = np.zeros((576,1024,3),dtype=np.uint8)
        # self.pred_max = np.zeros((360,640,3),dtype=np.uint8)    # Modified
        self.bridge = CvBridge()
        self.ok = False
    def callback(self, msg):
        self.pred_max = self.bridge.imgmsg_to_cv2(msg)
        self.ok = True

class Pred_Max_Numpy_Sub():
    def __init__(self):
        self.pred_max_sub = rospy.Subscriber(pred_max_numpy_sub_topic, numpy_msg(Floats), self.callback)
        self.pred_max = np.zeros((589824),dtype=np.float32)
        # self.pred_max = np.zeros((360,640,3),dtype=np.uint8)    # Modified
        self.ok = False
    def callback(self, msg):
        self.pred_max = msg.data
        self.pred_max = self.pred_max.reshape(576, 1024)
        self.ok = True


def demo(distance, pred_max):

    # ===== Extract alter lane ===== #
    alt_area = torch.tensor([0, 0, 255]).to(device).type('torch.cuda.ByteTensor')
    pred_max_alt = torch.all(pred_max == alt_area, dim = 2).type('torch.cuda.DoubleTensor')
    non_zero_alt_distance = torch.nonzero(distance * pred_max_alt)
    # print(non_zero_alt_distance.size()[0])

    filter_pred = torch.clone(pred_max)

    # # ===== Point image not on alt lane ===== #
    if(non_zero_alt_distance.size()[0] == 0):
        return filter_pred
    
    # # ===== Point image on alt lane ===== #
    else:
        # ===== Distinguish on left or right alt lane ===== #

        # Find Main area point index
        drive_area = torch.tensor([0, 255, 0]).to(device).type('torch.cuda.ByteTensor')
        main_y, main_x = torch.where(torch.all(pred_max == drive_area, dim = 2))
        main_x = main_x.float()
        mean_main_x = torch.mean(main_x)
        # print(mean_main_x)
        
        # Find Alter lane point index
        alt_y_idx, alt_x_idx = torch.where(torch.all(pred_max == alt_area, dim = 2))
        alt_x_idx = alt_x_idx.float()

        # Left and right alt lane index
        alt_x_idx_left_idx =  torch.where(alt_x_idx < mean_main_x)[0]
        alt_y_idx_left_idx = alt_y_idx[alt_x_idx_left_idx]

        alt_x_idx_right_idx = torch.where(alt_x_idx > mean_main_x)[0]
        alt_y_idx_right_idx = alt_y_idx[alt_x_idx_right_idx]

        # Left and right alt lane(with points image)
        non_zero_alt_distance = non_zero_alt_distance.float()
        alt_x_distance_left_idx = torch.where(non_zero_alt_distance[:, 1] < mean_main_x)[0]
        alt_x_distance_right_idx =  torch.where(non_zero_alt_distance[:, 1] > mean_main_x)[0]

        # Remove left alt lane
        background = torch.tensor([0, 0, 0]).to(device).type('torch.cuda.ByteTensor')
        if(alt_x_distance_left_idx.size()[0] > 0):
            filter_pred[alt_y_idx[0:alt_y_idx_left_idx.size()[0]].type(torch.long), alt_x_idx[0:alt_x_idx_left_idx.size()[0]].type(torch.long)] = background
            
        # Remove right alt lane
        if(alt_x_distance_right_idx.size()[0] > 0):
            filter_pred[alt_y_idx[0:alt_y_idx_right_idx.size()[0]].type(torch.long), alt_x_idx[0:alt_x_idx_right_idx.size()[0]].type(torch.long)] = background


        return filter_pred


def demo2d(distance, pred_max):

    # ===== Extract alter lane ===== #
    # alt_area = torch.tensor([2]).to(device).type('torch.cuda.FloatTensor')
    pred_max_alt = torch.where(pred_max == 2, torch.tensor(1).to(device), torch.tensor(0).to(device)).type('torch.cuda.DoubleTensor')
    non_zero_alt_distance = torch.nonzero(distance * pred_max_alt)
    # print(non_zero_alt_distance.size()[0])

    filter_pred = torch.clone(pred_max)

    # # ===== Point image not on alt lane ===== #
    if(non_zero_alt_distance.size()[0] == 0):
        return filter_pred
    
    # # ===== Point image on alt lane ===== #
    else:
        # ===== Distinguish on left or right alt lane ===== #

        # Find Main area point index
        # drive_area = torch.tensor([1]).to(device).type('torch.cuda.FloatTensor')
        main_y, main_x = torch.where(pred_max == 1)
        main_x = main_x.float()
        mean_main_x = torch.mean(main_x)
        print(mean_main_x)
        
        # Find Alter lane point index
        alt_y_idx, alt_x_idx = torch.where(pred_max == 2)
        alt_x_idx = alt_x_idx.float()

        # Left and right alt lane index
        alt_x_idx_left_idx =  torch.where(alt_x_idx < mean_main_x)[0]
        alt_y_idx_left_idx = alt_y_idx[alt_x_idx_left_idx]

        alt_x_idx_right_idx = torch.where(alt_x_idx > mean_main_x)[0]
        alt_y_idx_right_idx = alt_y_idx[alt_x_idx_right_idx]

        # Left and right alt lane(with points image)
        non_zero_alt_distance = non_zero_alt_distance.float()
        alt_x_distance_left_idx = torch.where(non_zero_alt_distance[:, 1] < mean_main_x)[0]
        alt_x_distance_right_idx =  torch.where(non_zero_alt_distance[:, 1] > mean_main_x)[0]

        # Remove left alt lane
        background = torch.tensor(0).to(device).type('torch.cuda.FloatTensor')
        if(alt_x_distance_left_idx.size()[0] > 0):
            filter_pred[alt_y_idx[0:alt_y_idx_left_idx.size()[0]].type(torch.long), alt_x_idx[0:alt_x_idx_left_idx.size()[0]].type(torch.long)] = background
            
        # Remove right alt lane
        if(alt_x_distance_right_idx.size()[0] > 0):
            filter_pred[alt_y_idx[0:alt_y_idx_right_idx.size()[0]].type(torch.long), alt_x_idx[0:alt_x_idx_right_idx.size()[0]].type(torch.long)] = background


        return filter_pred


def filter_lane_no_ground(filter_pred, distance):

    # Get the points image highest point
    distance_y, distance_x = torch.where(distance > 0.0)
    distance_y_min = torch.min(distance_y)

    # Filter the lane when its height is higher than points image
    filter_y_array = torch.ones(filter_pred.size()).to(device)
    filter_y_array[0:distance_y_min-1, :, :] = 0.0
    filter_y_array = filter_y_array.type('torch.cuda.ByteTensor')

    # Intersection filter_y_array and pred_max --> Filter the lane not on the ground
    filter_pred_no_ground = filter_pred * filter_y_array

    return filter_pred_no_ground


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rospack = rospkg.RosPack()
    pkg_root = os.path.join(rospack.get_path('drive_area_detection'),'src','FCHarDNet')

    with open(os.path.join(pkg_root,"configs/demo.yml")) as fp:
        cfg = yaml.load(fp)

    # ======= Init Node ======= #
    rospy.init_node('Filter_Lane_image_tensor', anonymous=True)
    rospack = rospkg.RosPack()
    pkg_root = os.path.join(rospack.get_path('points_raw_check'),'src')

    # ======= Publisher ======= #
    drive_area_filter = rospy.Publisher("/Drive_filter_image_tensor", Image)
    drive_area_filter_2d = rospy.Publisher("Drive_filter_numpy_tensor", Image)
    drive_area_filter_2d_numpy = rospy.Publisher("Drive_filter_numpy_tensor/numpy", numpy_msg(Floats))

    # ======= Subscriber ======= #
    points_image_sub = Img_Sub()
    points_image_sub_2 = Img_Sub2()
    pred_max_sub = Pred_Max_Sub()
    pred_max_numpy_sub = Pred_Max_Numpy_Sub()

    while not (points_image_sub.points_image_ok and points_image_sub_2.points_image_ok and pred_max_sub.ok):
        time.sleep(0.5)

    print("drive_area_filter is ok!!")

    while not rospy.is_shutdown():

        rate = rospy.Rate(15)

        # ======= global name bridge ======= #
        bridge = CvBridge()

        # Get Drive/pred_max and convert numpy to tensor
        # pred_max = torch.tensor(pred_max_sub.pred_max.copy()).to(device)
        pred_max_numpy = torch.tensor(pred_max_numpy_sub.pred_max.copy()).to(device)

        # Get /points_image and convert to tensor
        distance_2= torch.tensor(points_image_sub_2.distance.copy()).to(device)

        # ===== Filter alter lane on crosswalk ===== #
        # filter_pred = demo(distance_2, pred_max)
        # filter_pred = filter_pred.cpu().numpy()

        filter_pred_numpy = demo2d(distance_2, pred_max_numpy)
        filter_pred_numpy = filter_pred_numpy.cpu().numpy()

        # # Get /points_image_2 and convert to tensor
        # distance = torch.tensor(points_image_sub_2.distance.copy()).to(device)

        # # ===== Filter lane not on the road ===== #
        # filter_pred_no_ground = filter_lane_no_ground(filter_pred, distance)
        # filter_pred_no_ground = filter_pred_no_ground.cpu().numpy()


        # ===== Publish msg ===== #
        # drive_area_filter.publish(bridge.cv2_to_imgmsg(filter_pred.astype(np.uint8), 'bgr8'))
        drive_area_filter_2d_numpy.publish(filter_pred_numpy.copy().astype(np.float32).flatten())

        rate.sleep()

