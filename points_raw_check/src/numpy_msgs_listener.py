#!/usr/bin/env python
import numpy as np
import rospy
from rospy.numpy_msg import numpy_msg
# from rospy_tutorials.msg import Floats
import time
from drive_area_detection.msg import DrivePredMax


class Numpy_Msgs_Sub():
    def __init__(self):
        self.numpy_msgs_sub = rospy.Subscriber('/Numpy_Pub', numpy_msg(DrivePredMax), self.callback)
        self.numpy_msgs = np.zeros((10), dtype=np.int8)
        self.ok = False
    def callback(self, msg):
        self.numpy_msgs = msg.data.reshape(2,5).astype(np.uint8)
        # print(msg.data)
        self.ok = True


if __name__ == '__main__':

    # ===== Init Node ===== #
    rospy.init_node('numpy_msgs_listener', anonymous=True)

    # ===== Subscriber ===== #
    pred_max_numpy_sub = Numpy_Msgs_Sub()

    # Rate 
    rate = rospy.Rate(10)

    while not pred_max_numpy_sub.ok:
        time.sleep(0.5)

    while not rospy.is_shutdown():

        print("Heard msg: ", pred_max_numpy_sub.numpy_msgs)
        print(type(pred_max_numpy_sub.numpy_msgs))

        rate.sleep()



