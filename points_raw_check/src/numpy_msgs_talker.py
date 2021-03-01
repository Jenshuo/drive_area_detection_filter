#!/usr/bin/env python
import numpy as np
import rospy
from rospy.numpy_msg import numpy_msg
# from rospy_tutorials.msg import Floats
from drive_area_detection.msg import DrivePredMax



if __name__ == '__main__':

    # ===== Init Node ===== #
    rospy.init_node('numpy_msgs_talker', anonymous=True)

    # ===== Publisher ===== #
    pub = rospy.Publisher('/Numpy_Pub', numpy_msg(DrivePredMax))

    # Rate 
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():

        # Numpy array
        a = np.array([47,33,34,35,36,37,38,39,40,41]).astype(np.int8)
        new_msg = DrivePredMax()
        new_msg.data = a.tolist()
        pub.publish(new_msg)

        rate.sleep()



