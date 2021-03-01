#include <iostream>

#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/extract_indices.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>

#include <pcl/registration/icp.h>

#include <pcl/filters/voxel_grid.h>

#include <nav_msgs/Odometry.h>

#include <pcl/filters/extract_indices.h>
#include <stdlib.h>
#include <math.h>



// ============= Global variable ============= //
#define HEIGHT_THRES 0.20
ros::Publisher points_pub;
geometry_msgs::PoseStamped ndt_pose_base_link;    // ndt_pose on base_link frame
static tf::StampedTransform local_transform;
geometry_msgs::PointStamped points_base_link;   // points cloud on base_link

// ============= Subscriber ============= //
void points_callback(const sensor_msgs::PointCloud2::ConstPtr &input);
void pose_callback(const geometry_msgs::PoseStamped &pose_odom_);


// ============= Main Function ============= //
int main(int argc, char**argv){

    // ============= Init Node ============= //
    ros::init(argc, argv, "points_raw_check");
    ros::NodeHandle nh;

    // ============= Publisher ============= //
    points_pub = nh.advertise<sensor_msgs::PointCloud2>("/filter_points", 10);
    
    // ============= Subscriber ============= //
    ros::Subscriber pose_sub = nh.subscribe("/ndt_pose", 10, pose_callback);
    ros::Subscriber points_sub = nh.subscribe("/points_ground", 10, points_callback);

    // ======= TF transform ===== //
    // Velodyne frame to Base_link frame
    static tf::TransformListener transform_listener;
    try{
        ros::Time now = ros::Time(0);
        transform_listener.waitForTransform("/base_link", "/velodyne", now, ros::Duration(5.0));
        transform_listener.lookupTransform("/base_link", "/velodyne", now, local_transform);
    }
    catch (tf::TransformException& ex){
        ROS_ERROR("%s", ex.what());
    }

    ros::spin();

    return 0;
}

void points_callback(const sensor_msgs::PointCloud2::ConstPtr &input){

    // ============= PointsCloud2 --> PointXYZI ============= //
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_base_link(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*input, *cloud_ptr);

    // ============= Transform lidar points frame(velodyne --> base_link) ============= //
    pcl_ros::transformPointCloud(*cloud_ptr, *cloud_base_link, local_transform);

    // ============= Extract index of point cloud we want ============= //
    pcl::PointCloud<pcl::PointXYZI>::Ptr out_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::ExtractIndices<pcl::PointXYZI> extractor;
    extractor.setInputCloud(cloud_ptr);

    pcl::PointIndices::Ptr indices(new pcl::PointIndices);

    for(size_t i = 0; i < cloud_base_link->points.size(); i++){

        // Filter distance
        float distance = sqrtf(cloud_base_link->points[i].x * cloud_base_link->points[i].x + cloud_base_link->points[i].y * cloud_base_link->points[i].y);
        if(distance > 20.0) continue;

        // Filter y 
        if(cloud_base_link->points[i].y > ndt_pose_base_link.pose.position.y) continue;

        // Filter height
        if(cloud_base_link->points[i].z < ndt_pose_base_link.pose.position.z) continue;
        float diff = fabs(cloud_base_link->points[i].z - ndt_pose_base_link.pose.position.z);
        if(diff < HEIGHT_THRES) continue;

        // std::cout << "x: " << cloud_base_link->points[i].x << std::endl;
        // std::cout << "y: " << cloud_base_link->points[i].y << std::endl;
        // std::cout << "z: " << cloud_base_link->points[i].z << std::endl;
        // std::cout << "Pose x: " << ndt_pose_base_link.pose.position.x << std::endl;
        // std::cout << "Pose y: " << ndt_pose_base_link.pose.position.y << std::endl;
        // std::cout << "Pose z: " << ndt_pose_base_link.pose.position.z << std::endl;

        indices->indices.push_back(i);

    }

    extractor.setIndices(boost::make_shared<pcl::PointIndices>(*indices));
    extractor.setNegative(false);
    extractor.filter(*out_cloud_ptr);

    // ============= Output PointCloud2 ============= //
    sensor_msgs::PointCloud2 out_cloud_msg;
    pcl::toROSMsg(*out_cloud_ptr, out_cloud_msg);
    out_cloud_msg.header = input->header;
    points_pub.publish(out_cloud_msg);

}

void pose_callback(const geometry_msgs::PoseStamped &pose_odom_){

    ndt_pose_base_link.header = pose_odom_.header;

    // ============= TF Transform ============= //
    static tf::TransformListener local_transform_listener;
    ros::Time now = ros::Time(0);
    try{
        local_transform_listener.waitForTransform("/base_link", "/map", now, ros::Duration(5.0));
        local_transform_listener.transformPose("/base_link", now, pose_odom_, "/map", ndt_pose_base_link);
    }
    catch (tf::TransformException& ex){
        ROS_ERROR("%s", ex.what());
    }

    // ROS_INFO_STREAM(ndt_pose_velodyne);  
}