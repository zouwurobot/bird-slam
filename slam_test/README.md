# 视觉SLAM整理

@jiaj9815 2019/02/15

我们使用的相机是realsense D435，为RGBD相机  
一个方案汇总参考：[开源slam方案汇总](https://blog.csdn.net/lwx309025167/article/details/80257549)

## rtabmap
### 安装
rtabmap有一个ros package,可以直接安装  

` $ sudo apt-get install ros-kinetic-rtabmap-ros ` 

TX2可能比较特殊，安装需要参考具体官方指南。

### 运行
```
$ roslaunch realsense2_camera rs_camera.launch align_depth:=true  
$ roslaunch rtabmap_ros rtabmap.launch rtabmap_args:="--delete_db_on_start" depth_topic:=/camera/aligned_depth_to_color/image_raw rgb_topic:=/camera/color/image_raw camera_info_topic:=/camera/color/camera_info `
```
depth_topic和rgb_topic的名字可能要根据第一句launch的realsense的demo改变。  
可以选择rviz,rtab_rviz, odemetry的数据也可以读取（时间久远忘记具体操作了，但这个可以查rtab ROS wiki见参考链接2）。

### 结果
这是一个实时的，年前测试的结果是运动太快时，容易卡死，如果遇到卡死的情况可以手持摄像头沿着原路退回，再慢点运动或许可以调回去。

参考：  
1. 安装：[rtabmap_ros github](https://github.com/introlab/rtabmap_ros)  
2. 使用：[rtabmap ROS wiki](http://wiki.ros.org/rtabmap_ros#Overview)


## svo

这个是单目的，可以结合IMU，也是论文中使用的，但是有点浪费Realsense的深度功能，所以没有进行测试。 

### 安装

安装包：

x86_64 Ubuntu 14.04  
http://rpg.ifi.uzh.ch/svo2/svo_binaries_1404_indigo.zip 

x86_64 Ubuntu 16.04  
http://rpg.ifi.uzh.ch/svo2/svo_binaries_1604_kinetic.zip

armhf Ubuntu 14.04  
http://rpg.ifi.uzh.ch/svo2/svo_binaries_1404_indigo_armhf.zip

教程在安装的doc文件夹下面，或者论文的这个组的github里面也有。

### 运行

他需要一个相机的cali文件，ROS里面有一个包叫camera_calibration的包，可以生成需要格式的文件

教程：http://wiki.ros.org/camera_calibration/Tutorials/MonocularCalibration

## orb_slam2

### 安装

按照 https://github.com/raulmur/ORB_SLAM2 安装。
```
git clone https://github.com/raulmur/ORB_SLAM2.git ORB_SLAM2
cd ORB_SLAM2
chmod +x build.sh
./build.sh
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:PATH/ORB_SLAM2/Examples/ROS
chmod +x build_ros.sh
./build_ros.sh

```
### 运行
realsense的launch文件需要替换或者修改一下。

已经修改好的：rs_rgbd_JJ.launch(附在里面了)

*需要做的：*

*orb_slam2需要一个相机的cali文件，我用realsense官方的calibration软件得到的yaml文件不符合要求，建议下载那个巨大的数据集，参考一下例子里面的重新写一个。*

我测试用的指令：

```
$ roslaunch realsense2_camera rs_rgbd_JJ.launch
$ export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:/home/jing/intel_realsense/orb_slam2/ORB_SLAM2/Examples/ROS
$ rosrun ORB_SLAM2 RGBD /home/jing/intel_realsense/orb_slam2/ORB_SLAM2/Vocabulary/ORBvoc.txt /home/jing/d435.yaml

```
其中一些路径要根据自己的配置修改，具体参见 https://github.com/raulmur/ORB_SLAM2#building-the-nodes-for-mono-monoar-stereo-and-rgb-d

