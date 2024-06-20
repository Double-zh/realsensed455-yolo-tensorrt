# -*- coding: utf-8 -*-
"""
Created on Mon May 27 12:05:55 2024

@author: dudongjie
"""
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import open3d as o3d 
import os
import sys

view_ind = 0
breakLoopFlag = 0
backgroundColorFlag = 1


# 保存当前的RGBD和点云
def saveCurrentRGBD(vis):
    global view_ind, depth_image, color_image1, pcd
    if not os.path.exists('./output/'): 
        os.makedirs('./output')
    cv2.imwrite('./output/depth_'+str(view_ind)+'.png', depth_image)
    cv2.imwrite('./output/color_'+str(view_ind)+'.png', color_image1)
    o3d.io.write_point_cloud('./output/pointcloud_'+str(view_ind)+'.pcd', pcd)
    print('No.'+str(view_ind)+' shot is saved.')

    return False


# 退出当前程序
def breakLoop(vis):
    global breakLoopFlag
    breakLoopFlag += 1
    cv2.destroyAllWindows()
    vis.destroy_window()

    sys.exit()


def change_background_color(vis):
    global backgroundColorFlag
    opt = vis.get_render_option()
    if backgroundColorFlag:
        opt.background_color = np.asarray([0, 0, 0])
        backgroundColorFlag = 0
    else:
        opt.background_color = np.asarray([1, 1, 1])
        backgroundColorFlag = 1
    # background_color ~=backgroundColorFlag
    return False


if __name__ == "__main__":
    # 相机RGB和深度图对齐
    align = rs.align(rs.stream.color)
    # 配置视频流
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 15)

    # 创建一个上下文对象。该对象拥有所有连接的Realsense设备的句柄
    pipeline = rs.pipeline()

    # 开启视频流
    profile = pipeline.start(config)

    # get camera intrinsics 获取相机内在函数
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    # 设置open3d中的针孔相机数据
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
    # print(type(pinhole_camera_intrinsic))

    # 具有自定义按键Callback功能的可视化工具。
    geometrie_added = False
    vis = o3d.visualization.VisualizerWithKeyCallback()

    # 创建显示窗口
    vis.create_window("Pointcloud", 640, 480)
    # 定义点云类
    pointcloud = o3d.geometry.PointCloud()

    # 注册按键事件，触发后执行对应函数
    vis.register_key_callback(ord(" "), saveCurrentRGBD)
    vis.register_key_callback(ord("Q"), breakLoop)
    vis.register_key_callback(ord("K"), change_background_color)

    try:
        while True:
            time_start = time.time()

            # 清除几何中的所有元素。
            pointcloud.clear()

            # 等待图像进来
            frames = pipeline.wait_for_frames()
            # 将RGBD对齐
            aligned_frames = align.process(frames)

            # 获取RGB图像，并转为np格式数据
            color_frame = aligned_frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())

            # 获取深度信息
            depth_frame = aligned_frames.get_depth_frame()

            # 利用中值核进行滤波
            depth_frame = rs.decimation_filter(1).process(depth_frame)
            # 从深度表示转换为视差表示，反之亦然
            depth_frame = rs.disparity_transform(True).process(depth_frame)
            # 空间滤镜通过使用alpha和delta设置计算帧来平滑图像。
            depth_frame = rs.spatial_filter().process(depth_frame)
            # 时间滤镜通过使用alpha和delta设置计算多个帧来平滑图像。
            depth_frame = rs.temporal_filter().process(depth_frame)
            # 从视差表示转换为深度表示
            depth_frame = rs.disparity_transform(False).process(depth_frame)
            # depth_frame = rs.hole_filling_filter().process(depth_frame)

            # 将深度图转化为RGB准备显示
            depth_color_frame = rs.colorizer().colorize(depth_frame)
            depth_color_image = np.asanyarray(depth_color_frame.get_data())

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image1 = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR) # 由于cv中的显示是BGR所以需要再转换出一个

            # 在cv2中显示RGB-D图
            cv2.namedWindow('color image', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('color image', color_image1)
            cv2.namedWindow('depth image', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('depth image', depth_color_image)

            # 图像类存储具有可自定义的宽度，高度，通道数和每个通道字节数的图像。
            depth = o3d.geometry.Image(depth_image)
            color = o3d.geometry.Image(color_image)

            # 从颜色和深度图像制作RGBDImage的功能
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity=False)
            # 通过RGB-D图像和相机创建点云，并导入摄像机的固有参数
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
            # 将转换（4x4矩阵）应用于几何坐标。
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            # pcd = voxel_down_sample(pcd, voxel_size = 0.003)

            pointcloud += pcd  # 添加点云

            if not geometrie_added:
                # 第一次将几何体添加到场景并创建相应的着色器的功能，之后只需要更行的就好
                vis.add_geometry(pointcloud)
                geometrie_added = True

            # 当用于更新几何的功能。更改几何时必须调用此函数。
            vis.update_geometry(pointcloud)
            # 轮询事件的功能
            vis.poll_events()
            # 通知渲染需要更新的功能
            vis.update_renderer()

            # 帧数
            time_end = time.time()
            print("FPS = {0}".format(int(1 / (time_end - time_start))))

    finally:
        pipeline.stop()

cv2.destroyAllWindows()
