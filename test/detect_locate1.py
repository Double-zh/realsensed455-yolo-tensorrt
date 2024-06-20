#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:50:47 2024

@author: zzh
"""

# -*- coding: utf-8 -*-

import cv2
from ultralytics import YOLO

# 加载 YOLOv8 模型
model = YOLO("yolov8n.engine")

# 获取摄像头内容
#cap = cv2.VideoCapture(2)

# 获取视频内容
# video_path = "1.mp4"  # 替换为你的视频文件路径
# cap = cv2.VideoCapture(video_path)

# 获取原视频的大小
# original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 设置新的视频帧大小
# new_width = 1280
# new_height = 720

# 设置保存视频的文件名、编解码器和帧速率
# output_path = "output_video.avi"  # 替换为你的输出视频文件路径
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter(output_path, fourcc, 20.0, (original_width, original_height))


# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
 
# Create a pipeline
pipeline = rs.pipeline()       
 
# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
 
# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
print(device_product_line)
found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)
 
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
 
if device_product_line == 'L500' or device_product_line == 'D400':
    #config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.color, 1280, 800, rs.format.bgr8, 30)

else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
 
# Start streaming
profile = pipeline.start(config)
 
# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)
 
# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1  # 1 meter
clipping_distance = clipping_distance_in_meters / depth_scale
 
# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)
c = 0

def get_aligned_images():
    # frames = pipeline.wait_for_frames()  # 等待获取图像帧
    # aligned_frames = align.process(frames)  # 获取对齐帧
    aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
    color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧
    #aligned_depth_frame, color_frame= self.ur_robot.get_camera_data1() # meter

    ############### 相机参数的获取 #######################
    intr = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
    print("depth_intrin: " + str(depth_intrin) + '\n')
    '''camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
                        'ppx': intr.ppx, 'ppy': intr.ppy,
                        'height': intr.height, 'width': intr.width,
                        'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
                        }'''

    # 保存内参到本地
    # with open('./intrinsics.json', 'w') as fp:
    # json.dump(camera_parameters, fp)
    #######################################################

    depth_image = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）
    depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)  # 深度图（8位）
    depth_image_3d = np.dstack(
        (depth_image_8bit, depth_image_8bit, depth_image_8bit))  # 3通道深度图
    color_image = np.asanyarray(color_frame.get_data())  # RGB图

    # 返回相机内参、深度参数、彩色图、深度图、齐帧中的depth帧
    return intr, depth_intrin, color_image, depth_image, aligned_depth_frame, depth_image_3d


# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image
 
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        #aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        #color_frame = aligned_frames.get_color_frame()
        #depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        intr, depth_intrin, color_image, depth_image, aligned_depth_frame, depth_image_3d = get_aligned_images()
        
        # Validate that both frames are valid
        if not aligned_depth_frame or not aligned_frames:
            continue
 
        depth_colored_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # 调整帧的大小
        # aligned_frames = cv2.resize(aligned_frames, (new_width, new_height))
        # 使用模型进行目标检测,并返回相应数据
        results_list = model.predict(source=color_image)

        # 获取每个结果对象并进行处理
        for results in results_list:
            if results.boxes is not None:
                xyxy_boxes = results.boxes.xyxy
                conf_scores = results.boxes.conf
                cls_ids = results.boxes.cls

                for box, conf, cls_id in zip(xyxy_boxes, conf_scores, cls_ids):
                    x1, y1, x2, y2 = map(int, box)
                    cls_id = int(cls_id)
                    label = model.names[cls_id]
                    confidence = f"{conf:.2f}"

                    ux = int((x1 + x2)/2)
                    uy = int((y1 + y2)/2)
                    dis = aligned_depth_frame.get_distance(ux,uy)
                    camera_xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, (ux,uy), dis)
                    camera_xyz = np.round(np.array(camera_xyz), 3)
                    camera_xyz = camera_xyz.tolist()
                    print(str(label) + " camera_xyz: " + str(camera_xyz) + '\n')
                    cv2.circle(color_image, (ux,uy), 4, (255,255,255), 5)
                    cv2.putText(color_image, str(camera_xyz), (ux,uy), 0, 1, [255,255,255])
                    # 颜色
                    rectangle_color = (0, 255, 0)
                    label_color = (0, 0, 255)

                    # 在图像上绘制矩形框和标签
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), rectangle_color, 2)
                    cv2.putText(color_image, f"{label} {confidence}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color,2)



        # the size of color_frame is (720,1280,3)
        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
        # depth image is 1 channel, color is 3 channels
        # depth_image_3d shape is (720,1280,3)
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
        # the size of bg_removed is (720,1280,3)
        # Render images:
        #   depth align to color on left
        #   depth on right
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((bg_removed, depth_colormap))

        # 显示图像
        cv2.imshow("Color Image", color_image)
        cv2.imshow("depth_colored_image", depth_colored_image)


        # 将帧写入输出视频
        # out.write(frame)

        # 按'q'键保存图像并退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # 保存图像
            cv2.imwrite("/home/jetson/zzh/ultralytics-main/boluo/test/output/" + "rgb" + str(c) + ".png", color_image)
            cv2.imwrite("/home/jetson/zzh/ultralytics-main/boluo/test/output/" + "depth" + str(c) + ".png", depth_image)
            cv2.imwrite("/home/jetson/zzh/ultralytics-main/boluo/test/output/" + "depth_colored" + str(c) + ".png", depth_colored_image)
            c = c + 1
            print("photo " + str(c) + " shot success!")
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    
        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', images)

    cv2.destroyAllWindows()

finally:
    pipeline.stop()