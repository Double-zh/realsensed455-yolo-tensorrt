# -*- coding: utf-8 -*-
"""
Created on Mon May 27 11:12:39 2024

@author: dudongjie
"""
import pyrealsense2 as rs
import numpy as np
import cv2
 
# 创建一个pipeline
pipeline = rs.pipeline()
 
# 创建配置对象
config = rs.config()
 
 
# 启动pipeline
pipeline.start(config)

c = 0
try:
    while True:
        # 获取帧数据
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
 
        if not depth_frame or not color_frame:
            continue
 
        # 转换为OpenCV图像
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_colored_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

 
        # 显示图像
        cv2.imshow("Color Image", color_image)
        cv2.imshow("depth_image", depth_colored_image)

        # 按'q'键保存图像并退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # 保存图像
            cv2.imwrite(r"C:\Users\NINGMEI\Desktop\boluo\test/output/" + "rgb" + str(c) + ".png", color_image)
            cv2.imwrite(r"C:\Users\NINGMEI\Desktop\boluo\test/output/" + "depth" + str(c) + ".png", depth_image)
            cv2.imwrite(r"C:\Users\NINGMEI\Desktop\boluo\test/output/" + "depth_colored" + str(c) + ".png", depth_colored_image)

            c = c + 1

        if cv2.waitKey(1) & 0xFF == ord(' '):

            break
 
finally:
    # 停止pipeline
    pipeline.stop()
 
cv2.destroyAllWindows()
