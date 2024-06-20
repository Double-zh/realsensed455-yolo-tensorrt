# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
 
# 初始化摄像头
cap = cv2.VideoCapture(1)  # 0 通常是默认摄像头的标识
 
# 用来保存文件的编号
count = 1
 
while True:
    # 从摄像头读取一帧
    ret, frame = cap.read()
 
    # 如果正确读取帧，ret为True
    if not ret:
        print("无法读取摄像头。")
        break
 
    # 显示帧
    cv2.imshow('frame', frame)
 
    # 如果按下'q'键，保存图片并退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite(f'image_{count}.jpg', frame)
        break
    
    count += 1
 
# 释放摄像头资源
cap.release()
# 关闭所有OpenCV窗口
cv2.destroyAllWindows()