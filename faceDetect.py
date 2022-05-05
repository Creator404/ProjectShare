# -*- coding: utf-8 -*-
from __future__ import unicode_literals
# 操作文件
import os
# 科学计算
import numpy as np
# 图像识别
import cv2 as cv
# 数据预处理, 该项目中只使用了标签编码
import sklearn.preprocessing as sp


def load_imgs(directory):
    '''加载 directory 该文件夹下所有以 .jpg 结尾的图片'''
    # 识别 系统环境 自动分配当前系统的路径分隔符并替换
    directory = os.path.normpath(directory)
    # 判断当前路径是否为存在
    if not os.path.isdir(directory):
        # 手动抛出异常 biu biu biu
        raise IOError("The directory '" + directory + "' doesn't exist!")
    # 创建图片集合 用于存储文件夹名和该文件夹下所有的图片
    faces = {}
    # os.walk(directory) 获取当前文件夹下所有的文件夹以及文件
    # curdir: 当前文件夹路径
    # subdirs: 当前文件夹下所有文件夹 (列表)
    # files: 当前文件夹下所有文件 (列表)
    for curdir, subdirs, files in os.walk(directory):
        # 首先便利所有的文件 筛选.jpg结尾文件并循环
        for jpeg in (file for file in files if file.endswith('.jpg')):
                # 拼接图片路径
            path = os.path.join(curdir, jpeg)
            # 获取该图片分类名称
            label = path.split(os.path.sep)[-2]
            # 判断当前key值是否存在图片集合中, 如果为空则创建该键并赋值空列表
            # 否则给图片集合中的 key 添加图片路径
            if label not in faces:
                faces[label] = []
            faces[label].append(path)
    # 返回图片集合
    return faces


def LBPHModel(fd, codec, train_path):
    '''
    -------------------
    参数说明: fd, codec, [model_path]
    fd: Haar-like(人脸特征模型对象)
    codec: LabelEncoder(标签编码器对象)
    model_path: 服用模型路径(功能未实现, 没找到读取的函数...)
    -------------------
    返回: 训练后的模型对象
    '''
    # 加载当前文件加下所有.jpg结尾的图片
    train_faces = load_imgs(train_path)  # 'traom_imgs'
    # 将所有标签放入编码器进行训练
    codec.fit(list(train_faces.keys()))
    # 创建空的训练集数组x y
    train_x, train_y = [], []
    # 循环所有训练组
    for label, filenames in train_faces.items():
        # 循环当前样本组中的图片
        for filename in filenames:
            # 读取图片
            image = cv.imread(filename)
            # 将图片转成灰度图
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            # 获取人脸特征位置
            faces = fd.detectMultiScale(
                gray, 1.1, 2, minSize=(100, 100))
            # 循环脸部特征数组
            for l, t, w, h in faces:
                # 将图片中的脸部特征裁剪下来
                train_x.append(gray[t:t + h, l:l + w])
                # 标签编码结果存储
                train_y.append(codec.transform([label])[0])
    train_y = np.array(train_y)
    # 创建LBPH人脸检测器
    model = cv.face.LBPHFaceRecognizer_create()
    # 对训练集进行训练
    model.train(train_x, train_y)
    return model


if __name__ == "__main__":
    # 训练集图片存储路径
    train_path = 'train_imgs'
    # 读取人脸描述文件,  构建人脸检测器
    fd = cv.CascadeClassifier('D:/OpenCV4/sources/samples/winrt_universal/VideoCaptureXAML/video_capture_xaml/video_capture_xaml.Windows/Assets/haarcascade_frontalface_alt.xml')
    # 创建标签编码器
    codec = sp.LabelEncoder()
    # 获取model
    model = LBPHModel(fd, codec, train_path)
    # 打开视频捕捉设备
    vc = cv.VideoCapture(0)
    while True:
        # 读取视频帧
        frame = vc.read()[1]
        # 反转图片
        frame = cv.flip(frame, 1)
        # print(frame)
        # 人脸位置检测, 返回数组
        faces = fd.detectMultiScale(frame, 1.3, 5)
        # 循环人脸位置数组
        for l, t, w, h in faces:
            # 给人脸描边
            cv.rectangle(frame, (l, t), (l + w, t + h),
                         (255, 0, 0), 4)
            # 复制原图片文本
            gray = frame.copy()
            # 将图片变化成灰度图
            gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)
            # 对面部特征进行识别
            pred_test_y = model.predict(gray[t:t + h, l:l + w])[0]
            # 将预测后的结果进行标签解码
            face_name = codec.inverse_transform([pred_test_y])[0]
            # 给图片添加文本 图片矩阵, 添加文本名称, 设置文本显示位置,
            # 字体样式, 字体大小, 字体颜色, 字体粗细
            cv.putText(frame, face_name, (l + 5, t - 15),
                       cv.FONT_HERSHEY_SIMPLEX, 1,
                       (255, 255, 255), 3)
            # 打印名称
            # print(face_name)

        # 显示图片
        cv.imshow('VideoCapture', frame)
        # 等待按下ESC键退出, 每次等待33毫秒
        if cv.waitKey(33) == 27:
            break
    # 关闭视频捕捉设备
    vc.release()
    # 关闭视频窗口
    cv.destroyAllWindows()