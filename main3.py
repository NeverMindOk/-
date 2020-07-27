# coding=utf-8
"""
瞌睡检测
"""
import numpy as np
import cv2 as cv
import dlib
from scipy.spatial import distance
from imutils import face_utils
import time
import face_alignment
from tof_set import *
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
session = tf.compat.v1.Session(config=config)

# 设置session
tf.compat.v1.keras.backend.set_session(session)


use_CNN_model = 1
if use_CNN_model:
    from model_inceptionV3 import load_model
    weights_path = 'models/inception_v3_tof_1589359435.374173.h5'
    model = load_model(weights_path)
label = ['Normal', 'Watch phone', 'Call phone','Sleep', 'Nobody']
detector = dlib.get_frontal_face_detector()
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")


# 设定人眼标定点
LeftEye_Start = 36
LeftEye_End = 41
RightEye_Start = 42
RightEye_End = 47

# 设定嘴巴标定点
Mouth_Start = 60
Mouth_End = 68

Radio = 0.23
Radio_max=0.30
last=0.23
Mouth_Radio_th = 0.5
Low_radio_constant = 5  # 意味着连续多少帧横纵比小于Radio小于阈值时，判断疲劳
# Mouth_Low_radio_constant = 3
# 头部姿态估计
# 镜头内参
# 接近于普通RGB相机的默认参数
K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]
# 测试用的广角RGB参数
# K = [8.2893e+002, 0.0, 3.1950000000000000e+002,
#      0.0, 8.28462e+002, 2.3950000000000000e+002,
#      0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)
object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])

reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]

alarm = False  # 初始化警报
frame_counter = 0  # 连续帧计数

mouth_alarm = False  # 初始化警报
mouth_frame_counter = 0  # 连续帧计数

pose_alarm = False  # 初始化警报
pose_frame_counter = 0  # 连续帧计数

head_alarm = 0
head_frame_counter = 0

def get_head_pose(shape):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    _, rotation_vec, translation_vec = cv.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv.Rodrigues(rotation_vec)
    pose_mat = cv.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle
# 头部姿态估计end

def calculate_Ratio(eye):
    """
    计算眼睛横纵比
    """
    d1 = distance.euclidean(eye[1], eye[5])
    d2 = distance.euclidean(eye[2], eye[4])
    d3 = distance.euclidean(eye[0], eye[3])
    ratio = (d1 + d2) / (2 * d3)
    return ratio

def calculate_Mouth_Ratio(mouth):
    """
    计算嘴部横纵比
    """
    d1 = distance.euclidean(mouth[1], mouth[7])
    d2 = distance.euclidean(mouth[2], mouth[6])
    d3 = distance.euclidean(mouth[3], mouth[5])
    d4 = distance.euclidean(mouth[0], mouth[4])
    ratio = (d1 + d2 + d3) / (3 * d4)
    return ratio

def rotated_Img(predictor, img):
    """
    预旋转
    """
    left_eye_x = predictor[36][0]  # 36对应左眼角
    left_eye_y = predictor[36][1]
    right_eye_x = predictor[45][0]  # 45对应右眼角
    right_eye_y = predictor[45][1]
    center = predictor[30][0], predictor[30][1]
    dy = left_eye_y - right_eye_y
    dx = left_eye_x - right_eye_x
    angle = np.arctan(dy / dx) * 180 / np.pi
    warp_matrix = cv.getRotationMatrix2D(center, angle, 1)
    rotated = cv.warpAffine(img, warp_matrix, (img.shape[1], img.shape[0]))
    return rotated


def rot_20(img):
    rows, cols, channel = img.shape
    M1 = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 20, 1)
    M2 = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), -20, 1)
    dst1 = cv.warpAffine(img, M1, (cols, rows))
    dst2 = cv.warpAffine(img, M2, (cols, rows))
    return img, dst1, dst2

def ret_des(frame):
    # frame, frame20, frame340 = rot_20(frame)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rects = detector(gray, 0)  # 人脸检测

    # print(len(rects))
    # if len(rects) == 0:
    #     gray = cv.cvtColor(frame20, cv.COLOR_BGR2GRAY)
    #     rects = detector(gray, 0)  # 人脸检测
    # if len(rects) == 0:
    #     gray = cv.cvtColor(frame340, cv.COLOR_BGR2GRAY)
    #     rects = detector(gray, 0)  # 人脸检测
    return gray, rects


def drowsinessDetection(frame,img_amp):
    global alarm, frame_counter, mouth_alarm, mouth_frame_counter, head_alarm, head_frame_counter, pose_alarm, pose_frame_counter,last

        
    face_rects = detector(img_amp, 0)

    preds = fa.get_landmarks(frame)
    pose_alarm= False
    if preds!=None:
        shape = preds[0]        

        reprojectdst, euler_angle = get_head_pose(shape)

        for (x, y) in shape:
            cv.circle(frame, (x, y), 1, (0, 0, 255), -1)

        for start, end in line_pairs:
            cv.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))

        angle_x, angle_y, angle_z = euler_angle[0, 0], euler_angle[1, 0], euler_angle[2, 0]
            
        # 头部角度显示
        cv.putText(frame, "X: " + "{:7.2f}".format(angle_x), (20, 20), cv.FONT_HERSHEY_SIMPLEX,
                           0.75, (255, 255, 255), thickness=2)
        cv.putText(frame, "Y: " + "{:7.2f}".format(angle_y), (20, 50), cv.FONT_HERSHEY_SIMPLEX,
                           0.75, (255, 255, 255), thickness=2)
        cv.putText(frame, "Z: " + "{:7.2f}".format(angle_z), (20, 80), cv.FONT_HERSHEY_SIMPLEX,
                           0.75, (255, 255, 255), thickness=2)
        # 头部姿态估计END
        points1 = np.zeros((68, 2), dtype=int)
        for j in range(68):
                points1[j] = (shape[j][0], shape[j][1])

        if len(face_rects)>0:
            shape2 = predictor(img_amp, face_rects[0])
            shape3 = face_utils.shape_to_np(shape2)
            for (x, y) in shape3:
                cv.circle(img_amp, (x, y), 1, (0, 0, 255), -1)
            points2 = np.zeros((68, 2), dtype=int)
            for j in range(68):
                points2[j] = (shape2.part(j).x, shape2.part(j).y)
            points=points2
        else:
            points=points1

        # 获取眼睛特征点
            
        Lefteye = points[LeftEye_Start: LeftEye_End + 1]
        Righteye = points[RightEye_Start: RightEye_End + 1]

        # 计算眼睛横纵比
        Lefteye_Ratio = calculate_Ratio(Lefteye)
        Righteye_Ratio = calculate_Ratio(Righteye)
        mean_Ratio = (Lefteye_Ratio + Righteye_Ratio) / 2  # 计算两眼平均比例
        if mean_Ratio<Radio_max:
            last=mean_Ratio
        else:
            mean_Ratio=last
        # 计算凸包
        left_eye_hull = cv.convexHull(Lefteye)
        right_eye_hull = cv.convexHull(Righteye)

        # 绘制轮廓
        if len(face_rects)>0:
            cv.drawContours(img_amp, [left_eye_hull], -1, [0, 255, 0], 1)
            cv.drawContours(img_amp, [right_eye_hull], -1, [0, 255, 0], 1)
        else: 
            cv.drawContours(frame, [left_eye_hull], -1, [0, 255, 0], 1)
            cv.drawContours(frame, [right_eye_hull], -1, [0, 255, 0], 1)

        # 眨眼判断

        if mean_Ratio < Radio:
            frame_counter += 1
            if frame_counter >= Low_radio_constant:
                # 发出警报
                if not alarm:
                    alarm = True

                cv.putText(frame, "DROWSINESS!", (10, 30),
                                    cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            alarm = False
            frame_counter = 0


        points=points1
        # 获取嘴巴特征点
        Mouth = points[Mouth_Start: Mouth_End + 1]
        # 计算嘴巴纵横比
        Mouth_Ratio = calculate_Mouth_Ratio(Mouth)
        # 计算嘴巴凸包
        mouth_hull = cv.convexHull(Mouth)
        # 绘制嘴巴轮廓
        cv.drawContours(frame, [mouth_hull], -1, [0, 255, 0], 1)
        # 哈欠判断
        if Mouth_Ratio > Mouth_Radio_th:
            mouth_frame_counter += 1
            if mouth_frame_counter >= Low_radio_constant:
                # 发出警报
                if not mouth_alarm:
                    mouth_alarm = True

                cv.putText(frame, "YAWN!", (10, 60),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            mouth_alarm = False
            mouth_frame_counter = 0

        if angle_x > 10:
            head_alarm = 1
            # head_frame_counter += 1
            cv.putText(frame, "OverLook!(@*@)", (20, 140), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),
                               thickness=2)
        elif abs(angle_y) > 20 and mouth_alarm == False:
            head_alarm = 2
            # head_frame_counter += 1
            cv.putText(frame, "LOOK Side!(@@~)", (20, 110), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),
                               thickness=2)
        else:
            head_alarm = 0
            head_frame_counter = 0

        # 显示结果
        cv.putText(frame, "Eye_Ratio{:.2f}".format(mean_Ratio), (100  + 300, 30),
                        cv.FONT_HERSHEY_SIMPLEX, 0.7, [100  + 0, 0, 255], 2)
        cv.putText(frame, "Mouth_Ratio{:.2f}".format(Mouth_Ratio), (100  + 300, 60),
                        cv.FONT_HERSHEY_SIMPLEX, 0.7, [100  + 0, 0, 255], 2)
        cv.putText(frame, "Radio{:.2f}".format(Radio), (100  + 300, 90),
                        cv.FONT_HERSHEY_SIMPLEX, 0.7, [100  + 0, 0, 255], 2)
            
    else:
        pose_frame_counter+=1
        if pose_frame_counter >= Low_radio_constant:
                # 发出警报
                if not pose_alarm:
                    pose_alarm = True

                cv.putText(frame, "Wrong pose!", (10, 90),
                                cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame, alarm, mouth_alarm, head_alarm, pose_alarm, img_amp

class DelayOutput:
    def __init__(self, delay_frame=15):
        self.label_list = np.zeros(delay_frame, dtype=np.uint8)
        self.di = 0
        self.delay_frame = delay_frame

    def after_delay(self,label):
        self.label_list[self.di] = label
        self.di =  (self.di+1) % self.delay_frame

        counts = np.bincount(self.label_list)
        if counts[0] < self.delay_frame - 3:
            show_label = np.argmax(counts[1:]) + 1
        else:
            show_label = 0

        return show_label

class rgb_DelayOutput:
    def __init__(self, delay_frame=15):
        self.label_list = np.zeros(delay_frame, dtype=np.uint8)
        self.di = 0
        self.delay_frame = delay_frame

    def after_delay(self,label):
        self.label_list[self.di] = label
        self.di =  (self.di+1) % self.delay_frame

        counts = np.bincount(self.label_list)
        show_label = np.argmax(counts)

        return show_label

dl = DelayOutput()

def tof_main(head_alarm,img_to_pred):
    """
    ToF 主函数
    """
    img_show = img_to_pred.copy()
    # cv2.imwrite('ToF'+'/dep_{}.png'.format(time.time()),img_show)
    predict_label = 0
    if use_CNN_model:
        img_to_pred = np.expand_dims(img_to_pred, axis=0)
        y_pred = model.predict(img_to_pred)[0]
        if head_alarm==1:
            y_pred[1]+=0.2

        predict_label=np.argmax(y_pred)
        max_prob = np.max(y_pred)
        if max_prob < 0.55:
            predict_label = 0

    show_label = dl.after_delay(predict_label)
    return img_show, show_label


def result_show():
    panl = np.ones((240, 240, 3), dtype='uint8') * 255
    x0, x1, x2 = 10, 30, 50

    cv2.putText(panl, "Head Position: ", (x0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 0, 1)
    cv2.putText(panl, "Drowsiness State:", (x0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 0, 1)
    cv2.putText(panl, "Action State:", (x0, 170), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 0, 1)

    # RGB Alarm
    if pose_alarm == 1 and predict_label!=4:
        cv2.putText(panl, "Wrong pose!", (x2, 55), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        cv2.circle(panl, (x1, 50), 10, (0, 0, 255), -1)

        color_tof = (0,255,0) if predict_label == 0 else (0,0,255)
        cv2.putText(panl, label[predict_label], (x2, 195), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color_tof, 1)
        cv2.circle(panl, (x1, 190), 10, color_tof, -1)

    else:
        if predict_label != 4:    
            if head_alarm == 1:
                cv2.putText(panl, "OVERLO.OK!", (x2, 55), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                cv2.circle(panl, (x1, 50), 10, (0, 0, 255), -1)
            elif head_alarm==2:
                cv2.putText(panl, "LO.OKsIde", (x2, 55), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (50, 220, 230), 1)
                cv2.circle(panl, (x1, 50), 10, (50, 220, 230), -1)
            else:
                cv2.putText(panl, "Normal", (x2, 55), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
                cv2.circle(panl, (x1, 50), 10, (0, 255, 0), -1)


            if eye_alarm:
                if mouth_alarm:
                    cv2.putText(panl, "You are Sleepy!", (x2, 125), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                    cv2.circle(panl, (x1, 120), 10, (0, 0, 255), -1)
                else:
                    cv2.putText(panl, "Drowsiness!", (x2, 125), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                    cv2.circle(panl, (x1, 120), 10, (0, 0, 255), -1)
            else:
                if mouth_alarm:
                    cv2.putText(panl, "yaWn!", (x2, 125), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                    cv2.circle(panl, (x1, 120), 10, (0, 0, 255), -1)
                else:
                    cv2.putText(panl, "Normal", (x2, 125), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
                    cv2.circle(panl, (x1, 120), 10, (0, 255, 0), -1)
            # ToF Alarm
            color_tof = (0,255,0) if predict_label == 0 else (0,0,255)
            cv2.putText(panl, label[predict_label], (x2, 195), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color_tof, 1)
            cv2.circle(panl, (x1, 190), 10, color_tof, -1)
        else:
            if pose_alarm == 1:
                color_tof = (0,255,0) if predict_label == 0 else (0,0,255)
                cv2.putText(panl, label[predict_label], (x2, 195), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color_tof, 1)
                cv2.circle(panl, (x1, 190), 10, color_tof, -1) 
            else:
                color_tof = (0,255,0)
                cv2.putText(panl, 'Normal', (x2, 195), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color_tof, 1)
                cv2.circle(panl, (x1, 190), 10, color_tof, -1) 


        # ToF Alarm
        color_tof = (0,255,0) if predict_label == 0 else (0,0,255)
        cv2.putText(panl, label[predict_label], (x2, 195), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color_tof, 1)
        cv2.circle(panl, (x1, 190), 10, color_tof, -1)


    # 图像拼接
    show_left = np.concatenate((panl, tof_frame), axis=0)
    show_right = np.concatenate((RGB_frame,Gray_frame),axis=1)
    show_img = np.concatenate((show_left,show_right), axis=1)
    cv2.imshow('Driver Action Result', show_img)


"""
主函数
"""
cap = cv.VideoCapture(0 + cv.CAP_DSHOW)  # 0摄像头摄像
eye_dl = rgb_DelayOutput(5)
mouth_dl = rgb_DelayOutput(5)
head_dl = rgb_DelayOutput(10)
pose_dl = rgb_DelayOutput(5)
while True:
    ret1, frame = cap.read()  # 读取每一帧
    frame = cv.flip(frame, 1)

    # cv.imwrite('RGB'+'/dep_{}.png'.format(time.time()),frame)  
    finfo = dmcam.frame_t()
    ret2 = dmcam.cap_get_frames(dev, 1, f, finfo)
    # fps = 10   #保存视频的FPS，可以适当调整
    # size=(640,480)
    ## 可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg: sudo apt-get install ffmepg
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # videoWriter = cv2.VideoWriter('test.avi',fourcc,fps,size)#最后一个是保存图片的尺寸

    if ret1 and ret2:
        w = finfo.frame_info.width
        h = finfo.frame_info.height
        dist_cnt, dist = dmcam.frame_get_distance(dev, w * h, f, finfo.frame_info)
        gray_cnt, gray = dmcam.frame_get_gray(dev, w * h, f, finfo.frame_info)
        img_amp = gray.reshape(h, w)
        img_amp = cv2.convertScaleAbs(img_amp, None, 1 / 2)
        img_amp = cv.flip(img_amp, 1)
        img_amp = cv.merge([img_amp] * 3)
        RGB_frame, eye_alarm0, mouth_alarm0, head_alarm0, pose_alarm0, Gray_frame = drowsinessDetection(frame,img_amp)


        eye_alarm = eye_dl.after_delay(eye_alarm0)
        mouth_alarm = mouth_dl.after_delay(mouth_alarm0)
        head_alarm = head_dl.after_delay(head_alarm0)
        pose_alarm = pose_dl.after_delay(pose_alarm0)

        dep0 = (dist.reshape(h, w) * 1000)
        size=(240,240)
        img_dep = cv.convertScaleAbs(dep0, None, 1 / 16)
        img_dep = cv.resize(img_dep, size, interpolation=cv2.INTER_AREA)
        img_dep = cv.flip(img_dep, 1)
        img_to_pred = cv.merge([img_dep] * 3)
        img_show = img_to_pred.copy()

        tof_frame, predict_label = tof_main(head_alarm,img_to_pred)
        result_show()
    key = cv2.waitKey(1) & 0xFF
    if  key == ord('q') or key==27:
        cap.release()
        break
    
cap.release()
# videoWriter.release()
cv.destroyAllWindows()

