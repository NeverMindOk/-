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


# 是否使用GPU
UES_GPU = False #True

if UES_GPU == True:  # 调用CNN人脸检测器
    detector = dlib.cnn_face_detection_model_v1('models/mmod_human_face_detector.dat')
else:  # 调用HOG+SVM人脸检测器
    detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
# five_landmarks_predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat") #暂时无用

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
                            shape[8]])

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

def drowsinessDetection(cap):
    global alarm, frame_counter, mouth_alarm, mouth_frame_counter, head_alarm, head_frame_counter, pose_alarm, pose_frame_counter,last
    last=0.23
    ret, frame = cap.read()  # 读取每一帧
    frame = cv.flip(frame, 1)
    # cv.imwrite('RGB'+'/dep_{}.png'.format(time.time()),frame)    

    if ret:
    
        gray, rects = ret_des(frame)
        face_rects = detector(frame, 0)
        # if len(face_rects) == 0:
        #     face_rects = detector(frame, 1)
        # 头部姿态估计
        # 预旋转解决歪头问题
        # if len(face_rects) > 0:
        #     shape = predictor(frame, face_rects[0])
        #     shape = face_utils.shape_to_np(shape)
        #     frame = rotated_Img(predictor=shape, img=frame)
        if len(face_rects) > 0:
            pose_alarm=False
            if UES_GPU:
                for i, d in enumerate(face_rects):
                    face = d.rect
            else:
                face = face_rects[0]
            shape = predictor(frame, face)
            shape = face_utils.shape_to_np(shape)

            reprojectdst, euler_angle = get_head_pose(shape)

            for (x, y) in shape:
                cv.circle(frame, (x, y), 1, (0, 0, 255), -1)

            for start, end in line_pairs:
                cv.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))

            angle_x, angle_y, angle_z = euler_angle[0, 0], euler_angle[1, 0], euler_angle[2, 0]
            if angle_x > 10:
                head_alarm = 1
                # head_frame_counter += 1
                cv.putText(frame, "OverLook!(@*@)", (20, 140), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),
                           thickness=2)
            elif abs(angle_y) > 20:
                head_alarm = 2
                # head_frame_counter += 1
                cv.putText(frame, "LOOK Side!(@@~)", (20, 110), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),
                           thickness=2)
            else:
                head_alarm = 0
                head_frame_counter = 0
            # 头部角度显示
            cv.putText(frame, "X: " + "{:7.2f}".format(angle_x), (20, 20), cv.FONT_HERSHEY_SIMPLEX,
                       0.75, (255, 255, 255), thickness=2)
            cv.putText(frame, "Y: " + "{:7.2f}".format(angle_y), (20, 50), cv.FONT_HERSHEY_SIMPLEX,
                       0.75, (255, 255, 255), thickness=2)
            cv.putText(frame, "Z: " + "{:7.2f}".format(angle_z), (20, 80), cv.FONT_HERSHEY_SIMPLEX,
                       0.75, (255, 255, 255), thickness=2)
        # 头部姿态估计END
        else:
            pose_frame_counter+=1
            if pose_frame_counter >= Low_radio_constant:
                    # 发出警报
                    if not pose_alarm:
                        pose_alarm = True

                    cv.putText(frame, "Wrong pose!", (10, 90),
                                cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # gray = cv.equalizeHist(gray)
        for i, rect in enumerate(rects):
            if UES_GPU:
                rect = rect.rect
            else:
                rect = rect
            shape = predictor(gray, rect)
            points = np.zeros((68, 2), dtype=int)
            for j in range(68):
                points[j] = (shape.part(j).x, shape.part(j).y)

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
            cv.drawContours(frame, [left_eye_hull], -1, [0, 255, 0], 1)
            cv.drawContours(frame, [right_eye_hull], -1, [0, 255, 0], 1)

            # 眨眼判断

            if mean_Ratio < Radio:
                frame_counter += 1
                if frame_counter >= Low_radio_constant:
                    # 发出警报
                    if not alarm:
                        alarm = True

                    # cv.putText(frame, "DROWSINESS!", (10, 30),
                    #             cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                alarm = False
                frame_counter = 0

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

                    # cv.putText(frame, "YAWN!", (10, 60),
                    #            cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                mouth_alarm = False
                mouth_frame_counter = 0
            # 显示结果
            # cv.putText(frame, "Eye_Ratio{:.2f}".format(mean_Ratio), (100 * i + 300, 30),
            #            cv.FONT_HERSHEY_SIMPLEX, 0.7, [100 * i + 0, 0, 255], 2)
            # cv.putText(frame, "Mouth_Ratio{:.2f}".format(Mouth_Ratio), (100 * i + 300, 60),
            #            cv.FONT_HERSHEY_SIMPLEX, 0.7, [100 * i + 0, 0, 255], 2)
            # cv.putText(frame, "Radio{:.2f}".format(Radio), (100 * i + 300, 90),
            #            cv.FONT_HERSHEY_SIMPLEX, 0.7, [100 * i + 0, 0, 255], 2)

        # cv.imshow("Are You Sleepy?", frame)
    return frame, alarm, mouth_alarm, head_alarm, pose_alarm

def main():
    """
    主函数
    """
    cap = cv.VideoCapture(0 + cv.CAP_DSHOW)  # 0摄像头摄像
    
    # fps = 10   #保存视频的FPS，可以适当调整
    # size=(640,480)
    ## 可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg: sudo apt-get install ffmepg
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # videoWriter = cv2.VideoWriter('test.avi',fourcc,fps,size)#最后一个是保存图片的尺寸

    while cap.isOpened():
        frame = drowsinessDetection(cap)[0]
        cv.imshow("lilili", frame)
        # videoWriter.write(frame)
        cv.imwrite('eye'+'/{}.png'.format(time.time()),frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    # videoWriter.release()
    cv.destroyAllWindows()



if __name__ == '__main__':
    main()
