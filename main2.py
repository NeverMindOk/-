# -*- coding: utf-8 -*-

from tof_run2 import *
from Drowsiness_detection3 import *




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
    show_img = np.concatenate((show_left,RGB_frame), axis=1)
    cv2.imshow('Driver Action Result', show_img)


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


# cap = cv2.VideoCapture(0)  # 0摄像头摄像.
cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
eye_dl = rgb_DelayOutput(5)
mouth_dl = rgb_DelayOutput(5)
head_dl = rgb_DelayOutput(10)
pose_dl = rgb_DelayOutput(5)

while cap.isOpened():
    # RGB
    RGB_frame, eye_alarm0, mouth_alarm0, head_alarm0, pose_alarm0 = drowsinessDetection(cap)

    eye_alarm = eye_dl.after_delay(eye_alarm0)
    mouth_alarm = mouth_dl.after_delay(mouth_alarm0)
    head_alarm = head_dl.after_delay(head_alarm0)
    pose_alarm = pose_dl.after_delay(pose_alarm0)

    # ToF
    tof_frame, predict_label = tof_main(head_alarm)
    result_show()

    key = cv2.waitKey(1) & 0xFF
    if  key == ord('q') or key==27:
        cap.release()
        break

cv2.destroyAllWindows()

