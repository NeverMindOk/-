# -*- coding: utf-8 -*-
import sys
import cv2
import dmcam
import numpy as np

use_CNN_model = 1
if use_CNN_model:
    from model_inceptionV3 import load_model
    weights_path = 'models/inceptionV3_tof_1575287325.7145457.h5'
    model = load_model(weights_path)
label = ['Normal', 'Watch phone', 'Call phone', 'Sleep']


def read_tof_params(filename):
    params = []
    with open(filename, 'r', encoding='utf-8') as fileid:
        for line in fileid:
            params.append(int(line.split()[0]))  # no of the recorded frames
    return params

# 相机初始化
# 积分时间，HDR时间，FPS，幅值滤波
parms = [100,800,10,5]


dmcam.init(None)
dmcam.log_cfg(dmcam.LOG_LEVEL_INFO,
              dmcam.LOG_LEVEL_DEBUG, dmcam.LOG_LEVEL_NONE)

devs = dmcam.dev_list()
if devs is None:
    print(" No device found")
    sys.exit(1)

dev = dmcam.dev_open(None)
# dmcam.cap_set_frame_buffer(dev, None, 320 * 240 * 4 * 10)
dmcam.cap_set_callback_on_frame_ready(dev, None)
dmcam.cap_set_callback_on_error(dev, None)

# show batch param set
print("-> batch param parameters write...\n ")
wparams = {
    dmcam.PARAM_INTG_TIME: dmcam.param_val_u(),
    dmcam.PARAM_HDR_INTG_TIME: dmcam.param_val_u(),
    dmcam.PARAM_FRAME_RATE: dmcam.param_val_u()
}
wparams[dmcam.PARAM_INTG_TIME].intg.intg_us = parms[0]  # 1400
wparams[dmcam.PARAM_HDR_INTG_TIME].intg.intg_us = parms[1]  # 1400
wparams[dmcam.PARAM_FRAME_RATE].frame_rate.fps = parms[2]  # 10

## Oprerate fileter
# DMCAM_FILTER_ID_AMP
amp_min_val = dmcam.filter_args_u()
amp_min_val.min_amp = parms[3]  # 40
if not dmcam.filter_enable(dev, dmcam.DMCAM_FILTER_ID_AMP, amp_min_val, sys.getsizeof(amp_min_val)):
    print(" set amp to %d %% failed" % 0)
if not dmcam.filter_disable(dev, dmcam.DMCAM_FILTER_ID_MEDIAN):
    print(" disable median filter failed")
hdr = dmcam.filter_args_u()
if not dmcam.filter_enable(dev, dmcam.DMCAM_FILTER_ID_HDR, hdr, sys.getsizeof(hdr)):
    print(" enable hdr filter failed")

ret = dmcam.param_batch_set(dev, wparams)
assert ret is True

# show batch param get
print("-> batch param parameters reading...\n")
params_to_read = list(range(dmcam.PARAM_ENUM_COUNT))
param_val = dmcam.param_batch_get(dev, params_to_read)
assert param_val is not None
print("dev_mode = %d" % param_val[dmcam.PARAM_DEV_MODE].dev_mode)
print("mode_freq = %d" % param_val[dmcam.PARAM_MOD_FREQ].mod_freq)
print("vendor: %s" % param_val[dmcam.PARAM_INFO_VENDOR].info_vendor)
print("product: %s" % param_val[dmcam.PARAM_INFO_PRODUCT].info_product)
print("max frame info: %d x %d, depth=%d, fps=%d, intg_us=%d"
      % (param_val[dmcam.PARAM_INFO_CAPABILITY].info_capability.max_frame_width,
         param_val[dmcam.PARAM_INFO_CAPABILITY].info_capability.max_frame_height,
         param_val[dmcam.PARAM_INFO_CAPABILITY].info_capability.max_frame_depth,
         param_val[dmcam.PARAM_INFO_CAPABILITY].info_capability.max_fps,
         param_val[dmcam.PARAM_INFO_CAPABILITY].info_capability.max_intg_us))
print([hex(v) for v in param_val[dmcam.PARAM_INFO_SERIAL].info_serial.serial])
print("version: sw:%d, hw:%d, sw2:%d, hw2:%d"
      % (param_val[dmcam.PARAM_INFO_VERSION].info_version.sw_ver,
         param_val[dmcam.PARAM_INFO_VERSION].info_version.hw_ver,
         param_val[dmcam.PARAM_INFO_VERSION].info_version.sw2_ver,
         param_val[dmcam.PARAM_INFO_VERSION].info_version.hw2_ver))
print("frame format = %d" % param_val[dmcam.PARAM_FRAME_FORMAT].frame_format.format)
print("fps = %d" % param_val[dmcam.PARAM_FRAME_RATE].frame_rate.fps)
print("illum_power=%d %%" % param_val[dmcam.PARAM_ILLUM_POWER].illum_power.percent)
print("intg = %d us" % param_val[dmcam.PARAM_INTG_TIME].intg.intg_us)

print("tl:%.2f, tr:%.2f, bl:%.2f, br:%.2f, ib:%.2f\n"
      % (param_val[dmcam.PARAM_TEMP].temp.tl_cal / 10,
         param_val[dmcam.PARAM_TEMP].temp.tr_cal / 10,
         param_val[dmcam.PARAM_TEMP].temp.bl_cal / 10,
         param_val[dmcam.PARAM_TEMP].temp.br_cal / 10,
         param_val[dmcam.PARAM_TEMP].temp.ib_cal / 10
         ))

print(" Start capture ...")
dmcam.cap_start(dev)

# img = np.frombuffer(file_dep.read(2 * 320 * 240 * f_cnt), dtype=np.uint16)
f = bytearray(320 * 240 * 4 * 2)



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

dl = DelayOutput()
cut_edge = 40

## 主程序入口
def tof_main(head_alarm):
    """
    ToF 主函数
    """

    finfo = dmcam.frame_t()
    ret = dmcam.cap_get_frames(dev, 1, f, finfo)
    # print("get %d frames" % ret)
    if ret > 0:
        w = finfo.frame_info.width
        h = finfo.frame_info.height
        dist_cnt, dist = dmcam.frame_get_distance(dev, w * h, f, finfo.frame_info)
        gray_cnt, gray = dmcam.frame_get_gray(dev, w * h, f, finfo.frame_info)

        if dist_cnt == w * h:
            # -----------------------------------------
            dep0 = (dist.reshape(h, w) * 1000)
            Z = gray.reshape(h, w)

            img_dep = cv2.convertScaleAbs(dep0, None, 1 / 16)
            img_to_pred = img_dep[:, cut_edge:320 - cut_edge]
            img_to_pred = cv2.merge([img_to_pred] * 3)
            img_show = img_to_pred.copy()

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



if __name__=='__main__':
    
    while 1:
        tof_frame, predict_label = tof_main(0)


        color_tof = (0,255,0) if predict_label == 0 else (0,0,255)
        cv2.putText(tof_frame, label[predict_label], (60, 25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color_tof, 1)
        cv2.circle(tof_frame, (40, 20), 10, color_tof, -1)
        cv2.imshow('tof ', tof_frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
