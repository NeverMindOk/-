import sys, os
import numpy as np
import time, cv2

import dmcam

# --  init the lib with default log file
dmcam.init(None)
# --  init with specified log file
# dmcam.init("test.log")

# -- set debug level
dmcam.log_cfg(dmcam.LOG_LEVEL_INFO, dmcam.LOG_LEVEL_DEBUG, dmcam.LOG_LEVEL_NONE)

# -- list device
print(" Scanning dmcam device ..")
devs = dmcam.dev_list()
if devs is None:
    print(" No device found")
    sys.exit(1)

print("found %d device" % len(devs))

for i in range(len(devs)):
    print("#%d: %s" % (i, dmcam.dev_get_uri(devs[i], 256)[0]))

print(" Open dmcam device ..")
# open the first device
dev = dmcam.dev_open(devs[0])
# Or open by URI
# dev = dmcam.dev_open_by_uri("xxxx")
assert dev is not None

# - set capture config  -
cap_cfg = dmcam.cap_cfg_t()
cap_cfg.cache_frames_cnt = 10  # framebuffer = 10
cap_cfg.on_error = None  # use cap_set_callback_on_error to set cb
cap_cfg.on_frame_rdy = None  # use cap_set_callback_on_frame_ready to set cb
cap_cfg.en_save_replay = True  # True = save replay, False = not save
cap_cfg.en_save_dist_u16 = False  # True to save dist stream for openni replay
cap_cfg.en_save_gray_u16 = False  # True to save gray stream for openni replay
cap_cfg.fname_replay = os.fsencode("dm_replay.oni")  # set replay filename

dmcam.cap_config_set(dev, cap_cfg)
# dmcam.cap_set_callback_on_frame_ready(dev, on_frame_rdy)
# dmcam.cap_set_callback_on_error(dev, on_cap_err)

print(" Set paramters ...")
wparams = {
    dmcam.PARAM_INTG_TIME: dmcam.param_val_u(),
    dmcam.PARAM_FRAME_RATE: dmcam.param_val_u(),
}
wparams[dmcam.PARAM_INTG_TIME].intg.intg_us = 200
wparams[dmcam.PARAM_FRAME_RATE].frame_rate.fps = 10

# DMCAM_FILTER_ID_AMP
amp_min_val = dmcam.filter_args_u()
amp_min_val.min_amp = 0  # 40
if not dmcam.filter_enable(dev, dmcam.DMCAM_FILTER_ID_AMP, amp_min_val, sys.getsizeof(amp_min_val)):
    print(" set amp to %d %% failed" % 0)
if not dmcam.filter_disable(dev, dmcam.DMCAM_FILTER_ID_FLYNOISE):
    print(" disable fly noise filter failed")

if not dmcam.param_batch_set(dev, wparams):
    print(" set parameter failed")

print(" Start capture ...")
dmcam.cap_start(dev)

f = bytearray(640 * 480 * 4 * 2)

if __name__ == '__main__':
    count = 0
    run = True
    while run:
        # get one frame
        finfo = dmcam.frame_t()
        ret = dmcam.cap_get_frames(dev, 1, f, finfo)
        # print("get %d frames" % ret)
        if ret > 0:
            w = finfo.frame_info.width
            h = finfo.frame_info.height

            print(" frame @ %d, %d, %dx%d" %
                  (finfo.frame_info.frame_idx, finfo.frame_info.frame_size, w, h))

            dist_cnt, dist = dmcam.frame_get_distance(dev, w * h, f, finfo.frame_info)
            gray_cnt, gray = dmcam.frame_get_gray(dev, w * h, f, finfo.frame_info)
            # dist = dmcam.raw2dist(int(len(f) / 4), f)
            # gray = dmcam.raw2gray(int(len(f) / 4), f)
            img_dep = (dist.reshape(h, w) * 1000)
            img_amp = gray.reshape(h, w)
            cv2.imshow('dep', cv2.convertScaleAbs(img_dep, None, 1 / 8))
            cv2.imshow('ir', cv2.convertScaleAbs(img_amp, None, 1 / 2))

            count += 1
            # if count >= 300:
            #     break

        else:
            break
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    print(" Stop capture ...")
    dmcam.cap_stop(dev)

    print(" Close dmcam device ..")
    dmcam.dev_close(dev)

    dmcam.uninit()
