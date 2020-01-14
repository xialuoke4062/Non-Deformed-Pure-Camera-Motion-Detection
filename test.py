# ######## Successfully Read #######
# import numpy as np
# import cv2
#
# fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# cap = cv2.VideoCapture("/Users/apple/Desktop/Xingtong_2/www.gastrointestinalatlas.com/videos/2polposjuve.mpg")
# fps = cap.get(cv2.CAP_PROP_FPS)
#
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
#
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
#     print(frame_index)
#     if ret==True:
#         frame = cv2.flip(frame,0)
#
#         # write the flipped frame
#         out.write(frame)
#     else:
#         break
#
# # Release everything if job is finished
# cap.release()
# out.release()

######## Multiprocessing #######
# 64a3c6b83d6f71afc66c42529268598b9bbd4f05
import multiprocessing

import cv2
import numpy as np
from pathlib import Path
# from albumentations.pytorch.functional import img_to_tensor
# import torch
from skimage import data
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
import matplotlib.pyplot as plt
import tqdm

# Local import

def slice_video(video_path):
    print(video_path)
    # video_path = Path("/Users/apple/Desktop/Xingtong_2/www.gastrointestinalatlas.com/videos/2polposjuve.mpg")
    # video_path = Path("/Users/apple/Desktop/Xingtong_2/www.gastrointestinalatlas.com/videos/3ulceras.mpg")

    result_path = Path(str(video_path)[:-4])
    if not result_path.exists():
        result_path.mkdir()
    # else:
    #     continue

    if video_path.exists():
        video_fp = cv2.VideoCapture(str(video_path))
        width = int(video_fp.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_fp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        print("Video file {} does not exists".format(str(video_path)))
        raise IOError
    warming_imgs = np.zeros((3, height, width, 3), dtype=np.uint8)
    if not video_fp.isOpened():
        print("Error opening video file {}".format(str(video_path)))
        raise IOError
    total_frame_count = int(video_fp.get(cv2.CAP_PROP_FRAME_COUNT))
    start_time = 0

    fps = video_fp.get(cv2.CAP_PROP_FPS)
    print("frame rate is: {}".format(fps))
    video_sampling_rate = 1
    # video_fp.set(cv2.CAP_PROP_POS_FRAMES, fps * start_time)
    frame = None
    prev_frame = None
    result_video_fp = None
    sequence_count = 0
    no_move_detection_count = 0
    state = "searching"
    tq = tqdm.tqdm(total=int(total_frame_count), dynamic_ncols=False, ncols=125)
    tq.set_description("Video {}".format(video_path.name))
    while video_fp.isOpened():
        frame_index = int(video_fp.get(cv2.CAP_PROP_POS_FRAMES))
        # print(frame_index)
        if frame is not None:
            prev_frame = frame

        ret, frame = video_fp.read()
        ttt = int(video_fp.get(cv2.CAP_PROP_POS_FRAMES))
        if not ret:
            break

        if prev_frame is not None:
            for _ in range(4):
                ret, frame = video_fp.read()
            # Calculate dense optical flow using L-K method
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            opt_flow = np.zeros((prev_gray.shape[0], prev_gray.shape[1], 2), dtype=np.float32)

            # ## Plotting 1 ##
            # cv2.calcOpticalFlowFarneback(prev=prev_gray, next=cur_gray, flow=opt_flow, pyr_scale=0.5, levels=3,
            #                              winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            # # flow_display = display_flow(opt_flow, 0.05)
            # # plt.imshow(flow_display)
            # # plt.show()
            # ## Plotting 1 End ##

            ## Plotting 2 ##
            hsv = np.zeros_like(frame)
            hsv[..., 1] = 255
            flow = cv2.calcOpticalFlowFarneback(prev=prev_gray, next=cur_gray, flow=opt_flow, pyr_scale=0.5, levels=3,
                                         winsize=35, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            flow_display = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            print(np.mean(flow_display))
            continue
            ## Plotting 2 End ##

            ## check image ##
            _, o1 = plt.subplots()
            _, o2 = plt.subplots()
            o1.imshow(opt_flow[...,0], cmap='gray')
            o2.imshow(opt_flow[...,1], cmap='gray')
            plt.show()
            cv2.imwrite("prev_{}.jpg".format(frame_index), prev_gray)
            cv2.imwrite("cur_{}.jpg".format(frame_index), cur_gray)
            cv2.imwrite("opt_flow_{}.jpg".format(frame_index), flow_display)
            continue
            ## check image ends ##

            ## Build query and train 2D ##
            valid_query_2D_locations = []
            valid_train_2D_locations = []
            for row in range(len(opt_flow)):
                for col in range(len(opt_flow[0])):
                    valid_query_2D_locations.append([row, col])
                    valid_train_2D_locations.append([row+opt_flow[row,col,0], col+opt_flow[row,col,1]])
            valid_query_2D_locations = np.array(valid_query_2D_locations)
            valid_train_2D_locations = np.array(valid_train_2D_locations)
            ## Build ends ##

            locations_2D_difference = np.linalg.norm(valid_query_2D_locations - valid_train_2D_locations, axis=1)
            large_motion_indexes = np.where(locations_2D_difference >= 2)[0]

            if len(large_motion_indexes) < 20:
                continue
            try:
                model, inliers = ransac(data=(valid_query_2D_locations[large_motion_indexes],
                                              valid_train_2D_locations[large_motion_indexes]),
                                        model_class=FundamentalMatrixTransform,
                                        min_samples=8, residual_threshold=0.5, max_trials=1)
                                        # min_samples=8, residual_threshold=4, max_trials=20)
            except Exception as e:
                pass
                print(e)
                # continue
            ransac_inliner_count = float(inliers.sum())
            large_motion_matching_count = float(len(large_motion_indexes))
            matching_count = float(valid_query_2D_locations.shape[0])
            ratio = ransac_inliner_count / matching_count
            tq.set_postfix_str('I_T_ratio={:.5f}, Max_diff={:.5f}'.format(ransac_inliner_count / matching_count,
                                                                      np.amax(locations_2D_difference)))

            if state == "searching":
                tq.set_description("Status: searching")
                if ratio >= ratio_threshold:
                    state = "warming1"
            elif state == "warming1":
                warming_imgs[0] = frame
                tq.set_description("Status: warming1")
                if ratio >= ratio_threshold:
                    state = "warming2"
                else:
                    state = "searching"
            elif state == "warming2":
                warming_imgs[1] = frame
                tq.set_description("Status: warming2")
                if ratio >= ratio_threshold:
                    state = "warming3"
                else:
                    state = "searching"
            elif state == "warming3":
                warming_imgs[2] = frame
                tq.set_description("Status: warming3")
                if ratio >= ratio_threshold:
                    result_video_fp = cv2.VideoWriter(str(result_path / "{}.mp4".format(sequence_count)),
                                                      cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                                      fps, (width, height))
                    for i in range(3):
                        result_video_fp.write(warming_imgs[i])

                    sequence_count += 1
                    state = "recording"
                else:
                    state = "searching"
            elif state == "recording":
                # video_sampling_rate = 1
                tq.set_description("Status: recording")
                result_video_fp.write(frame)
                if ratio < ratio_threshold:
                    no_move_detection_count = 0
                    state = "holding"
            elif state == "holding":
                tq.set_description("Status: holding")
                result_video_fp.write(frame)
                if ratio < ratio_threshold:
                    no_move_detection_count += 1
                    if no_move_detection_count > max_no_move_count:
                        result_video_fp.release()
                        no_move_detection_count = 0
                        state = "searching"
                else:
                    no_move_detection_count = 0
                    state = "recording"

            # video_fp.set(cv2.CAP_PROP_POS_FRAMES, frame_index + video_sampling_rate)
            ttttt = int(video_fp.get(cv2.CAP_PROP_POS_FRAMES))
            tq.update(video_sampling_rate)
        cv2.waitKey(1)
    tq.close()

if __name__ == "__main__":
    np.random.seed(0)

    ratio_threshold = 0.5 # 0.01
    height = 0
    width = 0
    max_no_move_count = 30 # 30
    fast_forward_num = 1

    video_root = Path("/Users/apple/Desktop/Xingtong_2/www.gastrointestinalatlas.com/videos")
    # video_path_list = list(video_root.glob("*.mpg"))
    ## BigDivert2, BrunnerGlandAdenomax1, Barretxx1
    video_path_list = [Path("/Users/apple/Desktop/Xingtong_2/www.gastrointestinalatlas.com/videos/B/Barretxx1/2.mp4")]
    # warming_imgs = np.zeros((3, height * 4, width * 4, 3), dtype=np.uint8)
    pool = multiprocessing.Pool()
    result = pool.map(slice_video, video_path_list)

# return [left], [right], max