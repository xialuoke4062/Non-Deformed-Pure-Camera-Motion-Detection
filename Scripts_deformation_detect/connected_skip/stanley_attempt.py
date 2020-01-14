# 64a3c6b83d6f71afc66c42529268598b9bbd4f05
import time

import cv2
import numpy as np
from pathlib import Path
# from albumentations.pytorch.functional import img_to_tensor
# import torch
from skimage import data
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from matplotlib import cm
import tqdm
from PIL import Image
from collections import Counter
import os

def check_cut(frame, video_path):
    info = ""
    cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = canny(cur_gray, 2, 1, 25)

    # fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    # ax = axes.ravel()
    frequency = edges.sum(axis=0)
    mean = int(np.mean(frequency))
    half_len = len(edges[0])//2
    left = np.where(frequency[20:half_len] == max(frequency[20:half_len]))[0][0] + 20
    right = np.where(frequency[half_len:] == max(frequency[half_len:]))[0][0] + half_len
    LRatio = abs(frequency[left]-mean)//mean
    RRatio = abs(frequency[right]-mean)//mean

    ver_freq = edges.sum(axis=1)
    half_height = len(edges)//2
    high = np.where(ver_freq[:half_height] == max(ver_freq[:half_height]))[0][0]
    low = np.where(ver_freq[half_height:] == max(ver_freq[half_height:]))[0][0] + half_height
    if LRatio == 0 or RRatio == 0:
        return -1, -1
    # cur_gray[:, left] = 255
    # cur_gray[:, right] = 255
    # cur_gray[high] = 255
    # cur_gray[low] = 255
    #
    # ax[0].imshow(cur_gray, cmap=cm.gray)
    # ax[0].set_title('Input image')
    #
    # ax[1].imshow(edges, cmap=cm.gray)
    # ax[1].set_title('Canny edges')
    #
    # ax[2].imshow(edges * 0)
    #
    # ax[2].plot(ver_freq)
    # ax[2].set_xlim((0, cur_gray.shape[1]))
    # ax[2].set_ylim((cur_gray.shape[0], 0))
    # ax[2].set_title('Probabilistic Hough: L{} Lf{} R{} Rf{} u{} LRatio{} RRatio{}'
    #                 .format(left, frequency[left], right, frequency[right], mean, LRatio, RRatio))
    #
    # # plt.title(str(video_path))
    # plt.tight_layout()
    # # plt.savefig('/Users/xwang169/Desktop/Output/'+video_path.name[:-4]+'.png')
    # plt.savefig('/Users/apple/Desktop/Output/'+video_path.name[:-4]+'.png')
    # plt.show()
    return left, right, high, low

# def check_cut(frame, video_path):
#     info = ""
#     cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     high_thresh, thresh_im = cv2.threshold(cur_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     low_thresh = 0.5 * high_thresh
#     edges = cv2.Canny(cur_gray, low_thresh, high_thresh, apertureSize=3, L2gradient=True)
#
#     # info += str(low_thresh) + " " + str(high_thresh) + '\n'
#     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 10, minLineLength=5, maxLineGap=3)
#     if lines is None:
#         print("Jessie is" + str(video_path))
#         return -1, -1
#
#     for x1, y1, x2, y2 in lines[0]:
#         cv2.line(cur_gray, (x1, y1), (x2, y2), (255, 255, 255), 2)
#
#     """Left cut with image"""
#     half = len(cur_gray[0]) // 2
#     count_thres = 45
#     black_thres = 20
#     col_most_common_pair = [Counter(cur_gray[:, i]).most_common(1)[0]
#                             for i in range(half)]
#     col_most_common = []
#     for idx, elem in enumerate(col_most_common_pair):
#         if idx > 30:
#             if elem[1] < count_thres or elem[0] > black_thres:
#                 break
#         col_most_common.append(elem[0])
#     mc = Counter(col_most_common).most_common(1)[0][0]
#     last_mc = len(col_most_common) - 1 - col_most_common[::-1].index(mc)
#     mc2, last_mc2 = -1, -1
#     if len(Counter(col_most_common).most_common(2)) == 2:
#         mc2 = Counter(col_most_common).most_common(2)[1][0]
#         last_mc2 = len(col_most_common) - 1 - col_most_common[::-1].index(mc2)
#         info += "last_mc: {}, last_mc2: {}".format(last_mc, last_mc2) + '\n'
#         if (mc - 1 == mc2 or mc2 == mc + 1) and last_mc2 > last_mc:
#             last_mc = last_mc2
#
#     plt.figure()
#     # info += str(col_most_common_pair[20:30]) + '\n' + str(col_most_common_pair[30:40]) + '\n' + \
#     #     " len: " + str(len(col_most_common)) + \
#     #     " mc: " + str(mc) + " last_mc: " + str(last_mc) + " mc2: " + str(mc2) + " last_mc2: " + str(last_mc2)
#     # if last_mc > 40:
#     #     info = str(col_most_common_pair[last_mc-10:last_mc]) + \
#     #        '\n' + str(col_most_common_pair[last_mc:last_mc+10]) + '\n' + \
#     #        " len: " + str(len(col_most_common)) + \
#     #        " mc: " + str(mc) + " last_mc: " + str(last_mc) + " mc2: " + str(mc2) + " last_mc2: " + str(last_mc2)
#
#     """Right cut with image"""
#     col_most_common_pairrr = [Counter(cur_gray[:, i]).most_common(1)[0]
#                               for i in reversed(range(len(cur_gray[0]) - 35, len(cur_gray[0])))]
#     col_most_commonrr = []
#     for idx, elem in enumerate(col_most_common_pairrr):
#         if idx > 5:
#             if elem[1] < count_thres or elem[0] > black_thres:
#                 break
#         col_most_commonrr.append(elem[0])
#     mcrr = Counter(col_most_commonrr).most_common(1)[0][0]
#     last_mcrr = len(col_most_commonrr) - 1 - col_most_commonrr[::-1].index(mcrr) + 1
#
#     # plt.figure()
#     # info += str(col_most_common_pairrr[-10:]) + '\n' + str(col_most_common_pairrr[-20:-10]) + '\n' + \
#     #        " len: " + str(len(col_most_commonrr)) + \
#     #        " mc: " + str(mcrr) + " last_mc: " + str(last_mcrr) + " mc2: " + str(mc2rr) + " last_mc2: " + str(last_mc2rr)
#     # cur_gray[:, -last_mcrr] = 255
#     # cur_gray[:, last_mc] = 255
#     # cur_gray = cur_gray[:, last_mc:]
#     img = Image.fromarray(cur_gray)
#     plt.imshow(img, cmap='gray')
#     plt.title(info)
#     plt.show()
#     # print(info)
#     return last_mc, last_mcrr

def calc_optical_flow(prev_frame, frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    opt_flow = np.zeros((prev_gray.shape[0], prev_gray.shape[1], 2), dtype=np.float32)
    flow = cv2.calcOpticalFlowFarneback(prev=prev_gray, next=cur_gray, flow=opt_flow, pyr_scale=0.5,
                                        levels=3, winsize=35, iterations=3, poly_n=5, poly_sigma=1.1,
                                        flags=0)
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_display = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    mean_flow = np.mean(flow_display)
    return flow, opt_flow, mean_flow

def generate(video_path):
    min_length = 10
    skips = [20]
    r_thresholds = [0.5]
    for skip in skips:
        for r_threshold in r_thresholds:
            """Make Video Directory if not exist"""
            # print(video_path)
            result_path = Path(str(video_path)[:-4])
            if not result_path.exists():
                result_path.mkdir()
            # else:
            #     continue

            """Check if path and video is valid"""
            if video_path.exists():
                video_fp = cv2.VideoCapture(str(video_path))
            else:
                print("Video file {} does not exists".format(str(video_path)))
                raise IOError
            if not video_fp.isOpened():
                print("Error opening video file {}".format(str(video_path)))
                raise IOError

            """Video parameter"""
            fps = video_fp.get(cv2.CAP_PROP_FPS)
            width = int(video_fp.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video_fp.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # print("frame rate is: {}, dim(h * w) = {} * {}".format(fps, height, width))
            total_frame_count = int(video_fp.get(cv2.CAP_PROP_FRAME_COUNT))
            tq = tqdm.tqdm(total=int(total_frame_count), dynamic_ncols=False, ncols=125)
            tq.set_description("Video {}".format(video_path.name))

            '''Faster Query_2D Build'''
            valid_query_2D_locations = []

            """Algorithm Variable [2]"""
            sequence_count = 0  # Index of the Video Clipped
            no_move_detection_count = 0
            video_sampling_rate = 5  # To update tqdm
            frame = None
            prev_frame = None
            result_video_fp = None
            warming_imgs = np.zeros((3, height, width, 3), dtype=np.uint8)
            holding_imgs = []
            recording_imgs = []
            ratios = []
            means = []
            caches = []
            state = "searching"
            checked = 0  # checked==1, read_failed==2
            left_cut, right_cut, high_cut, low_cut = 0, 0, 0, 0

            while video_fp.isOpened():
                """
                    Set Local Variable ratio for each frame
                    Set frame from previous loop to be prev_frame
                    Read next frame and check if the read is valid
                """
                ratio = 0
                if frame is not None:
                    caches.append(frame)
                    caches.pop(0)
                    prev_frame = caches[0]
                else:
                    for _ in range(video_sampling_rate):
                        ret, frame = video_fp.read()
                        if not ret:
                            checked = 2
                            break
                        if not checked:
                            left_cut, right_cut, high_cut, low_cut = check_cut(frame, video_path)
                            if left_cut != -1:  # No black boundary
                                width = right_cut - left_cut
                                height = low_cut - high_cut
                                i_coords, j_coords = np.meshgrid(range(height), range(width), indexing='ij')
                                valid_query_2D_locations = np.concatenate(
                                    [i_coords.reshape((-1, 1)), j_coords.reshape((-1, 1))], axis=1)
                            checked = 1  # should change back to 1
                        frame = frame[high_cut:low_cut, left_cut:right_cut, :]
                        caches.append(frame)
                        tq.update(1)
                    prev_frame = caches[0]
                if checked == 2:
                    break
                ret, frame = video_fp.read()
                if not ret:
                    break
                frame = frame[high_cut:low_cut, left_cut:right_cut, :]
                """ Calculate dense optical flow using L-K method """
                flow, opt_flow, mean_flow = calc_optical_flow(prev_frame, frame)
                means.append(mean_flow / 100)

                ''' Faster Train_2D Build '''
                temp_train_2D = valid_query_2D_locations + opt_flow.reshape(height * width, 2)
                valid_train_2D_locations = temp_train_2D.reshape(height * width, 2)

                """ 
                    Uniformly sample points
                    Run RANSAC Alrogithm
                    Calculate Inlier ratio
                """
                if mean_flow <= 15:
                    ratio = 0
                else:
                    try:
                        arr_1 = valid_query_2D_locations
                        arr_2 = valid_train_2D_locations
                        arr_1 = arr_1[::skip]
                        arr_2 = arr_2[::skip]
                        model, inliers = ransac(data=(arr_1, arr_2),
                                                model_class=FundamentalMatrixTransform,
                                                # min_samples=8, residual_threshold=4, max_trials=1)
                                                min_samples=8, residual_threshold=r_threshold, max_trials=20)
                        ratio = float(inliers.sum()) / float(arr_1.shape[0])  # Inlier Sum / Total Sampled Points
                    except Exception as e:
                        ratio = 0
                        print(e)
                        pass

                '''Report to tqdm'''
                frame_index = int(video_fp.get(cv2.CAP_PROP_POS_FRAMES))
                tq.set_postfix_str('I_T_ratio={:.5f}, frame={}, video_name={}'
                                   .format(ratio, frame_index, video_path.name[:-4]))

                if state == "searching":
                    tq.set_description("Status: searching")
                    if ratio >= ratio_threshold:
                        state = "warming1"
                elif state == "warming1":
                    tq.set_description("Status: warming1")
                    if ratio >= ratio_threshold:
                        state = "warming2"
                    else:
                        state = "searching"
                elif state == "warming2":
                    tq.set_description("Status: warming2")
                    if ratio >= ratio_threshold:
                        state = "warming3"
                    else:
                        state = "searching"
                elif state == "warming3":
                    tq.set_description("Status: warming3")
                    if ratio >= ratio_threshold:
                        recording_imgs.extend(caches)  # Could get problematic
                        recording_imgs.append(frame)
                        state = "recording"
                    else:
                        state = "searching"
                elif state == "recording":
                    tq.set_description("Status: {}".format(state))
                    if ratio < ratio_threshold:
                        holding_imgs = [frame]
                        no_move_detection_count = 0
                        state = "holding"
                    else:
                        recording_imgs.append(frame)
                elif state == "holding":
                    tq.set_description("Status: {}".format(state))
                    holding_imgs.append(frame)
                    if ratio < ratio_threshold:
                        no_move_detection_count += 1
                        if no_move_detection_count > max_no_move_count:
                            if len(recording_imgs) > min_length:
                                result_video_fp = cv2.VideoWriter(
                                    str(result_path / "{}.mp4".format(sequence_count)),
                                    cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width, height))
                                for j in range(len(recording_imgs)):
                                    result_video_fp.write(recording_imgs[j])
                                sequence_count += 1
                                result_video_fp.release()
                            recording_imgs = []
                            holding_imgs = []
                            no_move_detection_count = 0
                            state = "searching"
                    else:
                        no_move_detection_count = 0
                        for k in range(len(holding_imgs)):
                            recording_imgs.append(holding_imgs[k])
                        holding_imgs = []
                        state = "recording"
                ratios.append(ratio)
                tq.update(1)
                # print(frame_index, ratio)
                # if frame_index == 80: break

            if checked == 2:
                break
            """Writing the last portion of video left from the while loop"""
            if len(recording_imgs) > min_length:  # At least 10
                result_video_fp = cv2.VideoWriter(str(result_path / "{}.mp4".format(sequence_count)),
                                                  cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width, height))
                for j in range(len(recording_imgs)):
                    result_video_fp.write(recording_imgs[j])
                result_video_fp.release()

            """ Plotting
                Two x-axises are plotted: 
                    Bottom is Frame Number.
                    Top is Time in video (seconds), calculated by (Frame Number / Frame Rate)
            """
            plt.figure()
            plt.plot(np.arange(len(ratios)), ratios)
            plt.plot(np.arange(len(means)), means)
            ax1 = plt.gca()
            ax1.set_xlabel(r"Frame Number")
            ax2 = ax1.twiny()
            ax1Xs = ax1.get_xticks()

            def caltime(x):
                v = (video_sampling_rate + x) / fps * 1
                return ["%.1f" % z for z in v]

            ax2.set_xlim(ax1.get_xlim())
            ax2.set_xticklabels(caltime(ax1Xs))
            ax2.set_xlabel(r"Time in video (s)")
            """ 
                Plotting 
                Horizontal and Vertical Lines
            """
            plt.hlines(ratio_threshold, xmin=0, xmax=len(ratios), colors='g', linestyles='dotted')
            # targets = [33,57,58,59,69]
            # for i in targets:
            #     plt.hlines(y=ratios[i], xmin=0, xmax=i, colors='g', linestyles='dotted')
            #
            # plt.vlines(x=targets,ymin=0,ymax=1,colors='y',linestyles='dotted')
            """Plotting the rest"""
            title = "residual_threshold={}, skip={}, frame_rate={}, min_length={}". \
                format(r_threshold, skip, fps, min_length)
            plt.title(title)
            plt.tight_layout()
            plt.savefig(str(result_path / "thres{}_skip_{}.png".format(r_threshold, skip)))
            plt.show()
            plt.close()
            tq.close()


if __name__ == "__main__":
    np.random.seed(0)

    """Algorithm Hyper-Parameter [1]"""
    ratio_threshold = 0.72
    max_no_move_count = 5
    fast_forward_num = 1

    """Video Loading Path"""
    alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    alpha = "B"
    for elem in alpha:
        path_str = "/Users/xwang169/Downloads/videos/" + elem
        path_str = "/Users/apple/Desktop/Xingtong_2/www.gastrointestinalatlas.com/videos/" + elem
        video_root = Path(path_str)
        video_path_list = list(video_root.glob("*.mpg"))
        # video_path_list = [
        #     Path(os.path.join(video_root, "CaesophagusSurgwery2.mpg")),
        #     Path(os.path.join(video_root, "CaesophagusSurgwery3.mpg")),
        #     Path(os.path.join(video_root, "CaesophagusSurgwery4.mpg")),
        # ]

        # video_path_list = [
        #     Path(os.path.join(video_root, "Dieulafoyxz4.mpg")),
        #     Path(os.path.join(video_root, "divertdr4.mpg")),
        #     Path(os.path.join(video_root, "divertdr13.mpg")),
        #     Path(os.path.join(video_root, "divertdr14.mpg")),
        #     Path(os.path.join(video_root, "DiverVarix3.mpg")),
        #     Path(os.path.join(video_root, "DiverVarix4.mpg")),
        #     Path(os.path.join(video_root, "DuodenalUlcercromat1.mpg")),
        #     Path(os.path.join(video_root, "DuodUlcerGigant3.mpg")),
        #     Path(os.path.join(video_root, "duoodenalinjection1.mpg")),
        # ]

        # video_path_list = [
        #     # Path(os.path.join(video_root, "BrunnerGlandAdenomax1.mpg")),
        #     # Path(os.path.join(video_root, "BigDivert2.mpg")),
        #     # Path(os.path.join(video_root, "BigDivert4.mpg")),
        #     # Path(os.path.join(video_root, "BlackEsophagus4.mpg")),
        #     # Path(os.path.join(video_root, "BBAArreett7.mpg")),
        #     # Path(os.path.join(video_root, "BalloonH2.mpg")),
        #     Path(os.path.join(video_root, "BalloonH3.mpg"))
        # ]
        c = cpu_count()
        pool = Pool(c)
        pool.map(generate, [video_path for video_path in video_path_list])
        pool.close()
