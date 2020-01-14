# 64a3c6b83d6f71afc66c42529268598b9bbd4f05
import pickle
import time
from multiprocessing import cpu_count, Pool, Manager
from textwrap import wrap

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
from itertools import product
import matplotlib.pyplot as plt
import tqdm
import os

def generate(skip, r_threshold, winsize):
    """Make Video Directory if not exist"""
    print(video_path)
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

    """Video parameters"""
    fps = video_fp.get(cv2.CAP_PROP_FPS)
    width = int(video_fp.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_fp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("frame rate is: {}, dim(h * w) = {} * {}".format(fps, height, width))
    total_frame_count = int(video_fp.get(cv2.CAP_PROP_FRAME_COUNT))
    tq = tqdm.tqdm(total=int(total_frame_count), dynamic_ncols=False, ncols=125)
    tq.set_description("Video {}".format(video_path.name))

    '''Faster Query_2D Build'''
    i_coords, j_coords = np.meshgrid(range(height), range(width), indexing='ij')
    valid_query_2D_locations = np.concatenate([i_coords.reshape((-1, 1)), j_coords.reshape((-1, 1))],
                                              axis=1)

    """Algorithm Variable [2]"""
    sequence_count = 0  # Index of the Video Clipped
    no_move_detection_count = 0
    video_sampling_rate = 9  # To update tqdm
    frame = None
    prev_frame = None
    result_video_fp = None
    warming_imgs = np.zeros((3, height, width, 3), dtype=np.uint8)
    holding_imgs = []
    recording_imgs = []
    ratios = []
    state = "searching"

    while video_fp.isOpened():
        """
            Set Local Variable ratio for each frame
            Set frame from previous loop to be prev_frame
            Read next frame and check if the read is valid
        """
        ratio = 0
        if frame is not None:
            prev_frame = frame
        ret, frame = video_fp.read()
        if not ret:
            break

        """ Bounding Frame index"""
        frame_index = int(video_fp.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_index < 90:
            continue
        if frame_index >= 221:
            break

        if prev_frame is not None:
            for _ in range(video_sampling_rate):
                ret, frame = video_fp.read()
                if not ret:
                    break
            if not ret:
                break
            """ Calculate dense optical flow using L-K method """
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            opt_flow = np.zeros((prev_gray.shape[0], prev_gray.shape[1], 2), dtype=np.float32)
            cv2.calcOpticalFlowFarneback(prev=prev_gray, next=cur_gray, flow=opt_flow, pyr_scale=0.5,
                                         levels=3, winsize=winsize, iterations=3, poly_n=5, poly_sigma=1.1,
                                         flags=0)

            ''' Faster Train_2D Build '''
            temp_train_2D = valid_query_2D_locations + opt_flow.reshape(height * width, 2)
            valid_train_2D_locations = temp_train_2D.reshape(height * width, 2)

            """ 
                Uniformly sample points
                Run RANSAC Alrogithm
                Calculate Inlier ratio
            """
            try:
                arr_1 = valid_query_2D_locations
                arr_2 = valid_train_2D_locations
                arr_1 = arr_1[::skip]
                arr_2 = arr_2[::skip]
                model, inliers = ransac(data=(arr_1, arr_2),
                                        model_class=FundamentalMatrixTransform,
                                        min_samples=8, residual_threshold=r_threshold, max_trials=20)
                ratio = float(inliers.sum()) / float(arr_1.shape[0])  # Inlier Sum / Total Sampled Points
            except Exception as e:
                ratio = 0
                print(e)
                pass

            '''Report to tqdm'''
            tq.set_postfix_str('I_T_ratio={:.5f}, frame={}'.format(ratio, frame_index))
            #
            # if state == "searching":
            #     tq.set_description("Status: searching")
            #     if ratio >= ratio_threshold:
            #         state = "warming1"
            # elif state == "warming1":
            #     warming_imgs[0] = frame
            #     tq.set_description("Status: warming1")
            #     if ratio >= ratio_threshold:
            #         state = "warming2"
            #     else:
            #         state = "searching"
            # elif state == "warming2":
            #     warming_imgs[1] = frame
            #     tq.set_description("Status: warming2")
            #     if ratio >= ratio_threshold:
            #         state = "warming3"
            #     else:
            #         state = "searching"
            # elif state == "warming3":
            #     warming_imgs[2] = frame
            #     tq.set_description("Status: warming3")
            #     if ratio >= ratio_threshold:
            #         for i in range(3):
            #             recording_imgs.append(warming_imgs[i])
            #         state = "recording"
            #     else:
            #         state = "searching"
            # elif state == "recording":
            #     tq.set_description("Status: {}".format(state))
            #     if ratio < ratio_threshold:
            #         holding_imgs = [frame]
            #         no_move_detection_count = 0
            #         state = "holding"
            #     else:
            #         recording_imgs.append(frame)
            # elif state == "holding":
            #     tq.set_description("Status: {}".format(state))
            #     holding_imgs.append(frame)
            #     if ratio < ratio_threshold:
            #         no_move_detection_count += 1
            #         if no_move_detection_count > max_no_move_count:
            #             if len(recording_imgs) > min_length:
            #                 result_video_fp = cv2.VideoWriter(
            #                     str(result_path / "{}.mp4".format(sequence_count)),
            #                     cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width, height))
            #                 for j in range(len(recording_imgs)):
            #                     result_video_fp.write(recording_imgs[j])
            #                 sequence_count += 1
            #                 result_video_fp.release()
            #             recording_imgs = []
            #             holding_imgs = []
            #             no_move_detection_count = 0
            #             state = "searching"
            #     else:
            #         no_move_detection_count = 0
            #         for k in range(len(holding_imgs)):
            #             recording_imgs.append(holding_imgs[k])
            #         holding_imgs = []
            #         state = "recording"
        ratios.append(ratio)
        # video_fp.set(cv2.CAP_PROP_POS_FRAMES, frame_index + video_sampling_rate)
        tq.update(video_sampling_rate)

    # """Writing the last portion of video left from the while loop"""
    # if len(recording_imgs) > min_length:
    #     result_video_fp = cv2.VideoWriter(str(result_path / "{}.mp4".format(sequence_count)),
    #                                       cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width, height))
    #     for j in range(len(recording_imgs)):
    #         result_video_fp.write(recording_imgs[j])
    #     result_video_fp.release()

    """ Plotting
        Two x-axises are plotted: 
            Bottom is Frame Number.
            Top is Time in video (seconds), calculated by (Frame Number / Frame Rate)
    """
    plt.plot(np.arange(len(ratios)), ratios)
    ax1 = plt.gca()
    ax1.set_ylim([0, 1])
    ax1.set_xlabel(r"Frame Number")
    ax2 = ax1.twiny()
    ax1Xs = ax1.get_xticks()

    def caltime(x):
        v = x / fps
        return ["%.1f" % z for z in v]

    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticklabels(caltime(ax1Xs))
    ax2.set_xlabel(r"Time in video (s)")
    """ 
        Plotting 
        Horizontal and Vertical Lines
    """

    bad_u, bad_std = np.mean(ratios), np.std(ratios)
    difs.append(bad_u)
    std_2s.append(bad_std)
    plt.hlines(bad_u, xmin=0, xmax=len(ratios), colors='r', linestyles='dotted')
    # targets = [33,57,58,59,69]
    # targets = [16]
    # for i in targets:
    #     plt.hlines(y=ratios[i], xmin=0, xmax=i, colors='g', linestyles='dotted')
    #
    # plt.vlines(x=targets, ymin=0, ymax=1, colors='purple', linestyles='dotted')
    """Plotting the rest"""
    title = "\n".join(wrap("thres={:.2f}, skip={}, winsize={}, frame_rate={}, min_length={}, "
                           "std_2={:.3f}".
                           format(r_threshold, skip, winsize, fps, min_length, bad_std), 60))
    print("thres={},skip={},winsize={}".format(r_threshold, skip, winsize))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(str(result_path / "sbad_skip_{}_thres{:.1f}_winsize_{}.png".format(skip, r_threshold, winsize)))
    # plt.show()
    plt.close()
    tq.close()

if __name__ == "__main__":
    np.random.seed(0)

    """Algorithm Hyper-Parameter [1]"""
    ratio_threshold = 0.4
    max_no_move_count = 5
    fast_forward_num = 1
    min_length = 10

    """Video Loading Path"""
    video_root = Path("/Users/xwang169/Downloads/videos/B")
    # video_root = Path("/Users/apple/Desktop/Xingtong_2/www.gastrointestinalatlas.com/videos/B")

    # video_path_list = list(video_root.glob("*.mpg"))
    video_path_list = [
        Path(os.path.join(video_root, "BBAArreett5.mpg"))
    ]
    for video_path in video_path_list:
        """Algorithm Hyper-Parameter Testing [1.5]"""
        skips = np.arange(10, 51, 10).tolist()  # 10, 81, 10
        r_thresholds = np.arange(0.25, 1.51, 0.25).tolist()  # 0.25, 1.51, 0.25
        winsizes = np.arange(10, 51, 10).tolist()  # 30, 81, 10

        # manager = Manager()
        # combos, difs, std_1s, std_2s = manager.list(), manager.list(), manager.list(), manager.list()
        # c = cpu_count()
        # pool = Pool(1)  # ??c??
        # pool.starmap(generate, [(skip, r_threshold, winsize) for skip, r_threshold, winsize in product(skips, r_thresholds, winsizes)])
        # pool.close()
        combos, difs, std_1s, std_2s = [], [], [], []
        for skip, r_threshold, winsize in product(skips, r_thresholds, winsizes):
            """Make Video Directory if not exist"""
            print(video_path)
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

            """Video parameters"""
            fps = video_fp.get(cv2.CAP_PROP_FPS)
            width = int(video_fp.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video_fp.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print("frame rate is: {}, dim(h * w) = {} * {}".format(fps, height, width))
            total_frame_count = int(video_fp.get(cv2.CAP_PROP_FRAME_COUNT))
            tq = tqdm.tqdm(total=int(total_frame_count), dynamic_ncols=False, ncols=125)
            tq.set_description("Video {}".format(video_path.name))

            '''Faster Query_2D Build'''
            i_coords, j_coords = np.meshgrid(range(height), range(width), indexing='ij')
            valid_query_2D_locations = np.concatenate([i_coords.reshape((-1, 1)), j_coords.reshape((-1, 1))],
                                                      axis=1)

            """Algorithm Variable [2]"""
            sequence_count = 0  # Index of the Video Clipped
            no_move_detection_count = 0
            video_sampling_rate = 9  # To update tqdm
            frame = None
            prev_frame = None
            result_video_fp = None
            warming_imgs = np.zeros((3, height, width, 3), dtype=np.uint8)
            holding_imgs = []
            recording_imgs = []
            ratios = []
            state = "searching"

            while video_fp.isOpened():
                """
                    Set Local Variable ratio for each frame
                    Set frame from previous loop to be prev_frame
                    Read next frame and check if the read is valid
                """
                ratio = 0
                if frame is not None:
                    prev_frame = frame
                ret, frame = video_fp.read()
                if not ret:
                    break

                """ Bounding Frame index"""
                frame_index = int(video_fp.get(cv2.CAP_PROP_POS_FRAMES))
                if frame_index < 90:
                    continue
                if frame_index >= 221:
                    break

                if prev_frame is not None:
                    for _ in range(video_sampling_rate):
                        ret, frame = video_fp.read()
                        if not ret:
                            break
                    if not ret:
                        break
                    """ Calculate dense optical flow using L-K method """
                    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    opt_flow = np.zeros((prev_gray.shape[0], prev_gray.shape[1], 2), dtype=np.float32)
                    cv2.calcOpticalFlowFarneback(prev=prev_gray, next=cur_gray, flow=opt_flow, pyr_scale=0.5,
                                                 levels=3, winsize=winsize, iterations=3, poly_n=5, poly_sigma=1.1,
                                                 flags=0)

                    ''' Faster Train_2D Build '''
                    temp_train_2D = valid_query_2D_locations + opt_flow.reshape(height * width, 2)
                    valid_train_2D_locations = temp_train_2D.reshape(height * width, 2)

                    """ 
                        Uniformly sample points
                        Run RANSAC Alrogithm
                        Calculate Inlier ratio
                    """
                    try:
                        arr_1 = valid_query_2D_locations
                        arr_2 = valid_train_2D_locations
                        arr_1 = arr_1[::skip]
                        arr_2 = arr_2[::skip]
                        model, inliers = ransac(data=(arr_1, arr_2),
                                                model_class=FundamentalMatrixTransform,
                                                min_samples=8, residual_threshold=r_threshold, max_trials=20)
                        ratio = float(inliers.sum()) / float(arr_1.shape[0])  # Inlier Sum / Total Sampled Points
                    except Exception as e:
                        ratio = 0
                        print(e)
                        pass

                    '''Report to tqdm'''
                    tq.set_postfix_str('I_T_ratio={:.5f}, frame={}'.format(ratio, frame_index))
                    #
                    # if state == "searching":
                    #     tq.set_description("Status: searching")
                    #     if ratio >= ratio_threshold:
                    #         state = "warming1"
                    # elif state == "warming1":
                    #     warming_imgs[0] = frame
                    #     tq.set_description("Status: warming1")
                    #     if ratio >= ratio_threshold:
                    #         state = "warming2"
                    #     else:
                    #         state = "searching"
                    # elif state == "warming2":
                    #     warming_imgs[1] = frame
                    #     tq.set_description("Status: warming2")
                    #     if ratio >= ratio_threshold:
                    #         state = "warming3"
                    #     else:
                    #         state = "searching"
                    # elif state == "warming3":
                    #     warming_imgs[2] = frame
                    #     tq.set_description("Status: warming3")
                    #     if ratio >= ratio_threshold:
                    #         for i in range(3):
                    #             recording_imgs.append(warming_imgs[i])
                    #         state = "recording"
                    #     else:
                    #         state = "searching"
                    # elif state == "recording":
                    #     tq.set_description("Status: {}".format(state))
                    #     if ratio < ratio_threshold:
                    #         holding_imgs = [frame]
                    #         no_move_detection_count = 0
                    #         state = "holding"
                    #     else:
                    #         recording_imgs.append(frame)
                    # elif state == "holding":
                    #     tq.set_description("Status: {}".format(state))
                    #     holding_imgs.append(frame)
                    #     if ratio < ratio_threshold:
                    #         no_move_detection_count += 1
                    #         if no_move_detection_count > max_no_move_count:
                    #             if len(recording_imgs) > min_length:
                    #                 result_video_fp = cv2.VideoWriter(
                    #                     str(result_path / "{}.mp4".format(sequence_count)),
                    #                     cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width, height))
                    #                 for j in range(len(recording_imgs)):
                    #                     result_video_fp.write(recording_imgs[j])
                    #                 sequence_count += 1
                    #                 result_video_fp.release()
                    #             recording_imgs = []
                    #             holding_imgs = []
                    #             no_move_detection_count = 0
                    #             state = "searching"
                    #     else:
                    #         no_move_detection_count = 0
                    #         for k in range(len(holding_imgs)):
                    #             recording_imgs.append(holding_imgs[k])
                    #         holding_imgs = []
                    #         state = "recording"
                ratios.append(ratio)
                # video_fp.set(cv2.CAP_PROP_POS_FRAMES, frame_index + video_sampling_rate)
                tq.update(video_sampling_rate)

            # """Writing the last portion of video left from the while loop"""
            # if len(recording_imgs) > min_length:
            #     result_video_fp = cv2.VideoWriter(str(result_path / "{}.mp4".format(sequence_count)),
            #                                       cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width, height))
            #     for j in range(len(recording_imgs)):
            #         result_video_fp.write(recording_imgs[j])
            #     result_video_fp.release()

            """ Plotting
                Two x-axises are plotted: 
                    Bottom is Frame Number.
                    Top is Time in video (seconds), calculated by (Frame Number / Frame Rate)
            """
            plt.plot(np.arange(len(ratios)), ratios)
            ax1 = plt.gca()
            ax1.set_ylim([0, 1])
            ax1.set_xlabel(r"Frame Number")
            ax2 = ax1.twiny()
            ax1Xs = ax1.get_xticks()


            def caltime(x):
                v = x / fps
                return ["%.1f" % z for z in v]


            ax2.set_xlim(ax1.get_xlim())
            ax2.set_xticklabels(caltime(ax1Xs))
            ax2.set_xlabel(r"Time in video (s)")
            """ 
                Plotting 
                Horizontal and Vertical Lines
            """

            bad_u, bad_std = np.mean(ratios), np.std(ratios)
            difs.append(bad_u)
            std_2s.append(bad_std)
            plt.hlines(bad_u, xmin=0, xmax=len(ratios), colors='r', linestyles='dotted')
            # targets = [33,57,58,59,69]
            # targets = [16]
            # for i in targets:
            #     plt.hlines(y=ratios[i], xmin=0, xmax=i, colors='g', linestyles='dotted')
            #
            # plt.vlines(x=targets, ymin=0, ymax=1, colors='purple', linestyles='dotted')
            """Plotting the rest"""
            title = "\n".join(wrap("thres={:.2f}, skip={}, winsize={}, frame_rate={}, min_length={}, "
                                   "std_2={:.3f}".
                                   format(r_threshold, skip, winsize, fps, min_length, bad_std), 60))
            print("thres={},skip={},winsize={}".format(r_threshold, skip, winsize))
            plt.title(title)
            plt.tight_layout()
            plt.savefig(str(result_path / "sbad_skip_{}_thres{:.1f}_winsize_{}.png".format(skip, r_threshold, winsize)))
            # plt.show()
            plt.close()
            tq.close()

        """ !!!!!!!!!!HAVE TO CONVERT MANAGER OBJECT BACK TO LIST!!!!!! """
        difs = list(difs)
        std_1s = list(std_1s)
        std_2s = list(std_2s)
        combos = list(combos)

        with open("bad_u.dat", "wb") as f2:
            pickle.dump(difs, f2)
        with open("std_2s.dat", "wb") as f2:
            pickle.dump(std_2s, f2)
        plt.plot(np.arange(len(difs)), difs)
        plt.plot(np.arange(len(std_2s)), std_2s)
        plt.legend(['bad u', 'bad std'], loc='upper left')
        ax1 = plt.gca()
        ax1.set_xlabel(r"Iterations")
        plt.savefig("Bad_Summary.png")
        plt.show()
        plt.close()