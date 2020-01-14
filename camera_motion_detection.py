# 64a3c6b83d6f71afc66c42529268598b9bbd4f05

import cv2
import numpy as np
from pathlib import Path
import torch
# from albumentations.pytorch.functional import img_to_tensor
from skimage import data
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
import matplotlib.pyplot as plt
import tqdm

# Local import
from . import models
from . import utils


def display_flow(flow, max_v):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0] / w, flow[:, :, 1] / h
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.uint8(np.minimum(v, max_v) * 1.0 / max_v * 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_dl_flow(flow, max_v):
    batch_size, channel, height, width = flow.shape
    flows_x_display = flow[0, 0].view(1, height, width)
    flows_y_display = flow[0, 1].view(1, height, width)

    flows_display = torch.cat([flows_x_display[0, :, :].view(1, flows_x_display.shape[1], flows_x_display.shape[2]),
                               flows_y_display[0, :, :].view(1, flows_x_display.shape[1], flows_x_display.shape[2])],
                              dim=0)
    flows_display = flows_display.data.cpu().numpy()
    flows_display = np.moveaxis(flows_display, source=[0, 1, 2], destination=[2, 0, 1])
    h, w = flows_display.shape[:2]
    fx, fy = flows_display[:, :, 0], flows_display[:, :, 1] * h / w
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.uint8(np.minimum(v, max_v) * 1.0 / max_v * 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def feature_model_import(trained_feature_model_path):
    # Feature architecture
    feature_model = models.FeatureFCDenseNet(
        in_channels=3, down_blocks=(3, 3, 3, 3, 3),
        up_blocks=(3, 3, 3, 3, 3), bottleneck_layers=4,
        growth_rate=8, out_chans_first_conv=16)
    # Initialize the network with Kaiming He initialization
    utils.init_net(feature_model, type="kaiming", mode="fan_in", activation_mode="relu",
                   distribution="normal")
    # Multi-GPU running
    feature_model = torch.nn.DataParallel(feature_model)

    if trained_feature_model_path is not None and trained_feature_model_path.exists():
        print("Loading {:s} ...".format(str(trained_feature_model_path)))
        pre_trained_state = torch.load(str(trained_feature_model_path))
        model_state = feature_model.state_dict()
        filtered_pre_trained_state = {k: v for k, v in pre_trained_state["model"].items() if k in model_state}
        for k, v in pre_trained_state["model"].items():
            if k in model_state:
                pass
            else:
                print(k)
        model_state.update(filtered_pre_trained_state)
        feature_model.load_state_dict(model_state)
    else:
        raise IOError

    return feature_model.module


def pair_feature_matching(feature_model, colors_1, colors_2, boundaries, feature_sampling_stride, matching_scale,
                          matching_threshold, cross_check_distance, display_matches=False):
    with torch.no_grad():
        batch_size, _, height, width = colors_1.shape
        rough_feature_maps_1, fine_feature_maps_1 = feature_model(colors_1)
        rough_feature_maps_2, fine_feature_maps_2 = feature_model(colors_2)
        rough_feature_maps_1 = rough_feature_maps_1 / torch.norm(rough_feature_maps_1,
                                                                 dim=1, keepdim=True)
        rough_feature_maps_2 = rough_feature_maps_2 / torch.norm(rough_feature_maps_2,
                                                                 dim=1, keepdim=True)
        fine_feature_maps_1 = fine_feature_maps_1 / torch.norm(fine_feature_maps_1,
                                                               dim=1, keepdim=True)
        fine_feature_maps_2 = fine_feature_maps_2 / torch.norm(fine_feature_maps_2,
                                                               dim=1, keepdim=True)
        location_1D_array, keypoint_list = utils.uniform_sample_generation(height, width,
                                                                           feature_sampling_stride)
        # sparse_flow_maps_1 = []
        # sparse_flow_masks_1 = []
        # sparse_flow_maps_2 = []
        # sparse_flow_masks_2 = []

        # Use these feature maps to calculate sparse stereo matching for
        # both input and output of the depth estimation network
        idx = 0
        sparse_flow_map_1, sparse_flow_map_2, sparse_flow_mask_1, sparse_flow_mask_2, \
        valid_query_2D_locations, valid_train_2D_locations, \
        display_matches_ai = utils.feature_matching_dl_flow_only(
            color_1=colors_1[idx],
            color_2=colors_2[idx],
            rough_feature_map_1=rough_feature_maps_1[idx],
            rough_feature_map_2=rough_feature_maps_2[idx],
            fine_feature_map_1=fine_feature_maps_1[idx],
            fine_feature_map_2=fine_feature_maps_2[idx],
            boundary=boundaries[idx],
            kps_1D_1=location_1D_array, scale=matching_scale,
            threshold=matching_threshold, cross_check_distance=cross_check_distance,
            kps_1=keypoint_list, gpu_id=0, display_matches=display_matches)
        # sparse_flow_maps_1.append(sparse_flow_map_1)
        # sparse_flow_masks_1.append(sparse_flow_mask_1)
        # sparse_flow_maps_2.append(sparse_flow_map_2)
        # sparse_flow_masks_2.append(sparse_flow_mask_2)
        # sparse_flow_maps_1 = torch.cat(sparse_flow_maps_1, dim=0)
        # sparse_flow_masks_1 = torch.cat(sparse_flow_masks_1, dim=0)
        # sparse_flow_maps_2 = torch.cat(sparse_flow_maps_2, dim=0)
        # sparse_flow_masks_2 = torch.cat(sparse_flow_masks_2, dim=0)

        return sparse_flow_map_1, sparse_flow_mask_1, sparse_flow_map_2, sparse_flow_mask_2, \
               valid_query_2D_locations, valid_train_2D_locations, display_matches_ai


# def display_matches_after_ransac(img_left, img_right, keypoints_left, keypoints_right, matches, inliers):
#     print("Number of matches:", matches.shape[0])
#     print("Number of inliers:", inliers.sum())
#
#     # Visualize the results.
#     fig, ax = plt.subplots(nrows=1, ncols=1)
#
#     plt.gray()
#
#     plot_matches(ax, img_left, img_right, keypoints_left, keypoints_right,
#                  matches[inliers], only_matches=True)
#     ax.axis("off")
#     ax.set_title("Inlier correspondences")
#
#     plt.show()
#
#     return


def example():
    img_left, img_right, groundtruth_disp = data.stereo_motorcycle()
    img_left, img_right = map(rgb2gray, (img_left, img_right))

    # Find sparse feature correspondences between left and right image.

    descriptor_extractor = ORB()

    descriptor_extractor.detect_and_extract(img_left)
    keypoints_left = descriptor_extractor.keypoints
    descriptors_left = descriptor_extractor.descriptors

    descriptor_extractor.detect_and_extract(img_right)
    keypoints_right = descriptor_extractor.keypoints
    descriptors_right = descriptor_extractor.descriptors

    matches = match_descriptors(descriptors_left, descriptors_right,
                                cross_check=True)

    # Estimate the epipolar geometry between the left and right image.
    model, inliers = ransac((keypoints_left[matches[:, 0]],
                             keypoints_right[matches[:, 1]]),
                            FundamentalMatrixTransform, min_samples=8,
                            residual_threshold=1, max_trials=1000)

    inlier_keypoints_left = keypoints_left[matches[inliers, 0]]
    inlier_keypoints_right = keypoints_right[matches[inliers, 1]]

    print("Number of matches:", matches.shape[0])
    print("Number of inliers:", inliers.sum())

    # Compare estimated sparse disparities to the dense ground-truth disparities.

    disp = inlier_keypoints_left[:, 1] - inlier_keypoints_right[:, 1]
    disp_coords = np.round(inlier_keypoints_left).astype(np.int64)
    disp_idxs = np.ravel_multi_index(disp_coords.T, groundtruth_disp.shape)
    disp_error = np.abs(groundtruth_disp.ravel()[disp_idxs] - disp)
    disp_error = disp_error[np.isfinite(disp_error)]

    # Visualize the results.

    fig, ax = plt.subplots(nrows=2, ncols=1)

    plt.gray()

    plot_matches(ax[0], img_left, img_right, keypoints_left, keypoints_right,
                 matches[inliers], only_matches=True)
    ax[0].axis("off")
    ax[0].set_title("Inlier correspondences")

    ax[1].hist(disp_error)
    ax[1].set_title("Histogram of disparity errors")

    plt.show()

    exit()


if __name__ == "__main__":

    np.random.seed(0)

    ratio_threshold = 0.5
    height = 256
    width = 320
    max_no_move_count = 30
    fast_forward_num = 5

    # feature_model_path = Path(
    #     "/home/xingtong/Projects/semantic-segmentation/depth-estimation/checkpoint_model_epoch_44.pt")
    # feature_model = feature_model_import(feature_model_path)
    # feature_model.eval()

    video_root = Path("/media/xingtong/Samsung_T5/Videos/Mono")
    video_path_list = list(video_root.glob("*.mp4"))

    warming_imgs = np.zeros((3, height * 4, width * 4, 3), dtype=np.uint8)
    for video_path in video_path_list:
        result_path = Path(str(video_path)[:-4])
        if not result_path.exists():
            result_path.mkdir()
        else:
            continue

        if video_path.exists():
            video_fp = cv2.VideoCapture(str(video_path))
        else:
            print("Video file {} does not exists".format(str(video_path)))
            raise IOError
        if not video_fp.isOpened():
            print("Error opening video file {}".format(str(video_path)))
            raise IOError
        total_frame_count = int(video_fp.get(cv2.CAP_PROP_FRAME_COUNT))
        start_time = 0

        fps = video_fp.get(cv2.CAP_PROP_FPS)
        print("frame rate is: {}".format(fps))
        video_sampling_rate = 1
        video_fp.set(cv2.CAP_PROP_POS_FRAMES, fps * start_time)
        frame = None
        prev_frame = None
        sequence_count = 0
        result_video_fp = None
        no_move_detection_count = 0
        state = "searching"
        tq = tqdm.tqdm(total=int(total_frame_count), dynamic_ncols=False, ncols=40)
        tq.set_description("Video {}".format(video_path.name))
        while video_fp.isOpened():
            frame_index = int(video_fp.get(cv2.CAP_PROP_POS_FRAMES))
            if frame is not None:
                prev_frame = frame

            ret, frame = video_fp.read()
            if not ret:
                break

            if prev_frame is not None:
                # Calculate dense optical flow using L-K method
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                opt_flow = np.zeros((prev_gray.shape[0], prev_gray.shape[1], 2), dtype=np.float32)
                cv2.calcOpticalFlowFarneback(prev=prev_gray, next=cur_gray, flow=opt_flow, pyr_scale=0.5, levels=3,
                                             winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
                flow_display = display_flow(opt_flow, 0.05)
                cv2.imshow("opt flow", flow_display)

                # downsampled_prev_frame = cv2.resize(prev_frame, dsize=(width, height))
                # downsampled_frame = cv2.resize(frame, dsize=(width, height))

                # colors_1 = img_to_tensor(
                #     downsampled_prev_frame.astype(np.float32) * 1.0 / 127.5 - 1.0).cuda().unsqueeze(0)
                # colors_2 = img_to_tensor(downsampled_frame.astype(np.float32) * 1.0 / 127.5 - 1.0).cuda().unsqueeze(0)
                # boundaries = torch.ones(colors_1.shape[0], 1, colors_1.shape[2], colors_1.shape[3]).float().cuda()
                # sparse_flow_maps_1, sparse_flow_masks_1, sparse_flow_maps_2, sparse_flow_masks_2, \
                valid_query_2D_locations, valid_train_2D_locations, display_matches_ai = \
                    pair_feature_matching(feature_model, colors_1, colors_2, boundaries, feature_sampling_stride=8,
                                          matching_scale=5.0,
                                          matching_threshold=0.9, cross_check_distance=30.0)
                # sparse_flow_1_display = draw_dl_flow(sparse_flow_maps_1, 0.05)

                # cv2.imshow("video + sparse flow", cv2.hconcat([cv2.resize(frame, dsize=(width * 2, height * 2)),
                #                                                cv2.resize(sparse_flow_1_display,
                #                                                           dsize=(width * 2, height * 2))]))



                locations_2D_difference = np.linalg.norm(valid_query_2D_locations - valid_train_2D_locations, axis=1)
                large_motion_indexes = np.where(locations_2D_difference >= 2)[0]

                if len(large_motion_indexes) < 20:
                    continue

                model, inliers = ransac(data=(valid_query_2D_locations[large_motion_indexes],
                                              valid_train_2D_locations[large_motion_indexes]),
                                        model_class=FundamentalMatrixTransform,
                                        min_samples=8, residual_threshold=4, max_trials=20)
                ransac_inliner_count = float(inliers.sum())
                large_motion_matching_count = float(len(large_motion_indexes))
                matching_count = float(valid_query_2D_locations.shape[0])
                ratio = ransac_inliner_count / matching_count
                tq.set_postfix(I_T_ratio='{:.5f}'.format(ransac_inliner_count / matching_count))

                if state == "searching":
                    tq.set_description("Status: searching")
                    video_sampling_rate = fast_forward_num
                    if ratio >= ratio_threshold:
                        state = "warming1"
                        video_sampling_rate = 1
                elif state == "warming1":
                    warming_imgs[0] = frame
                    video_sampling_rate = 1
                    tq.set_description("Status: warming1")
                    if ratio >= ratio_threshold:
                        state = "warming2"
                    else:
                        state = "searching"
                elif state == "warming2":
                    warming_imgs[1] = frame
                    video_sampling_rate = 1
                    tq.set_description("Status: warming2")
                    if ratio >= ratio_threshold:
                        state = "warming3"
                    else:
                        state = "searching"
                elif state == "warming3":
                    warming_imgs[2] = frame
                    video_sampling_rate = 1
                    tq.set_description("Status: warming3")
                    if ratio >= ratio_threshold:
                        result_video_fp = cv2.VideoWriter(str(result_path / "{}.mp4".format(sequence_count)),
                                                          cv2.VideoWriter_fourcc('M', 'P', '4', 'V'),
                                                          fps, (width * 4, height * 4))
                        for i in range(3):
                            result_video_fp.write(warming_imgs[i])

                        sequence_count += 1
                        state = "recording"
                    else:
                        state = "searching"
                elif state == "recording":
                    video_sampling_rate = 1
                    tq.set_description("Status: recording")
                    result_video_fp.write(frame)
                    if ratio < ratio_threshold:
                        no_move_detection_count = 0
                        state = "holding"
                elif state == "holding":
                    video_sampling_rate = 1
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

                video_fp.set(cv2.CAP_PROP_POS_FRAMES, frame_index + video_sampling_rate)
                tq.update(video_sampling_rate)
            cv2.waitKey(1)
tq.close()
