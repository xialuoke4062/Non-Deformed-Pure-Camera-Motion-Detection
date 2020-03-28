'''
Author: Xingtong Liu, Ayushi Sinha, Masaru Ishii, Gregory D. Hager, Austin Reiter, Russell H. Taylor, and Mathias Unberath

Copyright (C) 2019 Johns Hopkins University - All Rights Reserved
You may use, distribute and modify this code under the
terms of the GNU GENERAL PUBLIC LICENSE Version 3 license for non-commercial usage.

You should have received a copy of the GNU GENERAL PUBLIC LICENSE Version 3 license with
this file. If not, please write to: xliu89@jh.edu or rht@jhu.edu or unberath@jhu.edu
'''
import sys
import cv2
import yaml
import random
import torch
import datetime
import shutil
import math
import json
import os
import tqdm
import gc
import matplotlib
import math

matplotlib.use('Agg', warn=False, force=True)
from matplotlib import pyplot as plt

import numpy as np
import torchvision.utils as vutils
from sklearn.neighbors import KDTree
from pathlib import Path
from plyfile import PlyData, PlyElement
import torch.multiprocessing as mp
import torchgeometry as tgm
import multiprocessing

import models
import dataset


def mesh_inner_surface_extraction(camera_pose_list, depth_map_list, color_image_list):
    # TODO: KD Tree building using the point cloud from the first frame, and then only add points that are far away from the original points for the following frames
    pass


def feature_matching_single(color_1, color_2, feature_map_1, feature_map_2, kps_1D_1, des_1, des_2,
                            cross_check_distance, kps_1, kps_2, gpu_id=0):
    with torch.no_grad():
        color_1 = color_1.data.cpu().numpy()
        color_2 = color_2.data.cpu().numpy()
        # Color image 3 x H x W
        # Feature map C x H x W
        feature_length, height, width = feature_map_1.shape

        # Extend 1D locations to B x C x Sampling_size
        keypoint_number = len(kps_1D_1)
        source_feature_1d_locations = torch.from_numpy(kps_1D_1).long().cuda(gpu_id).view(
            1, 1,
            keypoint_number).expand(
            -1, feature_length, -1)

        # Sampled rough locator feature vectors
        sampled_feature_vectors = torch.gather(
            feature_map_1.view(1, feature_length, height * width), 2,
            source_feature_1d_locations.long())
        sampled_feature_vectors = sampled_feature_vectors.view(1, feature_length,
                                                               keypoint_number,
                                                               1,
                                                               1).permute(0, 2, 1, 3,
                                                                          4).view(1,
                                                                                  keypoint_number,
                                                                                  feature_length,
                                                                                  1, 1)

        filter_response_map = torch.nn.functional.conv2d(
            input=feature_map_2.view(1, feature_length, height, width),
            weight=sampled_feature_vectors.view(keypoint_number,
                                                feature_length,
                                                1, 1), padding=0)

        # Cleaning used variables to save space
        del sampled_feature_vectors
        del source_feature_1d_locations

        max_reponses, max_indexes = torch.max(filter_response_map.view(keypoint_number, -1), dim=1,
                                              keepdim=False)

        del filter_response_map
        # query is 1 and train is 2 here
        detected_target_1d_locations = max_indexes.view(-1)
        selected_max_responses = max_reponses.view(-1)
        # Do cross check
        feature_1d_locations_2 = detected_target_1d_locations.long().view(
            1, 1, -1).expand(-1, feature_length, -1)
        keypoint_number = keypoint_number

        # Sampled rough locator feature vectors
        sampled_feature_vectors_2 = torch.gather(
            feature_map_2.view(1, feature_length, height * width), 2,
            feature_1d_locations_2.long())
        sampled_feature_vectors_2 = sampled_feature_vectors_2.view(1, feature_length,
                                                                   keypoint_number,
                                                                   1,
                                                                   1).permute(0, 2, 1, 3,
                                                                              4).view(1,
                                                                                      keypoint_number,
                                                                                      feature_length,
                                                                                      1, 1)

        source_filter_response_map = torch.nn.functional.conv2d(
            input=feature_map_1.view(1, feature_length, height, width),
            weight=sampled_feature_vectors_2.view(keypoint_number,
                                                  feature_length,
                                                  1, 1), padding=0)

        del feature_1d_locations_2
        del sampled_feature_vectors_2

        max_reponses_2, max_indexes_2 = torch.max(source_filter_response_map.view(keypoint_number, -1),
                                                  dim=1,
                                                  keepdim=False)

        del source_filter_response_map

        keypoint_1d_locations_1 = torch.from_numpy(np.asarray(kps_1D_1)).float().cuda(gpu_id).view(
            keypoint_number, 1)
        keypoint_2d_locations_1 = torch.cat(
            [torch.fmod(keypoint_1d_locations_1, width),
             torch.floor(keypoint_1d_locations_1 / width)],
            dim=1).view(keypoint_number, 2).float()

        detected_source_keypoint_1d_locations = max_indexes_2.float().view(keypoint_number, 1)
        detected_source_keypoint_2d_locations = torch.cat(
            [torch.fmod(detected_source_keypoint_1d_locations, width),
             torch.floor(detected_source_keypoint_1d_locations / width)],
            dim=1).view(keypoint_number, 2).float()

        # We will accept the feature matches if the max indexes here is
        # not far away from the original key point location from descriptor
        cross_check_correspondence_distances = torch.norm(
            keypoint_2d_locations_1 - detected_source_keypoint_2d_locations, dim=1, p=2).view(
            keypoint_number)
        valid_correspondence_indexes = torch.nonzero(cross_check_correspondence_distances < cross_check_distance).view(
            -1)

        if valid_correspondence_indexes.shape[0] == 0:
            return None

        valid_detected_1d_locations_2 = torch.gather(detected_target_1d_locations.long().view(-1),
                                                     0, valid_correspondence_indexes.long())
        valid_max_responses = torch.gather(selected_max_responses.view(-1), 0,
                                           valid_correspondence_indexes.long())

        valid_detected_1d_locations_2 = valid_detected_1d_locations_2.data.cpu().numpy()
        valid_max_responses = valid_max_responses.data.cpu().numpy()
        valid_correspondence_indexes = valid_correspondence_indexes.data.cpu().numpy()

        detected_keypoints_2 = []
        for index in valid_detected_1d_locations_2:
            detected_keypoints_2.append(
                cv2.KeyPoint(x=float(np.floor(index % width)), y=float(np.floor(index / width)), _size=1.0))

        matches = []
        for i, (query_index, response) in enumerate(
                zip(valid_correspondence_indexes, valid_max_responses)):
            matches.append(cv2.DMatch(_queryIdx=query_index, _trainIdx=i, _distance=response))

        color_1 = np.moveaxis(color_1, source=[0, 1, 2], destination=[2, 0, 1])
        color_2 = np.moveaxis(color_2, source=[0, 1, 2], destination=[2, 0, 1])

        # Extract corner points
        color_1 = np.uint8(255 * (color_1 * 0.5 + 0.5))
        color_2 = np.uint8(255 * (color_2 * 0.5 + 0.5))

        display_matches_ai = cv2.drawMatches(color_1, kps_1, color_2, detected_keypoints_2, matches,
                                             flags=2,
                                             outImg=None)

        bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)
        feature_matches_craft = bf.knnMatch(des_1, des_2, k=1)

        good = []
        for m in feature_matches_craft:
            if len(m) != 0:
                good.append(m[0])
        display_matches_craft = cv2.drawMatches(color_1, kps_1, color_2, kps_2, good, flags=2,
                                                outImg=None)
        return display_matches_ai, display_matches_craft


def get_torch_training_data(pair_extrinsics, pair_projections, pair_indexes, point_cloud, mask_boundary,
                            view_indexes_per_point, clean_point_list, visible_view_indexes):
    height = mask_boundary.shape[0]
    width = mask_boundary.shape[1]
    pair_depth_mask_imgs = []
    pair_depth_imgs = []

    pair_flow_imgs = []
    flow_image_1 = np.zeros((height, width, 2), dtype=np.float32)
    flow_image_2 = np.zeros((height, width, 2), dtype=np.float32)

    pair_flow_mask_imgs = []
    flow_mask_image_1 = np.zeros((height, width, 1), dtype=np.float32)
    flow_mask_image_2 = np.zeros((height, width, 1), dtype=np.float32)

    # We only use inlier points
    array_3D_points = np.asarray(point_cloud).reshape((-1, 4))
    for i in range(2):
        projection_matrix = pair_projections[i]
        extrinsic_matrix = pair_extrinsics[i]

        if i == 0:
            points_2D_image_1 = np.einsum('ij,mj->mi', projection_matrix, array_3D_points)
            points_2D_image_1 = np.round(points_2D_image_1 / points_2D_image_1[:, 2].reshape((-1, 1)))
            points_3D_camera_1 = np.einsum('ij,mj->mi', extrinsic_matrix, array_3D_points)
            points_3D_camera_1 = points_3D_camera_1 / points_3D_camera_1[:, 3].reshape((-1, 1))
        else:
            points_2D_image_2 = np.einsum('ij,mj->mi', projection_matrix, array_3D_points)
            points_2D_image_2 = np.round(points_2D_image_2 / points_2D_image_2[:, 2].reshape((-1, 1)))
            points_3D_camera_2 = np.einsum('ij,mj->mi', extrinsic_matrix, array_3D_points)
            points_3D_camera_2 = points_3D_camera_2 / points_3D_camera_2[:, 3].reshape((-1, 1))

    mask_boundary = mask_boundary.reshape((-1, 1))
    flow_image_1 = flow_image_1.reshape((-1, 2))
    flow_image_2 = flow_image_2.reshape((-1, 2))
    flow_mask_image_1 = flow_mask_image_1.reshape((-1, 1))
    flow_mask_image_2 = flow_mask_image_2.reshape((-1, 1))

    points_2D_image_1 = points_2D_image_1.reshape((-1, 3))
    points_2D_image_2 = points_2D_image_2.reshape((-1, 3))
    points_3D_camera_1 = points_3D_camera_1.reshape((-1, 4))
    points_3D_camera_2 = points_3D_camera_2.reshape((-1, 4))

    point_visibility_1 = np.asarray(view_indexes_per_point[:, visible_view_indexes.index(pair_indexes[0])]).reshape(
        (-1))
    if len(clean_point_list) != 0:
        visible_point_indexes_1 = np.where((point_visibility_1 > 0.5) & (clean_point_list > 0.5))
    else:
        visible_point_indexes_1 = np.where((point_visibility_1 > 0.5))
    visible_point_indexes_1 = visible_point_indexes_1[0]
    point_visibility_2 = np.asarray(view_indexes_per_point[:, visible_view_indexes.index(pair_indexes[1])]).reshape(
        (-1))

    if len(clean_point_list) != 0:
        visible_point_indexes_2 = np.where((point_visibility_2 > 0.5) & (clean_point_list > 0.5))
    else:
        visible_point_indexes_2 = np.where((point_visibility_2 > 0.5))
    visible_point_indexes_2 = visible_point_indexes_2[0]
    visible_points_3D_camera_1 = points_3D_camera_1[visible_point_indexes_1, :].reshape((-1, 4))
    visible_points_2D_image_1 = points_2D_image_1[visible_point_indexes_1, :].reshape((-1, 3))
    visible_points_3D_camera_2 = points_3D_camera_2[visible_point_indexes_2, :].reshape((-1, 4))
    visible_points_2D_image_2 = points_2D_image_2[visible_point_indexes_2, :].reshape((-1, 3))

    in_image_indexes_1 = np.where(
        (visible_points_2D_image_1[:, 0] <= width - 1) & (visible_points_2D_image_1[:, 0] >= 0) &
        (visible_points_2D_image_1[:, 1] <= height - 1) & (visible_points_2D_image_1[:, 1] >= 0)
        & (visible_points_3D_camera_1[:, 2] > 0))
    in_image_indexes_1 = in_image_indexes_1[0]
    in_image_point_1D_locations_1 = (np.round(visible_points_2D_image_1[in_image_indexes_1, 0]) +
                                     np.round(visible_points_2D_image_1[in_image_indexes_1, 1]) * width).astype(
        np.int32).reshape((-1))
    temp_mask_1 = mask_boundary[in_image_point_1D_locations_1, :]
    in_mask_indexes_1 = np.where(temp_mask_1[:, 0] == 255)
    in_mask_indexes_1 = in_mask_indexes_1[0]
    in_mask_point_1D_locations_1 = in_image_point_1D_locations_1[in_mask_indexes_1]
    flow_mask_image_1[in_mask_point_1D_locations_1, 0] = 1.0

    in_image_indexes_2 = np.where(
        (visible_points_2D_image_2[:, 0] <= width - 1) & (visible_points_2D_image_2[:, 0] >= 0) &
        (visible_points_2D_image_2[:, 1] <= height - 1) & (visible_points_2D_image_2[:, 1] >= 0)
        & (visible_points_3D_camera_2[:, 2] > 0))
    in_image_indexes_2 = in_image_indexes_2[0]
    in_image_point_1D_locations_2 = (np.round(visible_points_2D_image_2[in_image_indexes_2, 0]) +
                                     np.round(visible_points_2D_image_2[in_image_indexes_2, 1]) * width).astype(
        np.int32).reshape((-1))
    temp_mask_2 = mask_boundary[in_image_point_1D_locations_2, :]
    in_mask_indexes_2 = np.where(temp_mask_2[:, 0] == 255)
    in_mask_indexes_2 = in_mask_indexes_2[0]
    in_mask_point_1D_locations_2 = in_image_point_1D_locations_2[in_mask_indexes_2]
    flow_mask_image_2[in_mask_point_1D_locations_2, 0] = 1.0

    flow_image_1[in_mask_point_1D_locations_1, :] = points_2D_image_2[
                                                    visible_point_indexes_1[in_image_indexes_1[in_mask_indexes_1]],
                                                    :2] - \
                                                    points_2D_image_1[
                                                    visible_point_indexes_1[in_image_indexes_1[in_mask_indexes_1]], :2]
    flow_image_2[in_mask_point_1D_locations_2, :] = points_2D_image_1[
                                                    visible_point_indexes_2[in_image_indexes_2[in_mask_indexes_2]],
                                                    :2] - \
                                                    points_2D_image_2[
                                                    visible_point_indexes_2[in_image_indexes_2[in_mask_indexes_2]], :2]

    flow_image_1[:, 0] /= width
    flow_image_1[:, 1] /= height
    flow_image_2[:, 0] /= width
    flow_image_2[:, 1] /= height

    outlier_indexes_1 = np.where((np.abs(flow_image_1[:, 0]) > 5.0) | (np.abs(flow_image_1[:, 1]) > 5.0))[0]
    outlier_indexes_2 = np.where((np.abs(flow_image_2[:, 0]) > 5.0) | (np.abs(flow_image_2[:, 1]) > 5.0))[0]
    flow_mask_image_1[outlier_indexes_1, 0] = 0.0
    flow_mask_image_2[outlier_indexes_2, 0] = 0.0
    flow_image_1[outlier_indexes_1, 0] = 0.0
    flow_image_2[outlier_indexes_2, 0] = 0.0
    flow_image_1[outlier_indexes_1, 1] = 0.0
    flow_image_2[outlier_indexes_2, 1] = 0.0

    depth_img_1 = np.zeros((height, width, 1), dtype=np.float32)
    depth_img_2 = np.zeros((height, width, 1), dtype=np.float32)
    depth_mask_img_1 = np.zeros((height, width, 1), dtype=np.float32)
    depth_mask_img_2 = np.zeros((height, width, 1), dtype=np.float32)
    depth_img_1 = depth_img_1.reshape((-1, 1))
    depth_img_2 = depth_img_2.reshape((-1, 1))
    depth_mask_img_1 = depth_mask_img_1.reshape((-1, 1))
    depth_mask_img_2 = depth_mask_img_2.reshape((-1, 1))

    depth_img_1[in_mask_point_1D_locations_1, 0] = points_3D_camera_1[
        visible_point_indexes_1[in_image_indexes_1[in_mask_indexes_1]], 2]
    depth_img_2[in_mask_point_1D_locations_2, 0] = points_3D_camera_2[
        visible_point_indexes_2[in_image_indexes_2[in_mask_indexes_2]], 2]
    depth_mask_img_1[in_mask_point_1D_locations_1, 0] = 1.0
    depth_mask_img_2[in_mask_point_1D_locations_2, 0] = 1.0

    pair_flow_imgs.append(flow_image_1)
    pair_flow_imgs.append(flow_image_2)
    pair_flow_imgs = np.array(pair_flow_imgs, dtype="float32")
    pair_flow_imgs = np.reshape(pair_flow_imgs, (-1, height, width, 2))

    pair_flow_mask_imgs.append(flow_mask_image_1)
    pair_flow_mask_imgs.append(flow_mask_image_2)
    pair_flow_mask_imgs = np.array(pair_flow_mask_imgs, dtype="float32")
    pair_flow_mask_imgs = np.reshape(pair_flow_mask_imgs, (-1, height, width, 1))

    pair_depth_mask_imgs.append(depth_mask_img_1)
    pair_depth_mask_imgs.append(depth_mask_img_2)
    pair_depth_mask_imgs = np.array(pair_depth_mask_imgs, dtype="float32")
    pair_depth_mask_imgs = np.reshape(pair_depth_mask_imgs, (-1, height, width, 1))

    pair_depth_imgs.append(depth_img_1)
    pair_depth_imgs.append(depth_img_2)
    pair_depth_imgs = np.array(pair_depth_imgs, dtype="float32")
    pair_depth_imgs = np.reshape(pair_depth_imgs, (-1, height, width, 1))

    return pair_depth_mask_imgs, pair_depth_imgs, pair_flow_mask_imgs, pair_flow_imgs


def draw_flow(flows, max_v=None):
    batch_size, channel, height, width = flows.shape
    flows_x_display = vutils.make_grid(flows[:, 0, :, :].view(batch_size, 1, height, width), normalize=False,
                                       scale_each=False)
    flows_y_display = vutils.make_grid(flows[:, 1, :, :].view(batch_size, 1, height, width), normalize=False,
                                       scale_each=False)
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
    if max_v is None:
        hsv[..., 2] = np.uint8(np.minimum(v / np.max(v), 1.0) * 255)
    else:
        hsv[..., 2] = np.uint8(np.minimum(v / max_v, 1.0) * 255)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), np.max(v)


def display_color_depth_dense_flow(idx, step, writer, colors_1, pred_depths_1, flows_from_depth_1,
                                   phase="Training", is_return_image=False, color_reverse=True,
                                   ):
    # print(torch.min(colors_1), torch.max(colors_1))
    colors_display = vutils.make_grid(colors_1 * 0.5 + 0.5, normalize=False)
    colors_display = np.moveaxis(colors_display.data.cpu().numpy(),
                                 source=[0, 1, 2], destination=[2, 0, 1])
    colors_display = cv2.cvtColor(colors_display, cv2.COLOR_HSV2RGB_FULL)

    pred_depths_display = vutils.make_grid(pred_depths_1, normalize=True, scale_each=True)
    pred_depths_display = cv2.applyColorMap(np.uint8(255 * np.moveaxis(pred_depths_display.data.cpu().numpy(),
                                                                       source=[0, 1, 2],
                                                                       destination=[2, 0, 1])), cv2.COLORMAP_JET)
    dense_flows_display, _ = draw_flow(flows_from_depth_1)
    if color_reverse:
        pred_depths_display = cv2.cvtColor(pred_depths_display, cv2.COLOR_BGR2RGB)
        dense_flows_display = cv2.cvtColor(dense_flows_display, cv2.COLOR_BGR2RGB)

    if is_return_image:
        return colors_display, pred_depths_display.astype(np.float32) / 255.0, \
               dense_flows_display.astype(np.float32) / 255.0
    else:
        writer.add_image(phase + '/Images/Color_' + str(idx), colors_display, step, dataformats="HWC")
        writer.add_image(phase + '/Images/Pred_Depth_' + str(idx), pred_depths_display, step, dataformats="HWC")
        writer.add_image(phase + '/Images/Dense_Flow_' + str(idx), dense_flows_display, step, dataformats="HWC")
        return


def uniform_sample_generation(height, width, sampling_stride, offset, sampling_size, is_random=True):
    location_2D_list = []
    if is_random:
        random_range = int(sampling_stride // 2)
        for h in range(offset, height - offset, sampling_stride):
            for w in range(offset, width - offset, sampling_stride):
                random_h = min(height - offset,
                               max(offset, h + np.random.randint(low=-random_range, high=random_range)))
                random_w = min(width - offset, max(offset, w + np.random.randint(low=-random_range, high=random_range)))
                location_2D_list.append(random_w)
                location_2D_list.append(random_h)
    else:
        for h in range(offset, height - offset, sampling_stride):
            for w in range(offset, width - offset, sampling_stride):
                sampled_h = min(height - offset, max(offset, h))
                sampled_w = min(width - offset, max(offset, w))
                location_2D_list.append(sampled_w)
                location_2D_list.append(sampled_h)

    return location_2D_list


def get_test_color_img(img_file_name, start_h, end_h, start_w, end_w, downsampling_factor, is_hsv, rgb_mode):
    img = cv2.imread(img_file_name)
    downsampled_img = cv2.resize(img, (0, 0), fx=1. / downsampling_factor, fy=1. / downsampling_factor)
    downsampled_img = downsampled_img[start_h:end_h, start_w:end_w, :]
    if is_hsv:
        downsampled_img = cv2.cvtColor(downsampled_img, cv2.COLOR_BGR2HSV_FULL)
    else:
        if rgb_mode == "rgb":
            downsampled_img = cv2.cvtColor(downsampled_img, cv2.COLOR_BGR2RGB)
    downsampled_img = np.array(downsampled_img, dtype="float32")
    return downsampled_img


def rgb_to_hsv_torch_tensor(colors):
    colors = colors.data.cpu().numpy()
    colors_list = []
    for color in colors:
        color = np.moveaxis(0.5 * color + 0.5, source=[0, 1, 2], destination=[2, 0, 1])
        color = np.asarray(255.0 * color, dtype=np.uint8)
        color = cv2.cvtColor(color, cv2.COLOR_RGB2HSV_FULL)
        color = np.moveaxis(color, source=[0, 1, 2], destination=[1, 2, 0])
        colors_list.append(torch.from_numpy(color / 255.0 * 2.0 - 1.0).float().cuda().unsqueeze(dim=0))
    colors = torch.cat(colors_list, dim=0)
    return colors


def visualize_descriptor_maps(descriptor_maps, title, batch_idx, slice_idx, min_val=None, max_val=None):
    batch_size, channel, height, width = descriptor_maps.shape
    descriptor_map = descriptor_maps[batch_idx, slice_idx:slice_idx + 3].view(3, height, width)

    if min_val is None:
        min_val, _ = torch.min(descriptor_map.view(3, -1), dim=1)
    if max_val is None:
        max_val, _ = torch.max(descriptor_map.view(3, -1), dim=1)

    descriptor_map = (descriptor_map - min_val.view(3, 1, 1)) / (max_val.view(3, 1, 1) - min_val.view(3, 1, 1))
    descriptor_map = np.asarray(255.0 * descriptor_map.data.cpu().numpy(), dtype=np.uint8)
    descriptor_map = np.moveaxis(descriptor_map, source=[0, 1, 2], destination=[2, 0, 1])

    cv2.imshow(title, descriptor_map)
    return min_val, max_val


def get_pair_color_imgs(prefix_seq, pair_indexes, start_h, end_h, start_w, end_w, downsampling_factor, is_hsv,
                        rgb_mode):
    imgs = []
    for i in pair_indexes:
        img = cv2.imread(str(Path(prefix_seq) / "{:08d}.jpg".format(i)))
        downsampled_img = cv2.resize(img, (0, 0), fx=1. / downsampling_factor, fy=1. / downsampling_factor)
        downsampled_img = downsampled_img[start_h:end_h, start_w:end_w, :]
        if is_hsv:
            downsampled_img = cv2.cvtColor(downsampled_img, cv2.COLOR_BGR2HSV_FULL)
        else:
            if rgb_mode == "rgb":
                downsampled_img = cv2.cvtColor(downsampled_img, cv2.COLOR_BGR2RGB)
        imgs.append(downsampled_img)
    height, width, channel = imgs[0].shape
    imgs = np.asarray(imgs, dtype=np.float32)
    imgs = imgs.reshape((-1, height, width, channel))
    return imgs


def read_color_img(image_path, start_h, end_h, start_w, end_w, downsampling_factor, is_hsv,
                   rgb_mode):
    img = cv2.imread(str(image_path))
    downsampled_img = cv2.resize(img, (0, 0), fx=1. / downsampling_factor, fy=1. / downsampling_factor)
    downsampled_img = downsampled_img[start_h:end_h, start_w:end_w, :]
    if is_hsv:
        downsampled_img = cv2.cvtColor(downsampled_img, cv2.COLOR_BGR2HSV_FULL)
    else:
        if rgb_mode == "rgb":
            downsampled_img = cv2.cvtColor(downsampled_img, cv2.COLOR_BGR2RGB)
    downsampled_img = downsampled_img.astype(np.float32)
    return downsampled_img


def get_torch_training_data(pair_extrinsics, pair_projections, pair_indexes, point_cloud, mask_boundary,
                            view_indexes_per_point, clean_point_list, visible_view_indexes):
    height = mask_boundary.shape[0]
    width = mask_boundary.shape[1]
    pair_depth_mask_imgs = []
    pair_depth_imgs = []

    pair_flow_imgs = []
    flow_image_1 = np.zeros((height, width, 2), dtype=np.float32)
    flow_image_2 = np.zeros((height, width, 2), dtype=np.float32)

    pair_flow_mask_imgs = []
    flow_mask_image_1 = np.zeros((height, width, 1), dtype=np.float32)
    flow_mask_image_2 = np.zeros((height, width, 1), dtype=np.float32)

    # We only use inlier points
    array_3D_points = np.asarray(point_cloud).reshape((-1, 4))
    for i in range(2):
        projection_matrix = pair_projections[i]
        extrinsic_matrix = pair_extrinsics[i]

        if i == 0:
            points_2D_image_1 = np.einsum('ij,mj->mi', projection_matrix, array_3D_points)
            points_2D_image_1 = np.round(points_2D_image_1 / points_2D_image_1[:, 2].reshape((-1, 1)))
            points_3D_camera_1 = np.einsum('ij,mj->mi', extrinsic_matrix, array_3D_points)
            points_3D_camera_1 = points_3D_camera_1 / points_3D_camera_1[:, 3].reshape((-1, 1))
        else:
            points_2D_image_2 = np.einsum('ij,mj->mi', projection_matrix, array_3D_points)
            points_2D_image_2 = np.round(points_2D_image_2 / points_2D_image_2[:, 2].reshape((-1, 1)))
            points_3D_camera_2 = np.einsum('ij,mj->mi', extrinsic_matrix, array_3D_points)
            points_3D_camera_2 = points_3D_camera_2 / points_3D_camera_2[:, 3].reshape((-1, 1))

    mask_boundary = mask_boundary.reshape((-1, 1))
    flow_image_1 = flow_image_1.reshape((-1, 2))
    flow_image_2 = flow_image_2.reshape((-1, 2))
    flow_mask_image_1 = flow_mask_image_1.reshape((-1, 1))
    flow_mask_image_2 = flow_mask_image_2.reshape((-1, 1))

    points_2D_image_1 = points_2D_image_1.reshape((-1, 3))
    points_2D_image_2 = points_2D_image_2.reshape((-1, 3))
    points_3D_camera_1 = points_3D_camera_1.reshape((-1, 4))
    points_3D_camera_2 = points_3D_camera_2.reshape((-1, 4))

    point_visibility_1 = np.asarray(view_indexes_per_point[:, visible_view_indexes.index(pair_indexes[0])]).reshape(
        (-1))
    if len(clean_point_list) != 0:
        visible_point_indexes_1 = np.where((point_visibility_1 > 0.5) & (clean_point_list > 0.5))
    else:
        visible_point_indexes_1 = np.where((point_visibility_1 > 0.5))
    visible_point_indexes_1 = visible_point_indexes_1[0]
    point_visibility_2 = np.asarray(view_indexes_per_point[:, visible_view_indexes.index(pair_indexes[1])]).reshape(
        (-1))

    if len(clean_point_list) != 0:
        visible_point_indexes_2 = np.where((point_visibility_2 > 0.5) & (clean_point_list > 0.5))
    else:
        visible_point_indexes_2 = np.where((point_visibility_2 > 0.5))
    visible_point_indexes_2 = visible_point_indexes_2[0]
    visible_points_3D_camera_1 = points_3D_camera_1[visible_point_indexes_1, :].reshape((-1, 4))
    visible_points_2D_image_1 = points_2D_image_1[visible_point_indexes_1, :].reshape((-1, 3))
    visible_points_3D_camera_2 = points_3D_camera_2[visible_point_indexes_2, :].reshape((-1, 4))
    visible_points_2D_image_2 = points_2D_image_2[visible_point_indexes_2, :].reshape((-1, 3))

    in_image_indexes_1 = np.where(
        (visible_points_2D_image_1[:, 0] <= width - 1) & (visible_points_2D_image_1[:, 0] >= 0) &
        (visible_points_2D_image_1[:, 1] <= height - 1) & (visible_points_2D_image_1[:, 1] >= 0)
        & (visible_points_3D_camera_1[:, 2] > 0))
    in_image_indexes_1 = in_image_indexes_1[0]
    in_image_point_1D_locations_1 = (np.round(visible_points_2D_image_1[in_image_indexes_1, 0]) +
                                     np.round(visible_points_2D_image_1[in_image_indexes_1, 1]) * width).astype(
        np.int32).reshape((-1))
    temp_mask_1 = mask_boundary[in_image_point_1D_locations_1, :]
    in_mask_indexes_1 = np.where(temp_mask_1[:, 0] == 255)
    in_mask_indexes_1 = in_mask_indexes_1[0]
    in_mask_point_1D_locations_1 = in_image_point_1D_locations_1[in_mask_indexes_1]
    flow_mask_image_1[in_mask_point_1D_locations_1, 0] = 1.0

    in_image_indexes_2 = np.where(
        (visible_points_2D_image_2[:, 0] <= width - 1) & (visible_points_2D_image_2[:, 0] >= 0) &
        (visible_points_2D_image_2[:, 1] <= height - 1) & (visible_points_2D_image_2[:, 1] >= 0)
        & (visible_points_3D_camera_2[:, 2] > 0))
    in_image_indexes_2 = in_image_indexes_2[0]
    in_image_point_1D_locations_2 = (np.round(visible_points_2D_image_2[in_image_indexes_2, 0]) +
                                     np.round(visible_points_2D_image_2[in_image_indexes_2, 1]) * width).astype(
        np.int32).reshape((-1))
    temp_mask_2 = mask_boundary[in_image_point_1D_locations_2, :]
    in_mask_indexes_2 = np.where(temp_mask_2[:, 0] == 255)
    in_mask_indexes_2 = in_mask_indexes_2[0]
    in_mask_point_1D_locations_2 = in_image_point_1D_locations_2[in_mask_indexes_2]
    flow_mask_image_2[in_mask_point_1D_locations_2, 0] = 1.0

    flow_image_1[in_mask_point_1D_locations_1, :] = points_2D_image_2[
                                                    visible_point_indexes_1[in_image_indexes_1[in_mask_indexes_1]],
                                                    :2] - \
                                                    points_2D_image_1[
                                                    visible_point_indexes_1[in_image_indexes_1[in_mask_indexes_1]], :2]
    flow_image_2[in_mask_point_1D_locations_2, :] = points_2D_image_1[
                                                    visible_point_indexes_2[in_image_indexes_2[in_mask_indexes_2]],
                                                    :2] - \
                                                    points_2D_image_2[
                                                    visible_point_indexes_2[in_image_indexes_2[in_mask_indexes_2]], :2]

    flow_image_1[:, 0] /= width
    flow_image_1[:, 1] /= height
    flow_image_2[:, 0] /= width
    flow_image_2[:, 1] /= height

    outlier_indexes_1 = np.where((np.abs(flow_image_1[:, 0]) > 5.0) | (np.abs(flow_image_1[:, 1]) > 5.0))[0]
    outlier_indexes_2 = np.where((np.abs(flow_image_2[:, 0]) > 5.0) | (np.abs(flow_image_2[:, 1]) > 5.0))[0]
    flow_mask_image_1[outlier_indexes_1, 0] = 0.0
    flow_mask_image_2[outlier_indexes_2, 0] = 0.0
    flow_image_1[outlier_indexes_1, 0] = 0.0
    flow_image_2[outlier_indexes_2, 0] = 0.0
    flow_image_1[outlier_indexes_1, 1] = 0.0
    flow_image_2[outlier_indexes_2, 1] = 0.0

    depth_img_1 = np.zeros((height, width, 1), dtype=np.float32)
    depth_img_2 = np.zeros((height, width, 1), dtype=np.float32)
    depth_mask_img_1 = np.zeros((height, width, 1), dtype=np.float32)
    depth_mask_img_2 = np.zeros((height, width, 1), dtype=np.float32)
    depth_img_1 = depth_img_1.reshape((-1, 1))
    depth_img_2 = depth_img_2.reshape((-1, 1))
    depth_mask_img_1 = depth_mask_img_1.reshape((-1, 1))
    depth_mask_img_2 = depth_mask_img_2.reshape((-1, 1))

    depth_img_1[in_mask_point_1D_locations_1, 0] = points_3D_camera_1[
        visible_point_indexes_1[in_image_indexes_1[in_mask_indexes_1]], 2]
    depth_img_2[in_mask_point_1D_locations_2, 0] = points_3D_camera_2[
        visible_point_indexes_2[in_image_indexes_2[in_mask_indexes_2]], 2]
    depth_mask_img_1[in_mask_point_1D_locations_1, 0] = 1.0
    depth_mask_img_2[in_mask_point_1D_locations_2, 0] = 1.0

    pair_flow_imgs.append(flow_image_1)
    pair_flow_imgs.append(flow_image_2)
    pair_flow_imgs = np.array(pair_flow_imgs, dtype="float32")
    pair_flow_imgs = np.reshape(pair_flow_imgs, (-1, height, width, 2))

    pair_flow_mask_imgs.append(flow_mask_image_1)
    pair_flow_mask_imgs.append(flow_mask_image_2)
    pair_flow_mask_imgs = np.array(pair_flow_mask_imgs, dtype="float32")
    pair_flow_mask_imgs = np.reshape(pair_flow_mask_imgs, (-1, height, width, 1))

    pair_depth_mask_imgs.append(depth_mask_img_1)
    pair_depth_mask_imgs.append(depth_mask_img_2)
    pair_depth_mask_imgs = np.array(pair_depth_mask_imgs, dtype="float32")
    pair_depth_mask_imgs = np.reshape(pair_depth_mask_imgs, (-1, height, width, 1))

    pair_depth_imgs.append(depth_img_1)
    pair_depth_imgs.append(depth_img_2)
    pair_depth_imgs = np.array(pair_depth_imgs, dtype="float32")
    pair_depth_imgs = np.reshape(pair_depth_imgs, (-1, height, width, 1))

    return pair_depth_mask_imgs, pair_depth_imgs, pair_flow_mask_imgs, pair_flow_imgs


def get_sparse_depth_and_indexes(pair_extrinsics, pair_projections, pair_indexes, point_cloud, mask_boundary,
                                 view_indexes_per_point, clean_point_list, visible_view_indexes):
    height = mask_boundary.shape[0]
    width = mask_boundary.shape[1]
    pair_depth_mask_imgs = []
    pair_depth_imgs = []
    pair_point_index_imgs = []
    pair_index_imgs = []
    # pair_flow_imgs = []
    # flow_image_1 = np.zeros((height, width, 2), dtype=np.float32)
    # flow_image_2 = np.zeros((height, width, 2), dtype=np.float32)
    #
    # pair_flow_mask_imgs = []
    # flow_mask_image_1 = np.zeros((height, width, 1), dtype=np.float32)
    # flow_mask_image_2 = np.zeros((height, width, 1), dtype=np.float32)

    # We only use inlier points
    array_3D_points = np.asarray(point_cloud).reshape((-1, 4))
    for i in range(2):
        projection_matrix = pair_projections[i]
        extrinsic_matrix = pair_extrinsics[i]

        if i == 0:
            points_2D_image_1 = np.einsum('ij,mj->mi', projection_matrix, array_3D_points)
            points_2D_image_1 = np.round(points_2D_image_1 / points_2D_image_1[:, 2].reshape((-1, 1)))
            points_3D_camera_1 = np.einsum('ij,mj->mi', extrinsic_matrix, array_3D_points)
            points_3D_camera_1 = points_3D_camera_1 / points_3D_camera_1[:, 3].reshape((-1, 1))
        else:
            points_2D_image_2 = np.einsum('ij,mj->mi', projection_matrix, array_3D_points)
            points_2D_image_2 = np.round(points_2D_image_2 / points_2D_image_2[:, 2].reshape((-1, 1)))
            points_3D_camera_2 = np.einsum('ij,mj->mi', extrinsic_matrix, array_3D_points)
            points_3D_camera_2 = points_3D_camera_2 / points_3D_camera_2[:, 3].reshape((-1, 1))

    mask_boundary = mask_boundary.reshape((-1, 1))
    # flow_image_1 = flow_image_1.reshape((-1, 2))
    # flow_image_2 = flow_image_2.reshape((-1, 2))
    # flow_mask_image_1 = flow_mask_image_1.reshape((-1, 1))
    # flow_mask_image_2 = flow_mask_image_2.reshape((-1, 1))

    points_2D_image_1 = points_2D_image_1.reshape((-1, 3))
    points_2D_image_2 = points_2D_image_2.reshape((-1, 3))
    points_3D_camera_1 = points_3D_camera_1.reshape((-1, 4))
    points_3D_camera_2 = points_3D_camera_2.reshape((-1, 4))

    point_visibility_1 = np.asarray(view_indexes_per_point[:, visible_view_indexes.index(pair_indexes[0])]).reshape(
        (-1))
    if len(clean_point_list) != 0:
        visible_point_indexes_1 = np.where((point_visibility_1 > 0.5) & (clean_point_list > 0.5))
    else:
        visible_point_indexes_1 = np.where((point_visibility_1 > 0.5))
    visible_point_indexes_1 = visible_point_indexes_1[0]
    point_visibility_2 = np.asarray(view_indexes_per_point[:, visible_view_indexes.index(pair_indexes[1])]).reshape(
        (-1))

    if len(clean_point_list) != 0:
        visible_point_indexes_2 = np.where((point_visibility_2 > 0.5) & (clean_point_list > 0.5))
    else:
        visible_point_indexes_2 = np.where((point_visibility_2 > 0.5))
    visible_point_indexes_2 = visible_point_indexes_2[0]
    visible_points_3D_camera_1 = points_3D_camera_1[visible_point_indexes_1, :].reshape((-1, 4))
    visible_points_2D_image_1 = points_2D_image_1[visible_point_indexes_1, :].reshape((-1, 3))
    visible_points_3D_camera_2 = points_3D_camera_2[visible_point_indexes_2, :].reshape((-1, 4))
    visible_points_2D_image_2 = points_2D_image_2[visible_point_indexes_2, :].reshape((-1, 3))

    in_image_indexes_1 = np.where(
        (visible_points_2D_image_1[:, 0] <= width - 1) & (visible_points_2D_image_1[:, 0] >= 0) &
        (visible_points_2D_image_1[:, 1] <= height - 1) & (visible_points_2D_image_1[:, 1] >= 0)
        & (visible_points_3D_camera_1[:, 2] > 0))
    in_image_indexes_1 = in_image_indexes_1[0]
    in_image_point_1D_locations_1 = (np.round(visible_points_2D_image_1[in_image_indexes_1, 0]) +
                                     np.round(visible_points_2D_image_1[in_image_indexes_1, 1]) * width).astype(
        np.int32).reshape((-1))
    temp_mask_1 = mask_boundary[in_image_point_1D_locations_1, :]
    in_mask_indexes_1 = np.where(temp_mask_1[:, 0] == 255)
    in_mask_indexes_1 = in_mask_indexes_1[0]
    in_mask_point_1D_locations_1 = in_image_point_1D_locations_1[in_mask_indexes_1]
    # flow_mask_image_1[in_mask_point_1D_locations_1, 0] = 1.0

    in_image_indexes_2 = np.where(
        (visible_points_2D_image_2[:, 0] <= width - 1) & (visible_points_2D_image_2[:, 0] >= 0) &
        (visible_points_2D_image_2[:, 1] <= height - 1) & (visible_points_2D_image_2[:, 1] >= 0)
        & (visible_points_3D_camera_2[:, 2] > 0))
    in_image_indexes_2 = in_image_indexes_2[0]
    in_image_point_1D_locations_2 = (np.round(visible_points_2D_image_2[in_image_indexes_2, 0]) +
                                     np.round(visible_points_2D_image_2[in_image_indexes_2, 1]) * width).astype(
        np.int32).reshape((-1))
    temp_mask_2 = mask_boundary[in_image_point_1D_locations_2, :]
    in_mask_indexes_2 = np.where(temp_mask_2[:, 0] == 255)
    in_mask_indexes_2 = in_mask_indexes_2[0]
    in_mask_point_1D_locations_2 = in_image_point_1D_locations_2[in_mask_indexes_2]
    # flow_mask_image_2[in_mask_point_1D_locations_2, 0] = 1.0

    # flow_image_1[in_mask_point_1D_locations_1, :] = points_2D_image_2[
    #                                                 visible_point_indexes_1[in_image_indexes_1[in_mask_indexes_1]],
    #                                                 :2] - \
    #                                                 points_2D_image_1[
    #                                                 visible_point_indexes_1[in_image_indexes_1[in_mask_indexes_1]], :2]
    # flow_image_2[in_mask_point_1D_locations_2, :] = points_2D_image_1[
    #                                                 visible_point_indexes_2[in_image_indexes_2[in_mask_indexes_2]],
    #                                                 :2] - \
    #                                                 points_2D_image_2[
    #                                                 visible_point_indexes_2[in_image_indexes_2[in_mask_indexes_2]], :2]
    #
    # flow_image_1[:, 0] /= width
    # flow_image_1[:, 1] /= height
    # flow_image_2[:, 0] /= width
    # flow_image_2[:, 1] /= height
    #
    # outlier_indexes_1 = np.where((np.abs(flow_image_1[:, 0]) > 5.0) | (np.abs(flow_image_1[:, 1]) > 5.0))[0]
    # outlier_indexes_2 = np.where((np.abs(flow_image_2[:, 0]) > 5.0) | (np.abs(flow_image_2[:, 1]) > 5.0))[0]
    # flow_mask_image_1[outlier_indexes_1, 0] = 0.0
    # flow_mask_image_2[outlier_indexes_2, 0] = 0.0
    # flow_image_1[outlier_indexes_1, 0] = 0.0
    # flow_image_2[outlier_indexes_2, 0] = 0.0
    # flow_image_1[outlier_indexes_1, 1] = 0.0
    # flow_image_2[outlier_indexes_2, 1] = 0.0

    depth_img_1 = np.zeros((height, width, 1), dtype=np.float32)
    depth_img_2 = np.zeros((height, width, 1), dtype=np.float32)
    depth_mask_img_1 = np.zeros((height, width, 1), dtype=np.float32)
    depth_mask_img_2 = np.zeros((height, width, 1), dtype=np.float32)
    point_index_img_1 = np.zeros((height, width, 1), dtype=np.float32)
    point_index_img_2 = np.zeros((height, width, 1), dtype=np.float32)

    depth_img_1 = depth_img_1.reshape((-1, 1))
    depth_img_2 = depth_img_2.reshape((-1, 1))
    depth_mask_img_1 = depth_mask_img_1.reshape((-1, 1))
    depth_mask_img_2 = depth_mask_img_2.reshape((-1, 1))

    depth_img_1[in_mask_point_1D_locations_1, 0] = points_3D_camera_1[
        visible_point_indexes_1[in_image_indexes_1[in_mask_indexes_1]], 2]
    depth_img_2[in_mask_point_1D_locations_2, 0] = points_3D_camera_2[
        visible_point_indexes_2[in_image_indexes_2[in_mask_indexes_2]], 2]
    depth_mask_img_1[in_mask_point_1D_locations_1, 0] = 1.0
    depth_mask_img_2[in_mask_point_1D_locations_2, 0] = 1.0

    # in_mask_point_2D_locations_1 = np.concatenate([in_mask_point_1D_locations_1.reshape((-1, 1)) % width,
    #                                                in_mask_point_1D_locations_1.reshape((-1, 1)) // width], axis=1)
    # in_mask_point_2D_locations_2 = np.concatenate([in_mask_point_1D_locations_2.reshape((-1, 1)) % width,
    #                                                in_mask_point_1D_locations_2.reshape((-1, 1)) // width], axis=1)
    point_cloud_indexes_1 = visible_point_indexes_1[in_image_indexes_1[in_mask_indexes_1]]
    point_cloud_indexes_2 = visible_point_indexes_2[in_image_indexes_2[in_mask_indexes_2]]

    point_index_img_1 = point_index_img_1.reshape((-1, 1))
    point_index_img_2 = point_index_img_2.reshape((-1, 1))
    point_index_img_1[in_mask_point_1D_locations_1, 0] = point_cloud_indexes_1.reshape((-1))
    point_index_img_2[in_mask_point_1D_locations_2, 0] = point_cloud_indexes_2.reshape((-1))

    pair_depth_mask_imgs.append(depth_mask_img_1)
    pair_depth_mask_imgs.append(depth_mask_img_2)
    pair_depth_mask_imgs = np.array(pair_depth_mask_imgs, dtype="float32")
    pair_depth_mask_imgs = np.reshape(pair_depth_mask_imgs, (-1, height, width, 1))

    pair_depth_imgs.append(depth_img_1)
    pair_depth_imgs.append(depth_img_2)
    pair_depth_imgs = np.array(pair_depth_imgs, dtype="float32")
    pair_depth_imgs = np.reshape(pair_depth_imgs, (-1, height, width, 1))

    pair_point_index_imgs.append(point_index_img_1)
    pair_point_index_imgs.append(point_index_img_2)
    pair_point_index_imgs = np.array(pair_point_index_imgs, dtype="float32")
    pair_point_index_imgs = np.reshape(pair_point_index_imgs, (-1, height, width, 1))

    return pair_depth_mask_imgs, pair_depth_imgs, pair_point_index_imgs


def scatter_points_to_image(image, visible_locations_x, visible_locations_y, invisible_locations_x,
                            invisible_locations_y, only_visible, point_size):
    fig = plt.figure()
    fig.set_dpi(100)
    fig.set_size_inches(image.shape[1] / 100, image.shape[0] / 100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(image, zorder=1)
    plt.scatter(x=visible_locations_x, y=visible_locations_y, s=point_size, c='b', zorder=2)
    if not only_visible:
        plt.scatter(x=invisible_locations_x, y=invisible_locations_y, s=point_size, c='y', zorder=3)
    fig.canvas.draw()

    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def type_float_and_reshape(array, shape):
    array = array.astype(np.float32)
    return array.reshape(shape)


# def get_color_file_names_by_bag(root, which_bag, bag_range, split_ratio=0.5):
#     training_image_list = []
#     rest_image_list = []
#     assert (len(bag_range) == 2)
#     for i in range(bag_range[0], bag_range[1]):
#         if i != which_bag:
#             training_image_list += list(root.glob('bag_' + str(i) + '/_start*/0*.jpg'))
#         else:
#             rest_image_list = list(root.glob('bag_' + str(i) + '/_start*/0*.jpg'))
#
#     training_image_list.sort()
#     rest_image_list.sort()
#     split_point = int(len(rest_image_list) * split_ratio)
#     return training_image_list, rest_image_list[:split_point], rest_image_list[split_point:]


def get_optimized_color_file_names_by_bag(root, which_bag, bag_range, split_ratio=(0.5, 0.5)):
    training_image_list = []
    rest_image_list = []
    for patient_id in range(bag_range[0], bag_range[1]):
        data_root = Path(root) / "bag_{}".format(patient_id)
        sub_folders = list(data_root.glob("_start*/"))
        sub_folders.sort()
        for data_root in sub_folders:
            if len(list(data_root.glob("mesh_id_*.ply"))) != 0:
                if patient_id != which_bag:
                    training_image_list += list(data_root.glob('0*.jpg'))
                else:
                    rest_image_list += list(data_root.glob('0*.jpg'))
            # if not (data_root / "mesh_id_bootstrapped_optimized.ply").exists():
            #     continue
            # else:
            #     if patient_id != which_bag:
            #         training_image_list += list(data_root.glob('0*.jpg'))
            #     else:
            #         rest_image_list += list(data_root.glob('0*.jpg'))
    training_image_list.sort()
    rest_image_list.sort()
    split_point = int(len(rest_image_list) * split_ratio[0])
    return training_image_list, rest_image_list[:split_point], rest_image_list[split_point:]

    # training_image_list = []
    # rest_image_list = []
    # assert (len(bag_range) == 2)
    # for i in range(bag_range[0], bag_range[1]):
    #     if i != which_bag:
    #         training_image_list += list(root.glob('*' + str(i) + '/_start*/0*.jpg'))
    #     else:
    #         rest_image_list = list(root.glob('*' + str(i) + '/_start*/0*.jpg'))
    #
    # training_image_list.sort()
    # rest_image_list.sort()
    # split_point = int(len(rest_image_list) * split_ratio[0])
    # return training_image_list, rest_image_list[:split_point], rest_image_list[split_point:]


def get_color_file_names(root, split_ratio=(0.9, 0.05, 0.05)):
    image_list = list(root.glob('*/_start*/0*.jpg'))
    image_list.sort()
    split_point = [int(len(image_list) * split_ratio[0]), int(len(image_list) * (split_ratio[0] + split_ratio[1]))]
    return image_list[:split_point[0]], image_list[split_point[0]:split_point[1]], image_list[split_point[1]:]


# def get_color_file_names_by_bag(root, validation_patient_id, testing_patient_id, id_range):
#     training_image_list = []
#     validation_image_list = []
#     testing_image_list = []
#
#     if not isinstance(validation_patient_id, list):
#         validation_patient_id = [validation_patient_id]
#     if not isinstance(testing_patient_id, list):
#         testing_patient_id = [testing_patient_id]
#
#     for i in range(id_range[0], id_range[1]):
#         if i not in testing_patient_id and i not in validation_patient_id:
#             training_image_list += list(root.glob('*' + str(i) + '/_start*/0*.jpg'))
#         if i in validation_patient_id:
#             validation_image_list += list(root.glob('*' + str(i) + '/_start*/0*.jpg'))
#         if i in testing_patient_id:
#             testing_image_list += list(root.glob('*' + str(i) + '/_start*/0*.jpg'))
#
#     training_image_list.sort()
#     testing_image_list.sort()
#     validation_image_list.sort()
#     return training_image_list, validation_image_list, testing_image_list


# def get_test_color_img(img_file_name, start_h, end_h, start_w, end_w, downsampling_factor, is_hsv):
#     img = cv2.imread(img_file_name)
#     downsampled_img = cv2.resize(img, (0, 0), fx=1. / downsampling_factor, fy=1. / downsampling_factor)
#     downsampled_img = downsampled_img[start_h:end_h, start_w:end_w, :]
#     if is_hsv:
#         downsampled_img = cv2.cvtColor(downsampled_img, cv2.COLOR_BGR2HSV_FULL)
#     downsampled_img = np.array(downsampled_img, dtype="float32")
#     return downsampled_img


def get_parent_folder_names(root, id_range):
    folder_list = []
    for i in range(id_range[0], id_range[1]):
        folder_list += list(root.glob('*' + str(i) + '/_start*/'))

    folder_list.sort()
    return folder_list


# def get_parent_folder_names(root, which_bag, bag_range):
#     training_folder_list = []
#     rest_folder_list = []
#     for patient_id in range(bag_range[0], bag_range[1]):
#         if patient_id != which_bag:
#             training_folder_list += list(root.glob('*' + str(patient_id) + '/_start*/'))
#         else:
#             rest_folder_list = list(root.glob('*' + str(patient_id) + '/_start*/'))
#     training_folder_list.sort()
#     rest_folder_list.sort()
#     return training_folder_list, rest_folder_list


def get_all_subfolder_names(root, bag_range):
    folder_list = []
    for i in range(bag_range[0], bag_range[1]):
        folder_list += list(root.glob('bag_' + str(i) + '/_start*/'))
    folder_list.sort()
    return folder_list


def downsample_and_crop_mask(mask, downsampling_factor, divide, suggested_h=256, suggested_w=320):
    downsampled_mask = cv2.resize(mask, (0, 0), fx=1. / downsampling_factor, fy=1. / downsampling_factor)
    end_h_index = downsampled_mask.shape[0]
    end_w_index = downsampled_mask.shape[1]
    # divide is related to the pooling times of the teacher model
    indexes = np.where(downsampled_mask >= 200)
    h = indexes[0].max() - indexes[0].min()
    w = indexes[1].max() - indexes[1].min()

    remainder_h = h % divide
    remainder_w = w % divide

    increment_h = divide - remainder_h
    increment_w = divide - remainder_w

    target_h = h + increment_h
    target_w = w + increment_w

    start_h = max(indexes[0].min() - increment_h // 2, 0)
    end_h = start_h + target_h

    start_w = max(indexes[1].min() - increment_w // 2, 0)
    end_w = start_w + target_w

    if suggested_h is not None:
        if suggested_h != h:
            remain_h = suggested_h - target_h
            start_h = max(start_h - remain_h // 2, 0)
            end_h = min(suggested_h + start_h, end_h_index)
            start_h = end_h - suggested_h

    if suggested_w is not None:
        if suggested_w != w:
            remain_w = suggested_w - target_w
            start_w = max(start_w - remain_w // 2, 0)
            end_w = min(suggested_w + start_w, end_w_index)
            start_w = end_w - suggested_w

    kernel = np.ones((5, 5), np.uint8)
    downsampled_mask_erode = cv2.erode(downsampled_mask, kernel, iterations=1)
    # print("hello", start_h, end_h, start_w, end_w)
    cropped_mask = downsampled_mask_erode[start_h:end_h, start_w:end_w]
    return cropped_mask, start_h, end_h, start_w, end_w


# def read_selected_indexes(prefix_seq):
#     selected_indexes = []
#     with open(prefix_seq + 'selected_indexes') as fp:
#         for line in fp:
#             selected_indexes.append(int(line))
#
#     stride = selected_indexes[1] - selected_indexes[0]
#     return stride, selected_indexes
#
#
# def read_visible_view_indexes(prefix_seq):
#     visible_view_indexes = []
#     if Path(prefix_seq + 'visible_view_indexes_filtered').exists():
#         with open(prefix_seq + 'visible_view_indexes_filtered') as fp:
#             for line in fp:
#                 visible_view_indexes.append(int(line))
#
#         visible_view_indexes_old = []
#         with open(prefix_seq + 'visible_view_indexes') as fp:
#             for line in fp:
#                 visible_view_indexes_old.append(int(line))
#
#         if len(visible_view_indexes) != len(visible_view_indexes_old):
#             print("We need to handle poses NOW!")
#             exit(1)
#     else:
#         with open(prefix_seq + 'visible_view_indexes') as fp:
#             for line in fp:
#                 visible_view_indexes.append(int(line))
#
#     return visible_view_indexes
#
#
# def read_camera_intrinsic_per_view(prefix_seq):
#     camera_intrinsics = []
#     param_count = 0
#     temp_camera_intrincis = np.zeros((3, 4))
#     with open(prefix_seq + 'camera_intrinsics_per_view') as fp:
#         for line in fp:
#             # Focal length
#             if param_count == 0:
#                 temp_camera_intrincis[0][0] = float(line)
#                 temp_camera_intrincis[1][1] = float(line)
#                 param_count = 1
#             elif param_count == 1:
#                 temp_camera_intrincis[0][2] = float(line)
#                 param_count = 2
#             elif param_count == 2:
#                 temp_camera_intrincis[1][2] = float(line)
#                 temp_camera_intrincis[2][2] = 1.0
#                 camera_intrinsics.append(temp_camera_intrincis)
#                 temp_camera_intrincis = np.zeros((3, 4))
#                 param_count = 0
#     return camera_intrinsics
#
#
# def modify_camera_intrinsic_matrix(intrinsic_matrix, start_h, start_w, downsampling_factor):
#     intrinsic_matrix_modified = np.copy(intrinsic_matrix)
#     intrinsic_matrix_modified[0][0] = intrinsic_matrix[0][0] / downsampling_factor
#     intrinsic_matrix_modified[1][1] = intrinsic_matrix[1][1] / downsampling_factor
#     intrinsic_matrix_modified[0][2] = intrinsic_matrix[0][2] / downsampling_factor - start_w
#     intrinsic_matrix_modified[1][2] = intrinsic_matrix[1][2] / downsampling_factor - start_h
#     return intrinsic_matrix_modified
#
#
# def read_point_cloud(prefix_seq):
#     lists_3D_points = []
#     plydata = PlyData.read(prefix_seq + "structure_filtered.ply")
#     for n in range(plydata['vertex'].count):
#         temp = list(plydata['vertex'][n])
#         temp[0] = temp[0]
#         temp[1] = temp[1]
#         temp[2] = temp[2]
#         temp.append(1.0)
#         lists_3D_points.append(temp)
#     return lists_3D_points
#
#
# def read_view_indexes_per_point(prefix_seq, visible_view_indexes, point_cloud_count):
#     # Read the view indexes per point into a 2-dimension binary matrix
#     view_indexes_per_point = np.zeros((point_cloud_count, len(visible_view_indexes)))
#     point_count = -1
#     with open(prefix_seq + 'view_indexes_per_point_filtered') as fp:
#         for line in fp:
#             if int(line) < 0:
#                 point_count = point_count + 1
#             else:
#                 view_indexes_per_point[point_count][visible_view_indexes.index(int(line))] = 1
#     return view_indexes_per_point
#
#
# def read_pose_data(prefix_seq):
#     stream = open(prefix_seq + "motion.yaml", 'r')
#     doc = yaml.load(stream)
#     keys, values = doc.items()
#     poses = values[1]
#     return poses
def read_selected_indexes(prefix_seq):
    selected_indexes = []
    with open(str(prefix_seq / 'selected_indexes')) as fp:
        for line in fp:
            selected_indexes.append(int(line))

    stride = selected_indexes[1] - selected_indexes[0]
    return stride, selected_indexes


def read_visible_image_path_list(data_root):
    visible_image_path_list = []
    visible_indexes_path_list = list(data_root.rglob("*visible_view_indexes_filtered"))
    for index_path in visible_indexes_path_list:
        with open(str(index_path)) as fp:
            for line in fp:
                visible_image_path_list.append(int(line))
    return visible_image_path_list


def read_visible_view_indexes(prefix_seq, suffix):
    path = prefix_seq / 'visible_view_indexes{}'.format(suffix)
    if not path.exists():
        return []

    visible_view_indexes = []
    with open(str(prefix_seq / 'visible_view_indexes{}'.format(suffix))) as fp:
        for line in fp:
            visible_view_indexes.append(int(line))
    return visible_view_indexes


def read_camera_intrinsic_per_view(prefix_seq, suffix):
    camera_intrinsics = []
    param_count = 0
    temp_camera_intrincis = np.zeros((3, 4))
    with open(str(prefix_seq / 'camera_intrinsics_per_view{}'.format(suffix))) as fp:
        for line in fp:
            # Focal length
            if param_count == 0:
                temp_camera_intrincis[0][0] = float(line)
                temp_camera_intrincis[1][1] = float(line)
                param_count = 1
            elif param_count == 1:
                temp_camera_intrincis[0][2] = float(line)
                param_count = 2
            elif param_count == 2:
                temp_camera_intrincis[1][2] = float(line)
                temp_camera_intrincis[2][2] = 1.0
                camera_intrinsics.append(temp_camera_intrincis)
                temp_camera_intrincis = np.zeros((3, 4))
                param_count = 0
    return camera_intrinsics


def modify_camera_intrinsic_matrix(intrinsic_matrix, start_h, start_w, downsampling_factor):
    intrinsic_matrix_modified = np.copy(intrinsic_matrix)
    intrinsic_matrix_modified[0][0] = intrinsic_matrix[0][0] / downsampling_factor
    intrinsic_matrix_modified[1][1] = intrinsic_matrix[1][1] / downsampling_factor
    intrinsic_matrix_modified[0][2] = intrinsic_matrix[0][2] / downsampling_factor - start_w
    intrinsic_matrix_modified[1][2] = intrinsic_matrix[1][2] / downsampling_factor - start_h
    return intrinsic_matrix_modified


def read_point_cloud(path):
    lists_3D_points = []
    plydata = PlyData.read(path)
    for n in range(plydata['vertex'].count):
        temp = list(plydata['vertex'][n])
        temp[0] = temp[0]
        temp[1] = temp[1]
        temp[2] = temp[2]
        temp.append(1.0)
        lists_3D_points.append(temp)
    return lists_3D_points


def read_view_indexes_per_point(prefix_seq, visible_view_indexes, point_cloud_count, suffix):
    # Read the view indexes per point into a 2-dimension binary matrix
    view_indexes_per_point = np.zeros((point_cloud_count, len(visible_view_indexes)))
    point_count = -1
    with open(str(prefix_seq / 'view_indexes_per_point{}'.format(suffix))) as fp:
        for line in fp:
            if int(line) < 0:
                point_count = point_count + 1
            else:
                view_indexes_per_point[point_count][visible_view_indexes.index(int(line))] = 1
    return view_indexes_per_point


def read_pose_data(prefix_seq, suffix):
    stream = open(str(prefix_seq / "motion{}.yaml".format(suffix)), 'r')
    doc = yaml.load(stream)
    keys, values = doc.items()
    poses = values[1]
    return poses


def get_data_balancing_scale(poses, visible_view_count):
    traveling_distance = 0.0
    translation = np.zeros((3,), dtype=np.float)
    for i in range(visible_view_count):
        pre_translation = np.copy(translation)
        translation[0] = poses["poses[" + str(i) + "]"]['position']['x']
        translation[1] = poses["poses[" + str(i) + "]"]['position']['y']
        translation[2] = poses["poses[" + str(i) + "]"]['position']['z']

        if i >= 1:
            traveling_distance += np.linalg.norm(translation - pre_translation)
    traveling_distance /= visible_view_count
    return traveling_distance


def get_extrinsic_matrix_and_projection_matrix(poses, intrinsic_matrix, visible_view_count):
    projection_matrices = []
    extrinsic_matrices = []

    for i in range(visible_view_count):
        rigid_transform = quaternion_matrix(
            [poses["poses[" + str(i) + "]"]['orientation']['w'], poses["poses[" + str(i) + "]"]['orientation']['x'],
             poses["poses[" + str(i) + "]"]['orientation']['y'],
             poses["poses[" + str(i) + "]"]['orientation']['z']])
        rigid_transform[0][3] = poses["poses[" + str(i) + "]"]['position']['x']
        rigid_transform[1][3] = poses["poses[" + str(i) + "]"]['position']['y']
        rigid_transform[2][3] = poses["poses[" + str(i) + "]"]['position']['z']

        transform = np.asmatrix(rigid_transform)
        transform = np.linalg.inv(transform)

        extrinsic_matrices.append(transform)
        projection_matrices.append(np.dot(intrinsic_matrix, transform))

    return extrinsic_matrices, projection_matrices


def get_optimized_extrinsic_matrix_and_projection_matrix(folder, intrinsic_matrix, visible_view_count):
    projection_matrices = []
    extrinsic_matrices = []
    folder_root = Path(folder)
    for i in range(visible_view_count):
        # read camera pose and inverse it to T^camera_world
        cam_pose = np.loadtxt(str(folder_root / "frame-{:06d}-optimized.pose.txt").format(i),
                              delimiter=' ')
        inverse_cam_pose = np.linalg.inv(cam_pose)
        extrinsic_matrices.append(inverse_cam_pose)
        projection_matrices.append(np.matmul(intrinsic_matrix, inverse_cam_pose))

    return extrinsic_matrices, projection_matrices


def get_color_imgs(prefix_seq, visible_view_indexes, start_h, end_h, start_w, end_w, downsampling_factor, is_hsv=False):
    imgs = []
    for i in visible_view_indexes:
        img = cv2.imread(str(prefix_seq / "{:08d}.jpg".format(i)))
        downsampled_img = cv2.resize(img, (0, 0), fx=1. / downsampling_factor, fy=1. / downsampling_factor)
        cropped_downsampled_img = downsampled_img[start_h:end_h, start_w:end_w, :]
        if is_hsv:
            cropped_downsampled_img = cv2.cvtColor(cropped_downsampled_img, cv2.COLOR_BGR2HSV_FULL)
        imgs.append(cropped_downsampled_img)
    height, width, channel = imgs[0].shape
    imgs = np.array(imgs, dtype="float32")
    imgs = np.reshape(imgs, (-1, height, width, channel))
    return imgs


def get_original_color_imgs(prefix_seq, visible_view_indexes):
    imgs = []
    for i in visible_view_indexes:
        img = cv2.imread(str(prefix_seq / "{:08d}.jpg".format(i)))
        imgs.append(img)
    height, width, channel = imgs[0].shape
    imgs = np.array(imgs, dtype="float32")
    imgs = np.reshape(imgs, (-1, height, width, channel))
    return imgs


def get_clean_point_list(imgs, point_cloud, view_indexes_per_point, mask_boundary, inlier_percentage,
                         projection_matrices,
                         extrinsic_matrices, is_hsv):
    array_3D_points = np.asarray(point_cloud).reshape((-1, 4))
    if inlier_percentage <= 0.0 or inlier_percentage >= 1.0:
        return list()

    point_cloud_contamination_accumulator = np.zeros(array_3D_points.shape[0], dtype=np.int32)
    point_cloud_appearance_count = np.zeros(array_3D_points.shape[0], dtype=np.int32)
    height, width, channel = imgs[0].shape
    valid_frame_count = 0
    mask_boundary = mask_boundary.reshape((-1, 1))
    for i in range(len(projection_matrices)):
        img = imgs[i]
        projection_matrix = projection_matrices[i]
        extrinsic_matrix = extrinsic_matrices[i]
        img = np.array(img, dtype=np.float32) / 255.0
        # imgs might be in HSV or BGR colorspace depending on the settings beyond this function
        if not is_hsv:
            img_filtered = cv2.bilateralFilter(src=img, d=7, sigmaColor=25, sigmaSpace=25)
            img_hsv = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2HSV_FULL)
        else:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_HSV2BGR_FULL)
            img_filtered = cv2.bilateralFilter(src=img_bgr, d=7, sigmaColor=25, sigmaSpace=25)
            img_hsv = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2HSV_FULL)

        view_indexes_frame = np.asarray(view_indexes_per_point[:, i]).reshape((-1))
        visible_point_indexes = np.where(view_indexes_frame > 0.5)
        visible_point_indexes = visible_point_indexes[0]
        points_3D_camera = np.einsum('ij,mj->mi', extrinsic_matrix, array_3D_points)
        points_3D_camera = points_3D_camera / points_3D_camera[:, 3].reshape((-1, 1))

        points_2D_image = np.einsum('ij,mj->mi', projection_matrix, array_3D_points)
        points_2D_image = points_2D_image / points_2D_image[:, 2].reshape((-1, 1))

        visible_points_2D_image = points_2D_image[visible_point_indexes, :].reshape((-1, 3))
        visible_points_3D_camera = points_3D_camera[visible_point_indexes, :].reshape((-1, 4))
        indexes = np.where((visible_points_2D_image[:, 0] <= width - 1) & (visible_points_2D_image[:, 0] >= 0) &
                           (visible_points_2D_image[:, 1] <= height - 1) & (visible_points_2D_image[:, 1] >= 0)
                           & (visible_points_3D_camera[:, 2] > 0))
        indexes = indexes[0]
        in_image_point_1D_locations = (np.round(visible_points_2D_image[indexes, 0]) +
                                       np.round(visible_points_2D_image[indexes, 1]) * width).astype(
            np.int32).reshape((-1))
        temp_mask = mask_boundary[in_image_point_1D_locations, :]
        indexes_2 = np.where(temp_mask[:, 0] == 255)
        indexes_2 = indexes_2[0]
        in_mask_point_1D_locations = in_image_point_1D_locations[indexes_2]
        points_depth = visible_points_3D_camera[indexes[indexes_2], 2]
        img_hsv = img_hsv.reshape((-1, 3))
        points_brightness = img_hsv[in_mask_point_1D_locations, 2]
        sanity_array = points_depth ** 2 * points_brightness
        point_cloud_appearance_count[visible_point_indexes[indexes[indexes_2]]] += 1
        if sanity_array.shape[0] < 2:
            continue
        valid_frame_count += 1
        sanity_threshold_min, sanity_threshold_max = compute_sanity_threshold(sanity_array, inlier_percentage)
        indexes_3 = np.where((sanity_array <= sanity_threshold_min) | (sanity_array >= sanity_threshold_max))
        indexes_3 = indexes_3[0]
        point_cloud_contamination_accumulator[visible_point_indexes[indexes[indexes_2[indexes_3]]]] += 1

    clean_point_cloud_array = (point_cloud_contamination_accumulator < point_cloud_appearance_count / 2).astype(
        np.float32)
    print("{} points eliminated".format(int(clean_point_cloud_array.shape[0] - np.sum(clean_point_cloud_array))))
    return clean_point_cloud_array


def compute_sanity_threshold(sanity_array, inlier_percentage):
    # Use histogram to cluster into different contaminated levels
    hist, bin_edges = np.histogram(sanity_array, bins=np.arange(1000) * np.max(sanity_array) / 1000.0,
                                   density=True)
    histogram_percentage = hist * np.diff(bin_edges)
    percentage = inlier_percentage
    # Let's assume there are a certain percent of points in each frame that are not contaminated
    # Get sanity threshold from counting histogram bins
    max_index = np.argmax(histogram_percentage)
    histogram_sum = histogram_percentage[max_index]
    pos_counter = 1
    neg_counter = 1
    # Assume the sanity value is a one-peak distribution
    while True:
        if max_index + pos_counter < len(histogram_percentage):
            histogram_sum = histogram_sum + histogram_percentage[max_index + pos_counter]
            pos_counter = pos_counter + 1
            if histogram_sum >= percentage:
                sanity_threshold_max = bin_edges[max_index + pos_counter]
                sanity_threshold_min = bin_edges[max_index - neg_counter + 1]
                break

        if max_index - neg_counter >= 0:
            histogram_sum = histogram_sum + histogram_percentage[max_index - neg_counter]
            neg_counter = neg_counter + 1
            if histogram_sum >= percentage:
                sanity_threshold_max = bin_edges[max_index + pos_counter]
                sanity_threshold_min = bin_edges[max_index - neg_counter + 1]
                break

        if max_index + pos_counter >= len(histogram_percentage) and max_index - neg_counter < 0:
            sanity_threshold_max = np.max(bin_edges)
            sanity_threshold_min = np.min(bin_edges)
            break
    return sanity_threshold_min, sanity_threshold_max


def get_contaminated_point_list(imgs, point_cloud, view_indexes_per_point, mask_boundary, inlier_percentage,
                                projection_matrices,
                                extrinsic_matrices, is_hsv):
    array_3D_points = np.asarray(point_cloud).reshape((-1, 4))
    contaminated_point_cloud_indexes = []
    if 0.0 < inlier_percentage < 1.0:
        point_cloud_contamination_accumulator = np.zeros(array_3D_points.shape[0], dtype=np.int32)
        point_cloud_appearance_count = np.zeros(array_3D_points.shape[0], dtype=np.int32)
        height, width, channel = imgs[0].shape
        valid_frame_count = 0
        mask_boundary = mask_boundary.reshape((-1, 1))
        for i in range(len(projection_matrices)):
            img = imgs[i]
            projection_matrix = projection_matrices[i]
            extrinsic_matrix = extrinsic_matrices[i]
            img = np.array(img, dtype=np.float32) / 255.0

            # imgs might be in HSV or BGR colorspace depending on the settings beyond this function
            if not is_hsv:
                img_filtered = cv2.bilateralFilter(src=img, d=7, sigmaColor=25, sigmaSpace=25)
                img_hsv = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2HSV_FULL)
            else:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_HSV2BGR_FULL)
                img_filtered = cv2.bilateralFilter(src=img_bgr, d=7, sigmaColor=25, sigmaSpace=25)
                img_hsv = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2HSV_FULL)

            view_indexes_frame = np.asarray(view_indexes_per_point[:, i]).reshape((-1))
            visible_point_indexes = np.where(view_indexes_frame > 0.5)
            visible_point_indexes = visible_point_indexes[0]
            points_3D_camera = np.einsum('ij,mj->mi', extrinsic_matrix, array_3D_points)
            points_3D_camera = points_3D_camera / points_3D_camera[:, 3].reshape((-1, 1))

            points_2D_image = np.einsum('ij,mj->mi', projection_matrix, array_3D_points)
            points_2D_image = points_2D_image / points_2D_image[:, 2].reshape((-1, 1))

            visible_points_2D_image = points_2D_image[visible_point_indexes, :].reshape((-1, 3))
            visible_points_3D_camera = points_3D_camera[visible_point_indexes, :].reshape((-1, 4))
            indexes = np.where((visible_points_2D_image[:, 0] <= width - 1) & (visible_points_2D_image[:, 0] >= 0) &
                               (visible_points_2D_image[:, 1] <= height - 1) & (visible_points_2D_image[:, 1] >= 0)
                               & (visible_points_3D_camera[:, 2] > 0))
            indexes = indexes[0]
            in_image_point_1D_locations = (np.round(visible_points_2D_image[indexes, 0]) +
                                           np.round(visible_points_2D_image[indexes, 1]) * width).astype(
                np.int32).reshape((-1))
            temp_mask = mask_boundary[in_image_point_1D_locations, :]
            indexes_2 = np.where(temp_mask[:, 0] == 255)
            indexes_2 = indexes_2[0]
            in_mask_point_1D_locations = in_image_point_1D_locations[indexes_2]
            points_depth = visible_points_3D_camera[indexes[indexes_2], 2]
            img_hsv = img_hsv.reshape((-1, 3))
            points_brightness = img_hsv[in_mask_point_1D_locations, 2]
            sanity_array = points_depth ** 2 * points_brightness
            point_cloud_appearance_count[visible_point_indexes[indexes[indexes_2]]] += 1
            if sanity_array.shape[0] < 2:
                continue
            valid_frame_count += 1
            sanity_threshold_min, sanity_threshold_max = compute_sanity_threshold(sanity_array, inlier_percentage)
            indexes_3 = np.where((sanity_array <= sanity_threshold_min) | (sanity_array >= sanity_threshold_max))
            indexes_3 = indexes_3[0]
            point_cloud_contamination_accumulator[visible_point_indexes[indexes[indexes_2[indexes_3]]]] += 1

        contaminated_point_cloud_indexes = np.where(point_cloud_contamination_accumulator >=
                                                    point_cloud_appearance_count / 2)
        contaminated_point_cloud_indexes = contaminated_point_cloud_indexes[0]
    print("{:d} points eliminated".format(contaminated_point_cloud_indexes.shape[0]))
    return contaminated_point_cloud_indexes

    # for idx in range(point_cloud_contamination_accumulator.shape[0]):
    #     if point_cloud_contamination_accumulator[idx] >= point_cloud_appearance_count[idx] // 2:
    #         contaminated_point_cloud_indexes.append(idx)

    # point_to_camera_distance_2 * img_hsv[round_v, round_u, 2]

    # view_indexes_frame = np.asarray(view_indexes_per_point[:, i]).reshape((-1))
    # visible_point_indexes = np.where(view_indexes_frame > 0.5)
    # invisible_point_indexes = np.where(view_indexes_frame <= 0.5)
    #
    # visible_points_2D_image = points_2D_image[visible_point_indexes, :].reshape((-1, 3))
    # invisible_points_2D_image = points_2D_image[invisible_point_indexes, :].reshape((-1, 3))
    #
    # visible_points_3D_camera = points_3D_camera[visible_point_indexes, :].reshape((-1, 4))
    # invisible_points_3D_camera = points_3D_camera[invisible_point_indexes, :].reshape((-1, 4))

    # indexes = np.where((visible_points_2D_image[:, 0] <= width - 1) & (visible_points_2D_image[:, 0] >= 0) &
    #                    (visible_points_2D_image[:, 1] <= height - 1) & (visible_points_2D_image[:, 1] >= 0)
    #                    & (visible_points_3D_camera[:, 2] > 0))
    # in_image_point_1D_locations = (np.round(visible_points_2D_image[indexes, 0]) +
    #                                np.round(visible_points_2D_image[indexes, 1]) * width).astype(
    #     np.int32).reshape((-1))
    # temp_mask = img_mask[in_image_point_1D_locations, :]
    # indexes_2 = np.where(temp_mask[:, 0] == 255)
    # visible_in_mask_point_1D_locations = in_image_point_1D_locations[indexes_2]

    # indexes = np.where((invisible_points_2D_image[:, 0] <= width - 1) & (invisible_points_2D_image[:, 0] >= 0) &
    #                    (invisible_points_2D_image[:, 1] <= height - 1) & (invisible_points_2D_image[:, 1] >= 0)
    #                    & (invisible_points_3D_camera[:, 2] > 0))
    # in_image_point_1D_locations = (np.round(invisible_points_2D_image[indexes, 0]) +
    #                                np.round(invisible_points_2D_image[indexes, 1]) * width).astype(
    #     np.int32).reshape((-1))
    # temp_mask = img_mask[in_image_point_1D_locations, :]
    # indexes_2 = np.where(temp_mask[:, 0] == 255)
    # invisible_in_mask_point_1D_locations = in_image_point_1D_locations[indexes_2]
    #
    # visible_locations_y = list(visible_in_mask_point_1D_locations / width)
    # visible_locations_x = list(visible_in_mask_point_1D_locations % width)
    #
    # invisible_locations_y = list(invisible_in_mask_point_1D_locations / width)
    # invisible_locations_x = list(invisible_in_mask_point_1D_locations % width)

    # for j in range(len(point_cloud)):
    #     point_3d_position = np.asarray(point_cloud[j])
    #     point_3d_position_camera = np.asarray(extrinsic_matrix).dot(point_3d_position)
    #     point_3d_position_camera = point_3d_position_camera / point_3d_position_camera[3]
    #     point_3d_position_camera = np.reshape(point_3d_position_camera[:3], (3,))
    #
    #     point_projected_undistorted = np.asarray(projection_matrix).dot(point_3d_position)
    #     point_projected_undistorted = point_projected_undistorted / point_projected_undistorted[2]
    #
    #     if np.isnan(point_projected_undistorted[0]) or np.isnan(point_projected_undistorted[1]):
    #         continue
    #
    #     round_u = int(round(point_projected_undistorted[0]))
    #     round_v = int(round(point_projected_undistorted[1]))
    #
    #     # We will treat this point as valid if it is projected onto the mask region
    #     if 0 <= round_u < width and 0 <= round_v < height and \
    #             mask_boundary[round_v, round_u] > 220 and point_3d_position_camera[2] > 0.0:
    #         point_to_camera_distance_2 = np.dot(point_3d_position_camera[:3], point_3d_position_camera[:3])
    #         sanity_array.append(point_to_camera_distance_2 * img_hsv[round_v, round_u, 2])
    # if len(sanity_array) >= 2:
    # for j in range(len(point_cloud)):
    #     point_3d_position = np.asarray(point_cloud[j])
    #     point_3d_position_camera = np.asarray(extrinsic_matrix).dot(point_3d_position)
    #     point_3d_position_camera = point_3d_position_camera / point_3d_position_camera[3]
    #     point_3d_position_camera = np.reshape(point_3d_position_camera[:3], (3,))
    #
    #     point_projected_undistorted = np.asarray(projection_matrix).dot(point_3d_position)
    #     point_projected_undistorted = point_projected_undistorted / point_projected_undistorted[2]
    #
    #     if np.isnan(point_projected_undistorted[0]) or np.isnan(point_projected_undistorted[1]):
    #         continue
    #
    #     round_u = int(round(point_projected_undistorted[0]))
    #     round_v = int(round(point_projected_undistorted[1]))
    #
    #     if 0 <= round_u < width and 0 <= round_v < height and \
    #             mask_boundary[round_v, round_u] > 220 and point_3d_position_camera[2] > 0.0:
    #         point_to_camera_distance_2 = np.dot(point_3d_position_camera[:3],
    #                                             point_3d_position_camera[:3])
    #         sanity_value = point_to_camera_distance_2 * img_hsv[round_v, round_u, 2]
    #         point_cloud_appearance_count[j] += 1
    #         if sanity_value <= sanity_threshold_min or sanity_value >= sanity_threshold_max:
    #             point_cloud_contamination_accumulator[j] += 1


def get_visible_count_per_point(view_indexes_per_point):
    appearing_count = np.reshape(np.sum(view_indexes_per_point, axis=-1), (-1, 1))
    return appearing_count


def generating_pos_and_increment(idx, visible_view_indexes, adjacent_range):
    # We use the remainder of the overall idx to retrieve the visible view
    visible_view_idx = idx % len(visible_view_indexes)

    adjacent_range_list = []
    adjacent_range_list.append(adjacent_range[0])
    adjacent_range_list.append(adjacent_range[1])

    if len(visible_view_indexes) <= 2 * adjacent_range_list[0]:
        adjacent_range_list[0] = len(visible_view_indexes) // 2

    if visible_view_idx <= adjacent_range_list[0] - 1:
        increment = random.randint(adjacent_range_list[0],
                                   min(adjacent_range_list[1], len(visible_view_indexes) - 1 - visible_view_idx))
    elif visible_view_idx >= len(visible_view_indexes) - adjacent_range_list[0]:
        increment = -random.randint(adjacent_range_list[0], min(adjacent_range_list[1], visible_view_idx))

    else:
        # which direction should we increment
        direction = random.randint(0, 1)
        if direction == 1:
            increment = random.randint(adjacent_range_list[0],
                                       min(adjacent_range_list[1], len(visible_view_indexes) - 1 - visible_view_idx))
        else:
            increment = -random.randint(adjacent_range_list[0], min(adjacent_range_list[1], visible_view_idx))

    return [visible_view_idx, increment]


# def get_pair_color_imgs(prefix_seq, pair_indexes, start_h, end_h, start_w, end_w, downsampling_factor, is_hsv):
#     imgs = []
#     for i in pair_indexes:
#         img = cv2.imread((prefix_seq + "%08d.jpg") % i)
#         downsampled_img = cv2.resize(img, (0, 0), fx=1. / downsampling_factor, fy=1. / downsampling_factor)
#         downsampled_img = downsampled_img[start_h:end_h, start_w:end_w, :]
#         if is_hsv:
#             downsampled_img = cv2.cvtColor(downsampled_img, cv2.COLOR_BGR2HSV_FULL)
#         imgs.append(downsampled_img)
#     height, width, channel = imgs[0].shape
#     imgs = np.array(imgs, dtype="float32")
#     imgs = np.reshape(imgs, (-1, height, width, channel))
#     return imgs


def get_single_color_img(prefix_seq, index, start_h, end_h, start_w, end_w, downsampling_factor, is_hsv, rgb_mode):
    img = cv2.imread(str(prefix_seq / "{:08d}.jpg".format(index)))
    downsampled_img = cv2.resize(img, (0, 0), fx=1. / downsampling_factor, fy=1. / downsampling_factor)
    downsampled_img = downsampled_img[start_h:end_h, start_w:end_w, :]
    if is_hsv:
        downsampled_img = cv2.cvtColor(downsampled_img, cv2.COLOR_BGR2HSV_FULL)
    else:
        if rgb_mode == "rgb":
            downsampled_img = cv2.cvtColor(downsampled_img, cv2.COLOR_BGR2RGB)
    downsampled_img = np.array(downsampled_img, dtype="float32")
    return downsampled_img


# # TODO: Need to handle these point outliers
# def get_torch_training_data(pair_extrinsics, pair_projections, pair_indexes, point_cloud, mask_boundary,
#                             view_indexes_per_point, contamination_point_list, visible_view_indexes):
#     height = mask_boundary.shape[0]
#     width = mask_boundary.shape[1]
#     pair_depth_mask_imgs = []
#     pair_depth_imgs = []
#
#     pair_flow_imgs = []
#     flow_image_1 = np.zeros((height, width, 2), dtype=np.float32)
#     flow_image_2 = np.zeros((height, width, 2), dtype=np.float32)
#
#     pair_flow_mask_imgs = []
#     flow_mask_image_1 = np.zeros((height, width, 1), dtype=np.float32)
#     flow_mask_image_2 = np.zeros((height, width, 1), dtype=np.float32)
#
#     array_3D_points = np.asarray(point_cloud).reshape((-1, 4))
#     for i in range(2):
#         projection_matrix = pair_projections[i]
#         extrinsic_matrix = pair_extrinsics[i]
#
#         if i == 0:
#             points_2D_image_1 = np.einsum('ij,mj->mi', projection_matrix, array_3D_points)
#             points_2D_image_1 = np.round(points_2D_image_1 / points_2D_image_1[:, 2].reshape((-1, 1)))
#             points_3D_camera_1 = np.einsum('ij,mj->mi', extrinsic_matrix, array_3D_points)
#             points_3D_camera_1 = points_3D_camera_1 / points_3D_camera_1[:, 3].reshape((-1, 1))
#         else:
#             points_2D_image_2 = np.einsum('ij,mj->mi', projection_matrix, array_3D_points)
#             points_2D_image_2 = np.round(points_2D_image_2 / points_2D_image_2[:, 2].reshape((-1, 1)))
#             points_3D_camera_2 = np.einsum('ij,mj->mi', extrinsic_matrix, array_3D_points)
#             points_3D_camera_2 = points_3D_camera_2 / points_3D_camera_2[:, 3].reshape((-1, 1))
#
#     mask_boundary = mask_boundary.reshape((-1, 1))
#     flow_image_1 = flow_image_1.reshape((-1, 2))
#     flow_image_2 = flow_image_2.reshape((-1, 2))
#     flow_mask_image_1 = flow_mask_image_1.reshape((-1, 1))
#     flow_mask_image_2 = flow_mask_image_2.reshape((-1, 1))
#
#     points_2D_image_1 = points_2D_image_1.reshape((-1, 3))
#     points_2D_image_2 = points_2D_image_2.reshape((-1, 3))
#     points_3D_camera_1 = points_3D_camera_1.reshape((-1, 4))
#     points_3D_camera_2 = points_3D_camera_2.reshape((-1, 4))
#
#     point_visibility_1 = np.asarray(view_indexes_per_point[:, visible_view_indexes.index(pair_indexes[0])]).reshape(
#         (-1))
#     visible_point_indexes_1 = np.where(point_visibility_1 > 0.5)
#     visible_point_indexes_1 = visible_point_indexes_1[0]
#     point_visibility_2 = np.asarray(view_indexes_per_point[:, visible_view_indexes.index(pair_indexes[1])]).reshape(
#         (-1))
#     visible_point_indexes_2 = np.where(point_visibility_2 > 0.5)
#     visible_point_indexes_2 = visible_point_indexes_2[0]
#     visible_points_3D_camera_1 = points_3D_camera_1[visible_point_indexes_1, :].reshape((-1, 4))
#     visible_points_2D_image_1 = points_2D_image_1[visible_point_indexes_1, :].reshape((-1, 3))
#     visible_points_3D_camera_2 = points_3D_camera_2[visible_point_indexes_2, :].reshape((-1, 4))
#     visible_points_2D_image_2 = points_2D_image_2[visible_point_indexes_2, :].reshape((-1, 3))
#
#     in_image_indexes_1 = np.where(
#         (visible_points_2D_image_1[:, 0] <= width - 1) & (visible_points_2D_image_1[:, 0] >= 0) &
#         (visible_points_2D_image_1[:, 1] <= height - 1) & (visible_points_2D_image_1[:, 1] >= 0)
#         & (visible_points_3D_camera_1[:, 2] > 0))
#     in_image_indexes_1 = in_image_indexes_1[0]
#     in_image_point_1D_locations_1 = (np.round(visible_points_2D_image_1[in_image_indexes_1, 0]) +
#                                      np.round(visible_points_2D_image_1[in_image_indexes_1, 1]) * width).astype(
#         np.int32).reshape((-1))
#     temp_mask_1 = mask_boundary[in_image_point_1D_locations_1, :]
#     in_mask_indexes_1 = np.where(temp_mask_1[:, 0] == 255)
#     in_mask_indexes_1 = in_mask_indexes_1[0]
#     in_mask_point_1D_locations_1 = in_image_point_1D_locations_1[in_mask_indexes_1]
#     flow_mask_image_1[in_mask_point_1D_locations_1, 0] = 1.0
#
#     in_image_indexes_2 = np.where(
#         (visible_points_2D_image_2[:, 0] <= width - 1) & (visible_points_2D_image_2[:, 0] >= 0) &
#         (visible_points_2D_image_2[:, 1] <= height - 1) & (visible_points_2D_image_2[:, 1] >= 0)
#         & (visible_points_3D_camera_2[:, 2] > 0))
#     in_image_indexes_2 = in_image_indexes_2[0]
#     in_image_point_1D_locations_2 = (np.round(visible_points_2D_image_2[in_image_indexes_2, 0]) +
#                                      np.round(visible_points_2D_image_2[in_image_indexes_2, 1]) * width).astype(
#         np.int32).reshape((-1))
#     temp_mask_2 = mask_boundary[in_image_point_1D_locations_2, :]
#     in_mask_indexes_2 = np.where(temp_mask_2[:, 0] == 255)
#     in_mask_indexes_2 = in_mask_indexes_2[0]
#     in_mask_point_1D_locations_2 = in_image_point_1D_locations_2[in_mask_indexes_2]
#     flow_mask_image_2[in_mask_point_1D_locations_2, 0] = 1.0
#
#     flow_image_1[in_mask_point_1D_locations_1, :] = points_2D_image_1[
#                                                     visible_point_indexes_1[in_image_indexes_1[in_mask_indexes_1]],
#                                                     :2] - \
#                                                     points_2D_image_2[
#                                                     visible_point_indexes_1[in_image_indexes_1[in_mask_indexes_1]], :2]
#     flow_image_2[in_mask_point_1D_locations_2, :] = points_2D_image_2[
#                                                     visible_point_indexes_2[in_image_indexes_2[in_mask_indexes_2]],
#                                                     :2] - \
#                                                     points_2D_image_1[
#                                                     visible_point_indexes_2[in_image_indexes_2[in_mask_indexes_2]], :2]
#
#     flow_image_1[:, 0] /= width
#     flow_image_1[:, 1] /= height
#     flow_image_2[:, 0] /= width
#     flow_image_2[:, 1] /= height
#
#     depth_img_1 = np.zeros((height, width, 1), dtype=np.float32)
#     depth_img_2 = np.zeros((height, width, 1), dtype=np.float32)
#     depth_mask_img_1 = np.zeros((height, width, 1), dtype=np.float32)
#     depth_mask_img_2 = np.zeros((height, width, 1), dtype=np.float32)
#
#     depth_img_1[in_mask_point_1D_locations_1, 0] = points_3D_camera_1[
#         visible_point_indexes_1[in_image_indexes_1[in_mask_indexes_1]], 2]
#     depth_img_2[in_mask_point_1D_locations_2, 0] = points_3D_camera_2[
#         visible_point_indexes_2[in_image_indexes_2[in_mask_indexes_2]], 2]
#     depth_mask_img_1[in_mask_point_1D_locations_1, 0] = 1.0
#     depth_mask_img_2[in_mask_point_1D_locations_2, 0] = 1.0
#
#     pair_flow_imgs.append(flow_image_1)
#     pair_flow_imgs.append(flow_image_2)
#     pair_flow_imgs = np.array(pair_flow_imgs, dtype="float32")
#     pair_flow_imgs = np.reshape(pair_flow_imgs, (-1, height, width, 2))
#
#     pair_flow_mask_imgs.append(flow_mask_image_1)
#     pair_flow_mask_imgs.append(flow_mask_image_2)
#     pair_flow_mask_imgs = np.array(pair_flow_mask_imgs, dtype="float32")
#     pair_flow_mask_imgs = np.reshape(pair_flow_mask_imgs, (-1, height, width, 1))
#
#     pair_depth_mask_imgs.append(depth_mask_img_1)
#     pair_depth_mask_imgs.append(depth_mask_img_2)
#     pair_depth_mask_imgs = np.array(pair_depth_mask_imgs, dtype="float32")
#     pair_depth_mask_imgs = np.reshape(pair_depth_mask_imgs, (-1, height, width, 1))
#
#     pair_depth_imgs.append(depth_img_1)
#     pair_depth_imgs.append(depth_img_2)
#     pair_depth_imgs = np.array(pair_depth_imgs, dtype="float32")
#     pair_depth_imgs = np.reshape(pair_depth_imgs, (-1, height, width, 1))
#
#     return pair_depth_mask_imgs, pair_depth_imgs, pair_flow_mask_imgs, pair_flow_imgs
#
#     # flow_array_1_to_2 = points_2D_image_2[, :2] - points_2D_image_1[, :2]
#     # flow_array_1_to_2[:, 0] /= width
#     # flow_array_1_to_2[:, 1] /= height
#     # count = 0
#     # for j in range(len(point_cloud)):
#     #     if j in contamination_point_list:
#     #         continue
#     #     point_3d_position = np.asarray(point_cloud[j])
#     #     point_projected_undistorted = np.asarray(projection_matrix).dot(point_3d_position)
#     #     point_projected_undistorted = point_projected_undistorted / point_projected_undistorted[2]
#     #
#     #     if np.isnan(point_projected_undistorted[0]) or np.isnan(point_projected_undistorted[1]):
#     #         continue
#     #
#     #     round_u = int(round(point_projected_undistorted[0]))
#     #     round_v = int(round(point_projected_undistorted[1]))
#     #
#     #     if i == 0:
#     #         point_projection_positions_1[count][0] = round_u
#     #         point_projection_positions_1[count][1] = round_v
#     #
#     #     elif i == 1:
#     #         point_projection_positions_2[count][0] = round_u
#     #         point_projection_positions_2[count][1] = round_v
#     #
#     #     count += 1
#
#     # count = 0
#     #
#     # for i in range(len(point_cloud)):
#     #     if i in contamination_point_list:
#     #         continue
#     #     u = point_projection_positions_1[count][0]
#     #     v = point_projection_positions_1[count][1]
#     #     u2 = point_projection_positions_2[count][0]
#     #     v2 = point_projection_positions_2[count][1]
#     #
#     #     if 0 <= u < width and 0 <= v < height:
#     #         if mask_boundary[int(v), int(u)] > 220:
#     #             distance = np.abs(float(u2 - u) / width) + np.abs(float(v2 - v) / height)
#     #             if distance <= 1.0:
#     #                 flow_image_1[int(v)][int(u)][0] = float(u2 - u) / width
#     #                 flow_image_1[int(v)][int(u)][1] = float(v2 - v) / height
#     #
#     #                 if use_view_indexes_per_point:
#     #                     if view_indexes_per_point[i][visible_view_indexes.index(pair_indexes[0])] > 0.5:
#     #                         flow_mask_image_1[int(v)][int(u)] = 1.0
#     #
#     #                     #     - np.exp(
#     #                     #                                 -appearing_count_per_point[i, 0] /
#     #                     #                                 count_weight)
#     #                     else:
#     #                         flow_mask_image_1[int(v)][int(u)] = 0.0
#     #                 else:
#     #                     flow_mask_image_1[int(v)][int(u)] = 1.0
#     #                     # - np.exp(
#     #                     #                             -appearing_count_per_point[i, 0] /
#     #                     #                             count_weight)
#     #                 # np.exp(-1.0 / (flow_factor * mean_flow_length) * distance)
#     #
#     #     if 0 <= u2 < width and 0 <= v2 < height:
#     #         if mask_boundary[int(v2), int(u2)] > 220:
#     #             distance = np.abs(float(u - u2) / width) + np.abs(float(v - v2) / height)
#     #             if distance <= 1.0:
#     #                 flow_image_2[int(v2)][int(u2)][0] = float(u - u2) / width
#     #                 flow_image_2[int(v2)][int(u2)][1] = float(v - v2) / height
#     #                 if use_view_indexes_per_point:
#     #                     if view_indexes_per_point[i][visible_view_indexes.index(pair_indexes[1])] > 0.5:
#     #                         flow_mask_image_2[int(v2)][int(u2)] = 1.0
#     #                         #  - np.exp(
#     #                         #                                 -appearing_count_per_point[i, 0] /
#     #                         #                                 count_weight)
#     #                     else:
#     #                         flow_mask_image_2[int(v2)][int(u2)] = 0.0
#     #                 else:
#     #                     flow_mask_image_2[int(v2)][int(u2)] = 1.0
#     #                     #  - np.exp(
#     #                     #                             -appearing_count_per_point[i, 0] /
#     #                     #                             count_weight)
#     #     count += 1
#     # for i in range(2):
#     #     img = pair_images[i]
#     #
#     #     if visualize:
#     #         display_img = np.copy(img)
#     #
#     #     projection_matrix = pair_projections[i]
#     #     extrinsic_matrix = pair_extrinsics[i]
#     #
#     #     masked_depth_img = np.zeros((height, width))
#     #     mask_img = np.zeros((height, width))
#     #
#     #     if use_view_indexes_per_point:
#     #         for j in range(len(point_cloud)):
#     #             if j in contamination_point_list:
#     #                 continue
#     #             point_3d_position = np.asarray(point_cloud[j])
#     #             point_3d_position_camera = np.asarray(extrinsic_matrix).dot(point_3d_position)
#     #             point_3d_position_camera = np.copy(point_3d_position_camera / point_3d_position_camera[3])
#     #
#     #             point_projected_undistorted = np.asarray(projection_matrix).dot(point_3d_position)
#     #             point_projected_undistorted = point_projected_undistorted / point_projected_undistorted[2]
#     #
#     #             if np.isnan(point_projected_undistorted[0]) or np.isnan(point_projected_undistorted[1]):
#     #                 continue
#     #
#     #             round_u = int(round(point_projected_undistorted[0]))
#     #             round_v = int(round(point_projected_undistorted[1]))
#     #             if view_indexes_per_point[j][visible_view_indexes.index(pair_indexes[i])] > 0.5:
#     #                 if 0 <= round_u < width and 0 <= round_v < height and \
#     #                         mask_boundary[round_v, round_u] > 220 and point_3d_position_camera[2] > 0.0:
#     #                     mask_img[round_v][
#     #                         round_u] = 1.0
#     #                     # - np.exp(-appearing_count_per_point[j, 0] / count_weight)
#     #                     masked_depth_img[round_v][round_u] = point_3d_position_camera[2]
#     #                     if visualize:
#     #                         cv2.circle(display_img, (round_u, round_v), 1,
#     #                                    (0, int(mask_img[round_v][round_u] * 255), 0))
#     #     else:
#     #         for j in range(len(point_cloud)):
#     #             if j in contamination_point_list:
#     #                 continue
#     #             point_3d_position = np.asarray(point_cloud[j])
#     #             point_3d_position_camera = np.asarray(extrinsic_matrix).dot(point_3d_position)
#     #             point_3d_position_camera = np.copy(point_3d_position_camera / point_3d_position_camera[3])
#     #
#     #             point_projected_undistorted = np.asarray(projection_matrix).dot(point_3d_position)
#     #             point_projected_undistorted[0] = point_projected_undistorted[0] / point_projected_undistorted[2]
#     #             point_projected_undistorted[1] = point_projected_undistorted[1] / point_projected_undistorted[2]
#     #
#     #             if np.isnan(point_projected_undistorted[0]) or np.isnan(point_projected_undistorted[1]):
#     #                 continue
#     #
#     #             round_u = int(round(point_projected_undistorted[0]))
#     #             round_v = int(round(point_projected_undistorted[1]))
#     #             if 0 <= round_u < width and 0 <= round_v < height and \
#     #                     mask_boundary[round_v, round_u] > 220 and point_3d_position_camera[2] > 0.0:
#     #                 mask_img[round_v][round_u] = 1.0
#     #                 # - np.exp(-appearing_count_per_point[j, 0] / count_weight)
#     #                 masked_depth_img[round_v][round_u] = point_3d_position_camera[2]
#     #                 if visualize:
#     #                     cv2.circle(display_img, (round_u, round_v), 1,
#     #                                (0, int(mask_img[round_v][round_u] * 255), 0))
#     #     if visualize:
#     #         cv2.imshow("img", np.uint8(display_img))
#     #         cv2.waitKey()
#     #
#     #     pair_mask_imgs.append(mask_img)
#     #     pair_sparse_depth_imgs.append(masked_depth_img)
#     #
#     # if visualize:
#     #     cv2.destroyAllWindows()
#

def display_colors(idx, step, writer, colors_1, phase="Training", is_return_image=False):
    colors_display = vutils.make_grid(colors_1 * 0.5 + 0.5, normalize=False)
    colors_display = np.moveaxis(colors_display.data.cpu().numpy(),
                                 source=[0, 1, 2], destination=[2, 0, 1])
    colors_display[colors_display < 0.0] = 0.0
    colors_display[colors_display > 1.0] = 1.0

    if is_return_image:
        return colors_display
    else:
        writer.add_image(phase + '/Images/Color_' + str(idx), colors_display, step, dataformats="HWC")
        return


def display_orb_feature_matches(step, writer, matches_display, phase="Training",
                                color_reverse=True, is_return_image=False):
    matches_display = vutils.make_grid(matches_display * 0.5 + 0.5, normalize=False)
    matches_display = np.moveaxis(matches_display.data.numpy(), source=[0, 1, 2], destination=[2, 0, 1])
    if color_reverse:
        matches_display = cv2.cvtColor(matches_display, cv2.COLOR_BGR2RGB)

    if is_return_image:
        return matches_display
    else:
        writer.add_image(phase + '/Images/ORB_Matches', matches_display, step, dataformats="HWC")
        return


def display_feature_response_map(idx, step, title, writer, feature_response_heat_map, phase="Training",
                                 color_reverse=True, is_return_image=False):
    batch_size, _, height, width = feature_response_heat_map.shape
    feature_response_heat_map = feature_response_heat_map.view(batch_size, 1, height, width)
    heatmap_display = vutils.make_grid(feature_response_heat_map, normalize=False, scale_each=True)
    heatmap_display = cv2.applyColorMap(np.uint8(255 * np.moveaxis(heatmap_display.data.cpu().numpy(),
                                                                   source=[0, 1, 2], destination=[2, 0, 1])),
                                        cv2.COLORMAP_HOT)
    if color_reverse:
        heatmap_display = cv2.cvtColor(heatmap_display, cv2.COLOR_BGR2RGB)

    if is_return_image:
        return heatmap_display
    else:
        writer.add_image(phase + '/Images/' + title + str(idx), heatmap_display, step, dataformats="HWC")
        return


def stack_and_display(phase, title, step, writer, image_list):
    writer.add_image(phase + '/Images/' + title, np.vstack(image_list), step, dataformats='HWC')
    return


def keypoints_descriptors_extraction(descriptor, color_1, color_2, boundary):
    color_1 = color_1.data.cpu().numpy()
    boundary = boundary.data.cpu().numpy()
    _, height, width = color_1.shape
    color_1 = np.moveaxis(color_1, source=[0, 1, 2], destination=[2, 0, 1])
    color_1 = np.uint8(255 * (color_1 * 0.5 + 0.5))
    boundary = np.uint8(255 * boundary.reshape((height, width)))
    kps_1, des_1 = descriptor.detectAndCompute(color_1, mask=boundary)

    color_2 = color_2.data.cpu().numpy()
    color_2 = np.moveaxis(color_2, source=[0, 1, 2], destination=[2, 0, 1])
    color_2 = np.uint8(255 * (color_2 * 0.5 + 0.5))
    kps_2, des_2 = descriptor.detectAndCompute(color_2, mask=boundary)

    kps_1D_1 = []
    kps_1D_2 = []

    if kps_1 is None or kps_2 is None or len(kps_1) == 0 or len(kps_2) == 0:
        return None

    for point in kps_1:
        kps_1D_1.append(np.round(point.pt[0]) + np.round(point.pt[1]) * width)
    for point in kps_2:
        kps_1D_2.append(np.round(point.pt[0]) + np.round(point.pt[1]) * width)

    return kps_1, kps_2, des_1, des_2, np.asarray(kps_1D_1), np.asarray(kps_1D_2)


def feature_matching_single_generation_raw_color(color_1, color_2, feature_map_1, feature_map_2,
                                                 kps_1D_1, cross_check_distance,
                                                 kps_1, display_matches, des_1=None, des_2=None, kps_2=None, gpu_id=0):
    with torch.no_grad():
        # Feature map C x H x W
        feature_length, height, width = feature_map_1.shape

        # Extend 1D locations to B x C x Sampling_size
        keypoint_number = len(kps_1D_1)
        source_feature_1d_locations = torch.from_numpy(kps_1D_1).long().cuda(gpu_id).view(
            1, 1,
            keypoint_number).expand(
            -1, feature_length, -1)

        # Sampled rough locator feature vectors
        sampled_feature_vectors = torch.gather(
            feature_map_1.view(1, feature_length, height * width), 2,
            source_feature_1d_locations.long())
        sampled_feature_vectors = sampled_feature_vectors.view(1, feature_length,
                                                               keypoint_number,
                                                               1,
                                                               1).permute(0, 2, 1, 3,
                                                                          4).view(1,
                                                                                  keypoint_number,
                                                                                  feature_length,
                                                                                  1, 1)

        # 1 x Sampling_size x H x W
        filter_response_map = torch.nn.functional.conv2d(
            input=feature_map_2.view(1, feature_length, height, width),
            weight=sampled_feature_vectors.view(keypoint_number,
                                                feature_length,
                                                1, 1), padding=0)

        max_reponses, max_indexes = torch.max(filter_response_map.view(keypoint_number, -1), dim=1,
                                              keepdim=False)
        del filter_response_map
        torch.cuda.empty_cache()

        # query is 1 and train is 2 here
        detected_target_1d_locations = max_indexes.view(-1)
        selected_max_responses = max_reponses.view(-1)
        # Do cross check
        feature_1d_locations_2 = detected_target_1d_locations.long().view(
            1, 1, -1).expand(-1, feature_length, -1)

        # Sampled rough locator feature vectors
        sampled_feature_vectors_2 = torch.gather(
            feature_map_2.view(1, feature_length, height * width), 2,
            feature_1d_locations_2.long())
        sampled_feature_vectors_2 = sampled_feature_vectors_2.view(1, feature_length,
                                                                   keypoint_number,
                                                                   1,
                                                                   1).permute(0, 2, 1, 3,
                                                                              4).view(1,
                                                                                      keypoint_number,
                                                                                      feature_length,
                                                                                      1, 1)

        # 1 x Sampling_size x H x W
        source_filter_response_map = torch.nn.functional.conv2d(
            input=feature_map_1.view(1, feature_length, height, width),
            weight=sampled_feature_vectors_2.view(keypoint_number,
                                                  feature_length,
                                                  1, 1), padding=0)

        max_reponses_2, max_indexes_2 = torch.max(source_filter_response_map.view(keypoint_number, -1),
                                                  dim=1,
                                                  keepdim=False)
        del source_filter_response_map
        torch.cuda.empty_cache()

        keypoint_1d_locations_1 = torch.from_numpy(np.asarray(kps_1D_1)).float().cuda(gpu_id).view(
            keypoint_number, 1)
        keypoint_2d_locations_1 = torch.cat(
            [torch.fmod(keypoint_1d_locations_1, width),
             torch.floor(keypoint_1d_locations_1 / width)],
            dim=1).view(keypoint_number, 2).float()

        detected_source_keypoint_1d_locations = max_indexes_2.float().view(keypoint_number, 1)
        detected_source_keypoint_2d_locations = torch.cat(
            [torch.fmod(detected_source_keypoint_1d_locations, width),
             torch.floor(detected_source_keypoint_1d_locations / width)],
            dim=1).view(keypoint_number, 2).float()

        # We will accept the feature matches if the max indexes here is
        # not far away from the original key point location from descriptor
        cross_check_correspondence_distances = torch.norm(
            keypoint_2d_locations_1 - detected_source_keypoint_2d_locations, dim=1, p=2).view(
            keypoint_number)
        valid_correspondence_indexes = torch.nonzero(cross_check_correspondence_distances < cross_check_distance).view(
            -1)

        if valid_correspondence_indexes.shape[0] == 0:
            return None

        valid_detected_1d_locations_2 = torch.gather(detected_target_1d_locations.long().view(-1),
                                                     0, valid_correspondence_indexes.long())
        valid_max_responses = torch.gather(selected_max_responses.view(-1), 0,
                                           valid_correspondence_indexes.long())

        if display_matches:
            valid_detected_1d_locations_2 = valid_detected_1d_locations_2.data.cpu().numpy()
            valid_max_responses = valid_max_responses.data.cpu().numpy()
            valid_correspondence_indexes = valid_correspondence_indexes.data.cpu().numpy()
            detected_keypoints_2 = []
            for index in valid_detected_1d_locations_2:
                detected_keypoints_2.append(
                    cv2.KeyPoint(x=float(np.floor(index % width)), y=float(np.floor(index / width)), _size=1.0))

            matches = []
            for i, (query_index, response) in enumerate(
                    zip(valid_correspondence_indexes, valid_max_responses)):
                matches.append(cv2.DMatch(_queryIdx=query_index, _trainIdx=i, _distance=response))

            display_matches_ai = cv2.drawMatches(color_1, kps_1, color_2, detected_keypoints_2, matches,
                                                 flags=2, outImg=None)
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            feature_matches_craft = bf.knnMatch(des_1, des_2, k=1)
            good = []
            for m in feature_matches_craft:
                if len(m) != 0:
                    good.append(m[0])
            display_matches_craft = cv2.drawMatches(color_1, kps_1, color_2, kps_2, good, flags=2,
                                                    outImg=None)
            return display_matches_ai, display_matches_craft
        else:
            valid_detected_1d_locations_2 = valid_detected_1d_locations_2.float()
            valid_detected_target_2d_locations = torch.cat(
                [torch.fmod(valid_detected_1d_locations_2, width).view(-1, 1),
                 torch.floor(valid_detected_1d_locations_2 / width).view(-1, 1)],
                dim=1).view(-1, 2).float()
            valid_source_keypoint_indexes = valid_correspondence_indexes.view(-1, 1).data.cpu().numpy()
            valid_detected_target_2d_locations = valid_detected_target_2d_locations.view(-1, 2).data.cpu().numpy()
            return valid_source_keypoint_indexes, valid_detected_target_2d_locations


def feature_matching_single_generation_subpixel(color_1, color_2, feature_map_1, feature_map_2,
                                                kps_1D_1, cross_check_distance,
                                                kps_1, display_matches, downsampling, des_1=None, des_2=None,
                                                kps_2=None, gpu_id=0):
    with torch.no_grad():
        # Feature map C x H x W
        feature_length, height, width = feature_map_1.shape

        # Extend 1D locations to B x C x Sampling_size
        keypoint_number = len(kps_1D_1)
        source_feature_1d_locations = torch.from_numpy(kps_1D_1).long().cuda(gpu_id).view(
            1, 1,
            keypoint_number).expand(
            -1, feature_length, -1)

        # Sampled rough locator feature vectors
        sampled_feature_vectors = torch.gather(
            feature_map_1.view(1, feature_length, height * width), 2,
            source_feature_1d_locations.long())
        sampled_feature_vectors = sampled_feature_vectors.view(1, feature_length,
                                                               keypoint_number,
                                                               1,
                                                               1).permute(0, 2, 1, 3,
                                                                          4).view(1,
                                                                                  keypoint_number,
                                                                                  feature_length,
                                                                                  1, 1)

        # 1 x Sampling_size x H x W
        filter_response_map = torch.nn.functional.conv2d(
            input=feature_map_2.view(1, feature_length, height, width),
            weight=sampled_feature_vectors.view(keypoint_number,
                                                feature_length,
                                                1, 1), padding=0)

        # Upsample the filter response map to the original size and find the subpixel location
        max_reponses, max_indexes = torch.max(
            torch.nn.functional.interpolate(filter_response_map, scale_factor=downsampling, mode='bicubic').view(
                keypoint_number, -1), dim=1,
            keepdim=False)

        del sampled_feature_vectors, filter_response_map, source_feature_1d_locations

        # query is 1 and train is 2 here
        full_detected_target_1d_locations = max_indexes.view(-1).float()
        full_detected_target_2d_locations = torch.cat(
            [torch.fmod(full_detected_target_1d_locations, width * downsampling).view(-1, 1),
             torch.floor(full_detected_target_1d_locations / (width * downsampling)).view(-1, 1)],
            dim=1).view(keypoint_number, 2).float()
        detected_target_2d_locations = full_detected_target_2d_locations / downsampling
        detected_target_1d_locations = torch.clamp(torch.round(detected_target_2d_locations[:, 0]), min=0,
                                                   max=width - 1) + \
                                       torch.clamp(torch.round(detected_target_2d_locations[:, 1]), min=0,
                                                   max=height - 1) * width

        selected_max_responses = max_reponses.view(-1)

        feature_1d_locations_2 = detected_target_1d_locations.long().view(
            1, 1, -1).expand(-1, feature_length, -1)

        # Sampled rough locator feature vectors
        sampled_feature_vectors_2 = torch.gather(
            feature_map_2.view(1, feature_length, height * width), 2,
            feature_1d_locations_2.long())
        sampled_feature_vectors_2 = sampled_feature_vectors_2.view(1, feature_length,
                                                                   keypoint_number,
                                                                   1,
                                                                   1).permute(0, 2, 1, 3,
                                                                              4).view(1,
                                                                                      keypoint_number,
                                                                                      feature_length,
                                                                                      1, 1)

        # 1 x Sampling_size x H x W
        source_filter_response_map = torch.nn.functional.conv2d(
            input=feature_map_1.view(1, feature_length, height, width),
            weight=sampled_feature_vectors_2.view(keypoint_number,
                                                  feature_length,
                                                  1, 1), padding=0)

        max_reponses_2, max_indexes_2 = torch.max(source_filter_response_map.view(keypoint_number, -1),
                                                  dim=1,
                                                  keepdim=False)

        del sampled_feature_vectors_2, source_filter_response_map, feature_1d_locations_2

        keypoint_1d_locations_1 = torch.from_numpy(np.asarray(kps_1D_1)).float().cuda(gpu_id).view(
            keypoint_number, 1)
        keypoint_2d_locations_1 = torch.cat(
            [torch.fmod(keypoint_1d_locations_1, width),
             torch.floor(keypoint_1d_locations_1 / width)],
            dim=1).view(keypoint_number, 2).float()

        detected_source_keypoint_1d_locations = max_indexes_2.float().view(keypoint_number, 1)
        detected_source_keypoint_2d_locations = torch.cat(
            [torch.fmod(detected_source_keypoint_1d_locations, width),
             torch.floor(detected_source_keypoint_1d_locations / width)],
            dim=1).view(keypoint_number, 2).float()

        # We will accept the feature matches if the max indexes here is
        # not far away from the original key point location from descriptor
        cross_check_correspondence_distances = torch.norm(
            keypoint_2d_locations_1 - detected_source_keypoint_2d_locations, dim=1, p=2).view(
            keypoint_number)
        valid_correspondence_indexes = torch.nonzero(cross_check_correspondence_distances < cross_check_distance).view(
            -1)

        if valid_correspondence_indexes.shape[0] == 0:
            return None

        valid_detected_1d_locations_2 = torch.gather(full_detected_target_1d_locations.long().view(-1),
                                                     0, valid_correspondence_indexes.long())
        valid_max_responses = torch.gather(selected_max_responses.view(-1), 0,
                                           valid_correspondence_indexes.long())

        if display_matches:
            valid_detected_1d_locations_2 = valid_detected_1d_locations_2.float()
            valid_detected_target_2d_locations = torch.cat(
                [torch.fmod(valid_detected_1d_locations_2, width).view(-1, 1),
                 torch.floor(valid_detected_1d_locations_2 / width).view(-1, 1)],
                dim=1).view(-1, 2).float()
            valid_source_keypoint_indexes = valid_correspondence_indexes.view(-1).data.cpu().numpy()
            valid_detected_target_2d_locations = valid_detected_target_2d_locations.view(-1, 2).data.cpu().numpy()

            valid_detected_1d_locations_2 = valid_detected_1d_locations_2.long().data.cpu().numpy()
            valid_max_responses = valid_max_responses.data.cpu().numpy()
            valid_correspondence_indexes = valid_correspondence_indexes.data.cpu().numpy()
            detected_keypoints_2 = []
            for index in valid_detected_1d_locations_2:
                detected_keypoints_2.append(
                    cv2.KeyPoint(x=float(np.floor(index % width)), y=float(np.floor(index / width)), _size=1.0))

            matches = []
            for i, (query_index, response) in enumerate(
                    zip(valid_correspondence_indexes, valid_max_responses)):
                matches.append(cv2.DMatch(_queryIdx=query_index, _trainIdx=i, _distance=response))

            # Color image 3 x H x W
            color_1 = np.moveaxis(color_1, source=[0, 1, 2], destination=[2, 0, 1])
            color_2 = np.moveaxis(color_2, source=[0, 1, 2], destination=[2, 0, 1])

            # Extract corner points
            color_1 = cv2.cvtColor(np.uint8(255 * (color_1 * 0.5 + 0.5)), cv2.COLOR_RGB2BGR)
            color_2 = cv2.cvtColor(np.uint8(255 * (color_2 * 0.5 + 0.5)), cv2.COLOR_RGB2BGR)

            display_matches_ai = cv2.drawMatches(color_1, kps_1, color_2, detected_keypoints_2, matches,
                                                 flags=2, outImg=None)
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            feature_matches_craft = bf.knnMatch(des_1, des_2, k=1)
            good = []
            for m in feature_matches_craft:
                if len(m) != 0:
                    good.append(m[0])
            display_matches_craft = cv2.drawMatches(color_1, kps_1, color_2, kps_2, good, flags=2,
                                                    outImg=None)

            return display_matches_ai, display_matches_craft, valid_source_keypoint_indexes, valid_detected_target_2d_locations
        else:
            valid_detected_1d_locations_2 = valid_detected_1d_locations_2.float()
            valid_detected_target_2d_locations = torch.cat(
                [torch.clamp(torch.fmod(valid_detected_1d_locations_2, downsampling * width).view(-1, 1), min=0,
                             max=downsampling * width - 1),
                 torch.clamp(torch.floor(valid_detected_1d_locations_2 / (downsampling * width)).view(-1, 1), min=0,
                             max=downsampling * height - 1)],
                dim=1).view(-1, 2).float()
            valid_source_keypoint_indexes = valid_correspondence_indexes.view(-1).data.cpu().numpy()
            valid_detected_target_2d_locations = valid_detected_target_2d_locations.view(-1, 2).data.cpu().numpy()
            return valid_source_keypoint_indexes, valid_detected_target_2d_locations


def feature_matching_single_generation(color_1, color_2, feature_map_1, feature_map_2,
                                       kps_1D_1, cross_check_distance,
                                       kps_1, display_matches, des_1=None, des_2=None, kps_2=None, gpu_id=0):
    with torch.no_grad():
        # Feature map C x H x W
        feature_length, height, width = feature_map_1.shape

        # Extend 1D locations to B x C x Sampling_size
        keypoint_number = len(kps_1D_1)
        source_feature_1d_locations = torch.from_numpy(kps_1D_1).long().cuda(gpu_id).view(
            1, 1,
            keypoint_number).expand(
            -1, feature_length, -1)

        # Sampled rough locator feature vectors
        sampled_feature_vectors = torch.gather(
            feature_map_1.view(1, feature_length, height * width), 2,
            source_feature_1d_locations.long())
        sampled_feature_vectors = sampled_feature_vectors.view(1, feature_length,
                                                               keypoint_number,
                                                               1,
                                                               1).permute(0, 2, 1, 3,
                                                                          4).view(1,
                                                                                  keypoint_number,
                                                                                  feature_length,
                                                                                  1, 1)

        # 1 x Sampling_size x H x W
        filter_response_map = torch.nn.functional.conv2d(
            input=feature_map_2.view(1, feature_length, height, width),
            weight=sampled_feature_vectors.view(keypoint_number,
                                                feature_length,
                                                1, 1), padding=0)

        max_reponses, max_indexes = torch.max(filter_response_map.view(keypoint_number, -1), dim=1,
                                              keepdim=False)
        del sampled_feature_vectors, filter_response_map, source_feature_1d_locations
        # query is 1 and train is 2 here
        detected_target_1d_locations = max_indexes.view(-1)
        selected_max_responses = max_reponses.view(-1)
        # Do cross check
        feature_1d_locations_2 = detected_target_1d_locations.long().view(
            1, 1, -1).expand(-1, feature_length, -1)

        # Sampled rough locator feature vectors
        sampled_feature_vectors_2 = torch.gather(
            feature_map_2.view(1, feature_length, height * width), 2,
            feature_1d_locations_2.long())
        sampled_feature_vectors_2 = sampled_feature_vectors_2.view(1, feature_length,
                                                                   keypoint_number,
                                                                   1,
                                                                   1).permute(0, 2, 1, 3,
                                                                              4).view(1,
                                                                                      keypoint_number,
                                                                                      feature_length,
                                                                                      1, 1)

        # 1 x Sampling_size x H x W
        source_filter_response_map = torch.nn.functional.conv2d(
            input=feature_map_1.view(1, feature_length, height, width),
            weight=sampled_feature_vectors_2.view(keypoint_number,
                                                  feature_length,
                                                  1, 1), padding=0)

        max_reponses_2, max_indexes_2 = torch.max(source_filter_response_map.view(keypoint_number, -1),
                                                  dim=1,
                                                  keepdim=False)
        del sampled_feature_vectors_2, source_filter_response_map, feature_1d_locations_2

        keypoint_1d_locations_1 = torch.from_numpy(np.asarray(kps_1D_1)).float().cuda(gpu_id).view(
            keypoint_number, 1)
        keypoint_2d_locations_1 = torch.cat(
            [torch.fmod(keypoint_1d_locations_1, width),
             torch.floor(keypoint_1d_locations_1 / width)],
            dim=1).view(keypoint_number, 2).float()

        detected_source_keypoint_1d_locations = max_indexes_2.float().view(keypoint_number, 1)
        detected_source_keypoint_2d_locations = torch.cat(
            [torch.fmod(detected_source_keypoint_1d_locations, width),
             torch.floor(detected_source_keypoint_1d_locations / width)],
            dim=1).view(keypoint_number, 2).float()

        # We will accept the feature matches if the max indexes here is
        # not far away from the original key point location from descriptor
        cross_check_correspondence_distances = torch.norm(
            keypoint_2d_locations_1 - detected_source_keypoint_2d_locations, dim=1, p=2).view(
            keypoint_number)
        valid_correspondence_indexes = torch.nonzero(cross_check_correspondence_distances < cross_check_distance).view(
            -1)

        if valid_correspondence_indexes.shape[0] == 0:
            return None

        valid_detected_1d_locations_2 = torch.gather(detected_target_1d_locations.long().view(-1),
                                                     0, valid_correspondence_indexes.long())
        valid_max_responses = torch.gather(selected_max_responses.view(-1), 0,
                                           valid_correspondence_indexes.long())

        if display_matches:
            valid_detected_1d_locations_2 = valid_detected_1d_locations_2.float()
            valid_detected_target_2d_locations = torch.cat(
                [torch.fmod(valid_detected_1d_locations_2, width).view(-1, 1),
                 torch.floor(valid_detected_1d_locations_2 / width).view(-1, 1)],
                dim=1).view(-1, 2).float()
            valid_source_keypoint_indexes = valid_correspondence_indexes.view(-1).data.cpu().numpy()
            valid_detected_target_2d_locations = valid_detected_target_2d_locations.view(-1, 2).data.cpu().numpy()

            valid_detected_1d_locations_2 = valid_detected_1d_locations_2.long().data.cpu().numpy()
            valid_max_responses = valid_max_responses.data.cpu().numpy()
            valid_correspondence_indexes = valid_correspondence_indexes.data.cpu().numpy()
            detected_keypoints_2 = []
            for index in valid_detected_1d_locations_2:
                detected_keypoints_2.append(
                    cv2.KeyPoint(x=float(np.floor(index % width)), y=float(np.floor(index / width)), _size=1.0))

            matches = []
            for i, (query_index, response) in enumerate(
                    zip(valid_correspondence_indexes, valid_max_responses)):
                matches.append(cv2.DMatch(_queryIdx=query_index, _trainIdx=i, _distance=response))

            # Color image 3 x H x W
            color_1 = np.moveaxis(color_1, source=[0, 1, 2], destination=[2, 0, 1])
            color_2 = np.moveaxis(color_2, source=[0, 1, 2], destination=[2, 0, 1])

            # Extract corner points
            color_1 = cv2.cvtColor(np.uint8(255 * (color_1 * 0.5 + 0.5)), cv2.COLOR_RGB2BGR)
            color_2 = cv2.cvtColor(np.uint8(255 * (color_2 * 0.5 + 0.5)), cv2.COLOR_RGB2BGR)

            display_matches_ai = cv2.drawMatches(color_1, kps_1, color_2, detected_keypoints_2, matches,
                                                 flags=2, outImg=None)
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            feature_matches_craft = bf.knnMatch(des_1, des_2, k=1)
            good = []
            for m in feature_matches_craft:
                if len(m) != 0:
                    good.append(m[0])
            display_matches_craft = cv2.drawMatches(color_1, kps_1, color_2, kps_2, good, flags=2,
                                                    outImg=None)

            return display_matches_ai, display_matches_craft, valid_source_keypoint_indexes, valid_detected_target_2d_locations
        else:
            valid_detected_1d_locations_2 = valid_detected_1d_locations_2.float()
            valid_detected_target_2d_locations = torch.cat(
                [torch.fmod(valid_detected_1d_locations_2, width).view(-1, 1),
                 torch.floor(valid_detected_1d_locations_2 / width).view(-1, 1)],
                dim=1).view(-1, 2).float()
            valid_source_keypoint_indexes = valid_correspondence_indexes.view(-1).data.cpu().numpy()
            valid_detected_target_2d_locations = valid_detected_target_2d_locations.view(-1, 2).data.cpu().numpy()
            return valid_source_keypoint_indexes, valid_detected_target_2d_locations


def feature_matching(color_1, color_2, rough_feature_map_1, rough_feature_map_2, fine_feature_map_1, fine_feature_map_2,
                     boundary, kps_1D_1, des_1, des_2,
                     scale, threshold, cross_check_distance,
                     kps_1, kps_2, gpu_id=0):
    with torch.no_grad():
        color_1 = color_1.data.cpu().numpy()
        color_2 = color_2.data.cpu().numpy()
        # Color image 3 x H x W
        # Feature map C x H x W
        rough_feature_length, height, width = rough_feature_map_1.shape
        fine_feature_length, height, width = fine_feature_map_1.shape

        # Extend 1D locations to B x C x Sampling_size
        keypoint_number = len(kps_1D_1)
        rough_source_feature_1d_locations = torch.from_numpy(kps_1D_1).long().cuda(gpu_id).view(
            1, 1,
            keypoint_number).expand(
            -1, rough_feature_length, -1)

        # Sampled rough locator feature vectors
        sampled_rough_feature_vectors = torch.gather(
            rough_feature_map_1.view(1, rough_feature_length, height * width), 2,
            rough_source_feature_1d_locations.long())
        sampled_rough_feature_vectors = sampled_rough_feature_vectors.view(1, rough_feature_length,
                                                                           keypoint_number,
                                                                           1,
                                                                           1).permute(0, 2, 1, 3,
                                                                                      4).view(1,
                                                                                              keypoint_number,
                                                                                              rough_feature_length,
                                                                                              1, 1)

        rough_filter_response_map = torch.nn.functional.conv2d(
            input=rough_feature_map_2.view(1, rough_feature_length, height, width),
            weight=sampled_rough_feature_vectors.view(keypoint_number,
                                                      rough_feature_length,
                                                      1, 1), padding=0)

        # 1 x Sampling_size x H x W
        rough_filter_response_map = 0.5 * rough_filter_response_map + 0.5
        rough_filter_response_map = torch.exp(scale * (rough_filter_response_map - threshold))
        rough_filter_response_map = rough_filter_response_map / torch.sum(rough_filter_response_map,
                                                                          dim=(2, 3),
                                                                          keepdim=True)
        # Cleaning used variables to save space
        del sampled_rough_feature_vectors
        del rough_source_feature_1d_locations

        # # Sampled texture matcher feature vectors
        fine_source_feature_1d_locations = torch.from_numpy(kps_1D_1).cuda(gpu_id).long().view(1, 1,
                                                                                               keypoint_number).expand(
            -1, fine_feature_length, -1)
        sampled_fine_feature_vectors = torch.gather(
            fine_feature_map_1.view(1, fine_feature_length, height * width), 2,
            fine_source_feature_1d_locations.long())
        sampled_fine_feature_vectors = sampled_fine_feature_vectors.view(1, fine_feature_length,
                                                                         keypoint_number, 1,
                                                                         1).permute(0, 2, 1, 3, 4).view(
            1, keypoint_number,
            fine_feature_length,
            1, 1)
        fine_filter_response_map = torch.nn.functional.conv2d(
            input=fine_feature_map_2.view(1, fine_feature_length, height, width),
            weight=sampled_fine_feature_vectors.view(keypoint_number,
                                                     fine_feature_length,
                                                     1, 1), padding=0)
        # 1 x Sampling_size x H x W
        fine_filter_response_map = 0.5 * fine_filter_response_map + 0.5
        fine_filter_response_map = torch.exp(
            scale * (fine_filter_response_map - threshold)) * boundary.view(1, 1, height, width).expand(
            -1, keypoint_number, -1, -1)
        fine_filter_response_map = fine_filter_response_map / torch.sum(fine_filter_response_map,
                                                                        dim=(2, 3), keepdim=True)

        # Cleaning used variables to save space
        del fine_source_feature_1d_locations
        del sampled_fine_feature_vectors

        merged_response_map = rough_filter_response_map * fine_filter_response_map
        max_reponses, max_indexes = torch.max(merged_response_map.view(keypoint_number, -1), dim=1,
                                              keepdim=False)

        # Cleaning used variables to save space
        del rough_filter_response_map
        del fine_filter_response_map

        # query is 1 and train is 2 here
        detected_target_1d_locations = max_indexes.view(-1)
        selected_max_responses = max_reponses.view(-1)
        # Do cross check
        rough_feature_1d_locations_2 = detected_target_1d_locations.long().view(
            1, 1, -1).expand(-1, rough_feature_length, -1)
        keypoint_number = keypoint_number

        # Sampled rough locator feature vectors
        sampled_rough_feature_vectors_2 = torch.gather(
            rough_feature_map_2.view(1, rough_feature_length, height * width), 2,
            rough_feature_1d_locations_2.long())
        sampled_rough_feature_vectors_2 = sampled_rough_feature_vectors_2.view(1, rough_feature_length,
                                                                               keypoint_number,
                                                                               1,
                                                                               1).permute(0, 2, 1, 3,
                                                                                          4).view(1,
                                                                                                  keypoint_number,
                                                                                                  rough_feature_length,
                                                                                                  1, 1)

        rough_source_filter_response_map = torch.nn.functional.conv2d(
            input=rough_feature_map_1.view(1, rough_feature_length, height, width),
            weight=sampled_rough_feature_vectors_2.view(keypoint_number,
                                                        rough_feature_length,
                                                        1, 1), padding=0)

        # 1 x Sampling_size x H x W
        rough_source_filter_response_map = 0.5 * rough_source_filter_response_map + 0.5
        rough_source_filter_response_map = torch.exp(scale * (rough_source_filter_response_map - threshold))
        rough_source_filter_response_map = rough_source_filter_response_map / torch.sum(
            rough_source_filter_response_map,
            dim=(2, 3),
            keepdim=True)

        del rough_feature_1d_locations_2
        del sampled_rough_feature_vectors_2

        # Sampled texture matcher feature vectors
        fine_source_feature_1d_locations_2 = detected_target_1d_locations.long().cuda(gpu_id).view(
            1, 1, -1).expand(-1, fine_feature_length, -1)
        sampled_fine_feature_vectors_2 = torch.gather(
            fine_feature_map_2.view(1, fine_feature_length, height * width), 2,
            fine_source_feature_1d_locations_2.long())
        sampled_fine_feature_vectors_2 = sampled_fine_feature_vectors_2.view(1, fine_feature_length,
                                                                             keypoint_number, 1,
                                                                             1).permute(0, 2, 1, 3,
                                                                                        4).view(
            1, keypoint_number,
            fine_feature_length,
            1, 1)
        fine_source_filter_response_map = torch.nn.functional.conv2d(
            input=fine_feature_map_1.view(1, fine_feature_length, height, width),
            weight=sampled_fine_feature_vectors_2.view(keypoint_number,
                                                       fine_feature_length,
                                                       1, 1), padding=0)
        # 1 x Sampling_size x H x W
        fine_source_filter_response_map = 0.5 * fine_source_filter_response_map + 0.5
        fine_source_filter_response_map = torch.exp(
            scale * (fine_source_filter_response_map - threshold)) * boundary.view(1, 1, height,
                                                                                   width).expand(
            -1, keypoint_number, -1, -1)
        fine_source_filter_response_map = fine_source_filter_response_map / torch.sum(fine_source_filter_response_map,
                                                                                      dim=(2, 3), keepdim=True)

        del fine_source_feature_1d_locations_2
        del sampled_fine_feature_vectors_2

        source_merged_response_map = rough_source_filter_response_map * fine_source_filter_response_map
        max_reponses_2, max_indexes_2 = torch.max(source_merged_response_map.view(keypoint_number, -1),
                                                  dim=1,
                                                  keepdim=False)

        del rough_source_filter_response_map
        del fine_source_filter_response_map

        keypoint_1d_locations_1 = torch.from_numpy(np.asarray(kps_1D_1)).float().cuda(gpu_id).view(
            keypoint_number, 1)
        keypoint_2d_locations_1 = torch.cat(
            [torch.fmod(keypoint_1d_locations_1, width),
             torch.floor(keypoint_1d_locations_1 / width)],
            dim=1).view(keypoint_number, 2).float()

        detected_source_keypoint_1d_locations = max_indexes_2.float().view(keypoint_number, 1)
        detected_source_keypoint_2d_locations = torch.cat(
            [torch.fmod(detected_source_keypoint_1d_locations, width),
             torch.floor(detected_source_keypoint_1d_locations / width)],
            dim=1).view(keypoint_number, 2).float()

        # We will accept the feature matches if the max indexes here is
        # not far away from the original key point location from descriptor
        cross_check_correspondence_distances = torch.norm(
            keypoint_2d_locations_1 - detected_source_keypoint_2d_locations, dim=1, p=2).view(
            keypoint_number)
        valid_correspondence_indexes = torch.nonzero(cross_check_correspondence_distances < cross_check_distance).view(
            -1)

        if valid_correspondence_indexes.shape[0] == 0:
            return None

        valid_detected_1d_locations_2 = torch.gather(detected_target_1d_locations.long().view(-1),
                                                     0, valid_correspondence_indexes.long())
        valid_max_responses = torch.gather(selected_max_responses.view(-1), 0,
                                           valid_correspondence_indexes.long())

        valid_detected_1d_locations_2 = valid_detected_1d_locations_2.data.cpu().numpy()
        valid_max_responses = valid_max_responses.data.cpu().numpy()
        valid_correspondence_indexes = valid_correspondence_indexes.data.cpu().numpy()

        detected_keypoints_2 = []
        for index in valid_detected_1d_locations_2:
            detected_keypoints_2.append(
                cv2.KeyPoint(x=float(np.floor(index % width)), y=float(np.floor(index / width)), _size=1.0))

        matches = []
        for i, (query_index, response) in enumerate(
                zip(valid_correspondence_indexes, valid_max_responses)):
            matches.append(cv2.DMatch(_queryIdx=query_index, _trainIdx=i, _distance=response))

        color_1 = np.moveaxis(color_1, source=[0, 1, 2], destination=[2, 0, 1])
        color_2 = np.moveaxis(color_2, source=[0, 1, 2], destination=[2, 0, 1])

        # Extract corner points
        color_1 = np.uint8(255 * (color_1 * 0.5 + 0.5))
        color_2 = np.uint8(255 * (color_2 * 0.5 + 0.5))

        display_matches_ai = cv2.drawMatches(color_1, kps_1, color_2, detected_keypoints_2, matches,
                                             flags=2,
                                             outImg=None)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        feature_matches_craft = bf.knnMatch(des_1, des_2, k=1)
        good = []
        for m in feature_matches_craft:
            if len(m) != 0:
                good.append(m[0])
        display_matches_craft = cv2.drawMatches(color_1, kps_1, color_2, kps_2, good, flags=2,
                                                outImg=None)
        return display_matches_ai, display_matches_craft


# def feature_matching_single(color_1, color_2, feature_map_1, feature_map_2, kps_1D_1, des_1, des_2,
#                             scale, threshold, cross_check_distance, kps_1, kps_2, gpu_id=0):
#     with torch.no_grad():
#         color_1 = color_1.data.cpu().numpy()
#         color_2 = color_2.data.cpu().numpy()
#         # Color image 3 x H x W
#         # Feature map C x H x W
#         feature_length, height, width = feature_map_1.shape
#
#         # Extend 1D locations to B x C x Sampling_size
#         keypoint_number = len(kps_1D_1)
#         source_feature_1d_locations = torch.from_numpy(kps_1D_1).long().cuda(gpu_id).view(
#             1, 1,
#             keypoint_number).expand(
#             -1, feature_length, -1)
#
#         # Sampled rough locator feature vectors
#         sampled_feature_vectors = torch.gather(
#             feature_map_1.view(1, feature_length, height * width), 2,
#             source_feature_1d_locations.long())
#         sampled_feature_vectors = sampled_feature_vectors.view(1, feature_length,
#                                                                keypoint_number,
#                                                                1,
#                                                                1).permute(0, 2, 1, 3,
#                                                                           4).view(1,
#                                                                                   keypoint_number,
#                                                                                   feature_length,
#                                                                                   1, 1)
#
#         filter_response_map = torch.nn.functional.conv2d(
#             input=feature_map_2.view(1, feature_length, height, width),
#             weight=sampled_feature_vectors.view(keypoint_number,
#                                                 feature_length,
#                                                 1, 1), padding=0)
#
#         # 1 x Sampling_size x H x W
#         filter_response_map = 0.5 * filter_response_map + 0.5
#         filter_response_map = torch.exp(scale * (filter_response_map - threshold))
#         filter_response_map = filter_response_map / torch.sum(filter_response_map,
#                                                               dim=(2, 3),
#                                                               keepdim=True)
#         # Cleaning used variables to save space
#         del sampled_feature_vectors
#         del source_feature_1d_locations
#
#         max_reponses, max_indexes = torch.max(filter_response_map.view(keypoint_number, -1), dim=1,
#                                               keepdim=False)
#
#         del filter_response_map
#         # query is 1 and train is 2 here
#         detected_target_1d_locations = max_indexes.view(-1)
#         selected_max_responses = max_reponses.view(-1)
#         # Do cross check
#         feature_1d_locations_2 = detected_target_1d_locations.long().view(
#             1, 1, -1).expand(-1, feature_length, -1)
#         keypoint_number = keypoint_number
#
#         # Sampled rough locator feature vectors
#         sampled_feature_vectors_2 = torch.gather(
#             feature_map_2.view(1, feature_length, height * width), 2,
#             feature_1d_locations_2.long())
#         sampled_feature_vectors_2 = sampled_feature_vectors_2.view(1, feature_length,
#                                                                    keypoint_number,
#                                                                    1,
#                                                                    1).permute(0, 2, 1, 3,
#                                                                               4).view(1,
#                                                                                       keypoint_number,
#                                                                                       feature_length,
#                                                                                       1, 1)
#
#         source_filter_response_map = torch.nn.functional.conv2d(
#             input=feature_map_1.view(1, feature_length, height, width),
#             weight=sampled_feature_vectors_2.view(keypoint_number,
#                                                   feature_length,
#                                                   1, 1), padding=0)
#
#         # 1 x Sampling_size x H x W
#         source_filter_response_map = 0.5 * source_filter_response_map + 0.5
#         source_filter_response_map = torch.exp(scale * (source_filter_response_map - threshold))
#         source_filter_response_map = source_filter_response_map / torch.sum(
#             source_filter_response_map,
#             dim=(2, 3),
#             keepdim=True)
#
#         del feature_1d_locations_2
#         del sampled_feature_vectors_2
#
#         max_reponses_2, max_indexes_2 = torch.max(source_filter_response_map.view(keypoint_number, -1),
#                                                   dim=1,
#                                                   keepdim=False)
#
#         del source_filter_response_map
#
#         keypoint_1d_locations_1 = torch.from_numpy(np.asarray(kps_1D_1)).float().cuda(gpu_id).view(
#             keypoint_number, 1)
#         keypoint_2d_locations_1 = torch.cat(
#             [torch.fmod(keypoint_1d_locations_1, width),
#              torch.floor(keypoint_1d_locations_1 / width)],
#             dim=1).view(keypoint_number, 2).float()
#
#         detected_source_keypoint_1d_locations = max_indexes_2.float().view(keypoint_number, 1)
#         detected_source_keypoint_2d_locations = torch.cat(
#             [torch.fmod(detected_source_keypoint_1d_locations, width),
#              torch.floor(detected_source_keypoint_1d_locations / width)],
#             dim=1).view(keypoint_number, 2).float()
#
#         # We will accept the feature matches if the max indexes here is
#         # not far away from the original key point location from descriptor
#         cross_check_correspondence_distances = torch.norm(
#             keypoint_2d_locations_1 - detected_source_keypoint_2d_locations, dim=1, p=2).view(
#             keypoint_number)
#         valid_correspondence_indexes = torch.nonzero(cross_check_correspondence_distances < cross_check_distance).view(
#             -1)
#
#         if valid_correspondence_indexes.shape[0] == 0:
#             return None
#
#         valid_detected_1d_locations_2 = torch.gather(detected_target_1d_locations.long().view(-1),
#                                                      0, valid_correspondence_indexes.long())
#         valid_max_responses = torch.gather(selected_max_responses.view(-1), 0,
#                                            valid_correspondence_indexes.long())
#
#         valid_detected_1d_locations_2 = valid_detected_1d_locations_2.data.cpu().numpy()
#         valid_max_responses = valid_max_responses.data.cpu().numpy()
#         valid_correspondence_indexes = valid_correspondence_indexes.data.cpu().numpy()
#
#         detected_keypoints_2 = []
#         for index in valid_detected_1d_locations_2:
#             detected_keypoints_2.append(
#                 cv2.KeyPoint(x=float(np.floor(index % width)), y=float(np.floor(index / width)), _size=1.0))
#
#         matches = []
#         for i, (query_index, response) in enumerate(
#                 zip(valid_correspondence_indexes, valid_max_responses)):
#             matches.append(cv2.DMatch(_queryIdx=query_index, _trainIdx=i, _distance=response))
#
#         color_1 = np.moveaxis(color_1, source=[0, 1, 2], destination=[2, 0, 1])
#         color_2 = np.moveaxis(color_2, source=[0, 1, 2], destination=[2, 0, 1])
#
#         # Extract corner points
#         color_1 = np.uint8(255 * (color_1 * 0.5 + 0.5))
#         color_2 = np.uint8(255 * (color_2 * 0.5 + 0.5))
#
#         display_matches_ai = cv2.drawMatches(color_1, kps_1, color_2, detected_keypoints_2, matches,
#                                              flags=2,
#                                              outImg=None)
#
#         bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)
#         feature_matches_craft = bf.knnMatch(des_1, des_2, k=1)
#
#         good = []
#         for m in feature_matches_craft:
#             if len(m) != 0:
#                 good.append(m[0])
#         display_matches_craft = cv2.drawMatches(color_1, kps_1, color_2, kps_2, good, flags=2,
#                                                 outImg=None)
#         return display_matches_ai, display_matches_craft


def get_torch_training_data_feature_matching(height, width, pair_projections, pair_indexes, point_cloud,
                                             mask_boundary, view_indexes_per_point, clean_point_list,
                                             visible_view_indexes):
    num_clean_point = int(np.sum(np.asarray(clean_point_list)))
    point_projection_positions_1 = np.zeros((num_clean_point, 2), dtype=np.float32)
    point_projection_positions_2 = np.zeros((num_clean_point, 2), dtype=np.float32)
    # TODO: Vectorize this process
    for i in range(2):
        projection_matrix = pair_projections[i]
        count = 0
        for j in range(len(point_cloud)):
            if clean_point_list[j] < 0.5:
                continue
            point_3d_position = np.asarray(point_cloud[j])
            point_projected_undistorted = np.asarray(projection_matrix).dot(point_3d_position)
            point_projected_undistorted = point_projected_undistorted / point_projected_undistorted[2]

            if np.isnan(point_projected_undistorted[0]) or np.isnan(point_projected_undistorted[1]):
                continue
            # round_u = int(round(point_projected_undistorted[0]))
            # round_v = int(round(point_projected_undistorted[1]))

            if i == 0:
                point_projection_positions_1[count][0] = point_projected_undistorted[0]
                point_projection_positions_1[count][1] = point_projected_undistorted[1]

            elif i == 1:
                point_projection_positions_2[count][0] = point_projected_undistorted[0]
                point_projection_positions_2[count][1] = point_projected_undistorted[1]

            count += 1

    count = 0
    feature_matching = list()
    frame_index_1 = visible_view_indexes.index(pair_indexes[0])
    frame_index_2 = visible_view_indexes.index(pair_indexes[1])
    for i in range(len(point_cloud)):
        if clean_point_list[i] < 0.5:
            continue
        u = point_projection_positions_1[count][0]
        v = point_projection_positions_1[count][1]
        u2 = point_projection_positions_2[count][0]
        v2 = point_projection_positions_2[count][1]
        count += 1
        if 0 <= u < width and 0 <= v < height and 0 <= u2 < width and 0 <= v2 < height:
            if mask_boundary[int(v), int(u)] > 220 and mask_boundary[int(v2), int(u2)] > 220:
                if view_indexes_per_point[i][frame_index_1] > 0.5 and \
                        view_indexes_per_point[i][frame_index_2] > 0.5:
                    feature_matching.append(np.asarray([u, v, u2, v2]))

    return feature_matching


def get_torch_training_data_feature_matching_test(height, width, pair_projections, pair_indexes, point_cloud,
                                                  mask_boundary, view_indexes_per_point, clean_point_list,
                                                  visible_view_indexes):
    num_clean_point = int(np.sum(np.asarray(clean_point_list)))
    point_projection_positions_1 = np.zeros((num_clean_point, 2), dtype=np.float32)
    point_projection_positions_2 = np.zeros((num_clean_point, 2), dtype=np.float32)
    # TODO: Vectorize this process
    for i in range(2):
        projection_matrix = pair_projections[i]
        count = 0
        for j in range(len(point_cloud)):
            if clean_point_list[j] < 0.5:
                continue
            point_3d_position = np.asarray(point_cloud[j])
            point_projected_undistorted = np.asarray(projection_matrix).dot(point_3d_position)
            point_projected_undistorted = point_projected_undistorted / point_projected_undistorted[2]

            if np.isnan(point_projected_undistorted[0]) or np.isnan(point_projected_undistorted[1]):
                continue

            round_u = int(round(point_projected_undistorted[0]))
            round_v = int(round(point_projected_undistorted[1]))

            if i == 0:
                point_projection_positions_1[count][0] = round_u
                point_projection_positions_1[count][1] = round_v

            elif i == 1:
                point_projection_positions_2[count][0] = round_u
                point_projection_positions_2[count][1] = round_v

            count += 1

    count = 0
    feature_matching = list()
    frame_index_1 = pair_indexes[0]
    frame_index_2 = pair_indexes[1]
    for i in range(len(point_cloud)):
        if clean_point_list[i] < 0.5:
            continue
        u = point_projection_positions_1[count][0]
        v = point_projection_positions_1[count][1]
        u2 = point_projection_positions_2[count][0]
        v2 = point_projection_positions_2[count][1]
        count += 1
        if 0 <= u < width and 0 <= v < height and 0 <= u2 < width and 0 <= v2 < height:
            if mask_boundary[int(v), int(u)] > 220 and mask_boundary[int(v2), int(u2)] > 220:
                if view_indexes_per_point[i][frame_index_1] > 0.5 and \
                        view_indexes_per_point[i][frame_index_2] > 0.5:
                    feature_matching.append(np.asarray([u, v, u2, v2]))

    return feature_matching


def init_fn(worker_id):
    np.random.seed(10086 + worker_id)


def init_net(net, type="kaiming", mode="fan_in", activation_mode="relu", distribution="normal", gpu_id=0):
    assert (torch.cuda.is_available())
    net = net.cuda(gpu_id)
    if type == "glorot":
        glorot_weight_zero_bias(net, distribution=distribution)
    else:
        kaiming_weight_zero_bias(net, mode=mode, activation_mode=activation_mode, distribution=distribution)
    return net


def glorot_weight_zero_bias(model, distribution="uniform"):
    """
    Initalize parameters of all modules
    by initializing weights with glorot  uniform/xavier initialization,
    and setting biases to zero.
    Weights from batch norm layers are set to 1.

    Parameters
    ----------
    model: Module
    distribution: string
    """
    for module in model.modules():
        if hasattr(module, 'weight'):
            if not ('BatchNorm' in module.__class__.__name__):
                if distribution == "uniform":
                    torch.nn.init.xavier_uniform_(module.weight, gain=1)
                else:
                    torch.nn.init.xavier_normal_(module.weight, gain=1)
            else:
                torch.nn.init.constant_(module.weight, 1)
        if hasattr(module, 'bias'):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)


def kaiming_weight_zero_bias(model, mode="fan_in", activation_mode="relu", distribution="uniform"):
    if activation_mode == "leaky_relu":
        print("Leaky relu is not supported yet")
        assert False

    for module in model.modules():
        if hasattr(module, 'weight'):
            if not ('BatchNorm' in module.__class__.__name__):
                if distribution == "uniform":
                    torch.nn.init.kaiming_uniform_(module.weight, mode=mode, nonlinearity=activation_mode)
                else:
                    torch.nn.init.kaiming_normal_(module.weight, mode=mode, nonlinearity=activation_mode)
            else:
                torch.nn.init.constant_(module.weight, 1)
        if hasattr(module, 'bias'):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)


def save_model(model, optimizer, epoch, step, model_path, validation_loss):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'validation': validation_loss
    }, str(model_path))
    return


# def save_model(model, optimizer, epoch, step, model_path, failure_sequences, validation_loss):
#     try:
#         torch.save({
#             'model': model.state_dict(),
#             'optimizer': optimizer.state_dict(),
#             'epoch': epoch,
#             'step': step,
#             'failure': failure_sequences,
#             'validation': validation_loss
#         }, str(model_path))
#     except IOError:
#         torch.save({
#             'model': model.state_dict(),
#             'optimizer': optimizer.state_dict(),
#             'epoch': epoch,
#             'step': step,
#             'validation': validation_loss
#         }, str(model_path))
#
#     return


def visualize_color_image(title, images, rebias=False, is_hsv=False, idx=None):
    if idx is None:
        for i in range(images.shape[0]):
            image = images.data.cpu().numpy()[i]
            image = np.moveaxis(image, source=[0, 1, 2], destination=[2, 0, 1])
            if rebias:
                image = image * 0.5 + 0.5
            if is_hsv:
                image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR_FULL)
            cv2.imshow(title + "_" + str(i), image)
    else:
        for id in idx:
            image = images.data.cpu().numpy()[id]
            image = np.moveaxis(image, source=[0, 1, 2], destination=[2, 0, 1])
            if rebias:
                image = image * 0.5 + 0.5
            if is_hsv:
                image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR_FULL)
            cv2.imshow(title + "_" + str(id), image)


def visualize_depth_map(title, depth_maps, min_value_=None, max_value_=None, idx=None, color_mode=cv2.COLORMAP_HOT):
    min_value_list = []
    max_value_list = []
    if idx is None:
        for i in range(depth_maps.shape[0]):
            depth_map_cpu = depth_maps[i].data.cpu().numpy()

            if min_value_ is None and max_value_ is None:
                min_value = np.min(depth_map_cpu)
                max_value = np.max(depth_map_cpu)
                min_value_list.append(min_value)
                max_value_list.append(max_value)
            else:
                min_value = min_value_[i]
                max_value = max_value_[i]

            depth_map_cpu = np.moveaxis(depth_map_cpu, source=[0, 1, 2], destination=[2, 0, 1])
            depth_map_visualize = np.abs((depth_map_cpu - min_value) / (max_value - min_value) * 255)
            depth_map_visualize[depth_map_visualize > 255] = 255
            depth_map_visualize[depth_map_visualize <= 0.0] = 0
            depth_map_visualize = cv2.applyColorMap(np.uint8(depth_map_visualize), color_mode)
            cv2.imshow(title + "_" + str(i), depth_map_visualize)
        return min_value_list, max_value_list
    else:
        for id in idx:
            depth_map_cpu = depth_maps[id].data.cpu().numpy()

            if min_value_ is None and max_value_ is None:
                min_value = np.min(depth_map_cpu)
                max_value = np.max(depth_map_cpu)
                min_value_list.append(min_value)
                max_value_list.append(max_value)
            else:
                min_value = min_value_[id]
                max_value = max_value_[id]

            depth_map_cpu = np.moveaxis(depth_map_cpu, source=[0, 1, 2], destination=[2, 0, 1])
            depth_map_visualize = np.abs((depth_map_cpu - min_value) / (max_value - min_value) * 255)
            depth_map_visualize[depth_map_visualize > 255] = 255
            depth_map_visualize[depth_map_visualize <= 0.0] = 0
            depth_map_visualize = cv2.applyColorMap(np.uint8(depth_map_visualize), color_mode)
            cv2.imshow(title + "_" + str(id), depth_map_visualize)
        return min_value_list, max_value_list


def display_depth_map(depth_map, min_value=None, max_value=None, colormode=cv2.COLORMAP_JET, scale=None):
    if (min_value is None or max_value is None) and scale is None:
        if len(depth_map[depth_map > 0]) > 0:
            min_value = np.min(depth_map[depth_map > 0])
        else:
            min_value = 0.0
    elif scale is not None:
        min_value = 0.0
        max_value = scale
    else:
        pass

    depth_map_visualize = np.abs((depth_map - min_value) / (max_value - min_value) * 255)
    depth_map_visualize[depth_map_visualize > 255] = 255
    depth_map_visualize[depth_map_visualize <= 0.0] = 0
    depth_map_visualize = cv2.applyColorMap(np.uint8(depth_map_visualize), colormode)
    return depth_map_visualize


def draw_hsv(flows, title, idx=None):
    if idx is None:
        flows_cpu = flows.data.cpu().numpy()
        for i in range(flows_cpu.shape[0]):
            flow = np.moveaxis(flows_cpu[i], [0, 1, 2], [2, 0, 1])
            h, w = flow.shape[:2]
            fx, fy = flow[:, :, 0] * w, flow[:, :, 1] * h
            ang = np.arctan2(fy, fx) + np.pi
            v = np.sqrt(fx * fx + fy * fy)
            hsv = np.zeros((h, w, 3), np.uint8)
            hsv[..., 0] = ang * (180 / np.pi / 2)
            hsv[..., 1] = 255
            hsv[..., 2] = np.uint8(
                np.minimum(v, np.sqrt(0.01 * w * w + 0.01 * h * h)) / np.sqrt(0.01 * w * w + 0.01 * h * h) * 255)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imshow(title + str(i), bgr)
    else:
        flows_cpu = flows.data.cpu().numpy()
        for id in idx:
            flow = np.moveaxis(flows_cpu[id], [0, 1, 2], [2, 0, 1])
            h, w = flow.shape[:2]
            fx, fy = flow[:, :, 0] * w, flow[:, :, 1] * h
            ang = np.arctan2(fy, fx) + np.pi
            v = np.sqrt(fx * fx + fy * fy)
            hsv = np.zeros((h, w, 3), np.uint8)
            hsv[..., 0] = ang * (180 / np.pi / 2)
            hsv[..., 1] = 255
            hsv[..., 2] = np.uint8(
                np.minimum(v, np.sqrt(0.01 * w * w + 0.01 * h * h)) / np.sqrt(0.01 * w * w + 0.01 * h * h) * 255)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imshow(title + str(id), bgr)


def write_event(log, step, **data):
    data['step'] = step
    data['dt'] = datetime.time().isoformat()
    log.write(unicode(json.dumps(data, sort_keys=True)))
    log.write(unicode('\n'))
    log.flush()


def point_cloud_from_depth(depth_map, color_img, mask_img, intrinsic_matrix, point_cloud_downsampling,
                           min_threshold=None, max_threshold=None):
    point_clouds = []
    height, width, channel = color_img.shape

    f_x = intrinsic_matrix[0, 0]
    c_x = intrinsic_matrix[0, 2]
    f_y = intrinsic_matrix[1, 1]
    c_y = intrinsic_matrix[1, 2]

    for h in range(height):
        for w in range(width):
            if h % point_cloud_downsampling == 0 and w % point_cloud_downsampling == 0 and mask_img[h, w] > 0.5:
                z = depth_map[h, w]
                x = (w - c_x) / f_x * z
                y = (h - c_y) / f_y * z
                r = color_img[h, w, 2]
                g = color_img[h, w, 1]
                b = color_img[h, w, 0]
                if max_threshold is not None and min_threshold is not None:
                    if np.max([r, g, b]) >= max_threshold and np.min([r, g, b]) <= min_threshold:
                        point_clouds.append((x, y, z, np.uint8(r), np.uint8(g), np.uint8(b)))
                else:
                    point_clouds.append((x, y, z, np.uint8(r), np.uint8(g), np.uint8(b)))

    point_clouds = np.array(point_clouds, dtype='float32')
    point_clouds = np.reshape(point_clouds, (-1, 6))
    return point_clouds


def write_point_cloud(path, point_cloud):
    point_clouds_list = []
    for i in range(point_cloud.shape[0]):
        point_clouds_list.append((point_cloud[i, 0], point_cloud[i, 1], point_cloud[i, 2], point_cloud[i, 3],
                                  point_cloud[i, 4], point_cloud[i, 5]))

    vertex = np.array(point_clouds_list,
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], text=True).write(path)
    return


# def display_colors(idx, step, writer, colors_1, phase="Training"):
#     colors_display = vutils.make_grid(0.5 * colors_1 + 0.5, normalize=False)
#     colors_display_hsv = np.moveaxis(colors_display.data.cpu().numpy(),
#                                      source=[0, 1, 2], destination=[2, 0, 1])
#     # colors_display_hsv[colors_display_hsv < 0.0] = 0.0
#     # colors_display_hsv[colors_display_hsv > 1.0] = 1.0
#     colors_display_hsv = cv2.cvtColor(np.uint8(255 * colors_display_hsv), cv2.COLOR_HSV2RGB_FULL)
#     writer.add_image(phase + '/Images/Color_' + str(idx),
#                      np.moveaxis(colors_display_hsv, source=[0, 1, 2], destination=[1, 2, 0]), step)
#     return colors_display_hsv


def display_training_output(idx, step, writer, colors_1, scaled_mean_depth_maps_1, scaled_std_depth_maps_1,
                            phase="Training"):
    colors_display = vutils.make_grid(colors_1 * 0.5 + 0.5, normalize=False)
    colors_display_hsv = np.moveaxis(colors_display.data.cpu().numpy(),
                                     source=[0, 1, 2], destination=[2, 0, 1])
    colors_display_hsv[colors_display_hsv < 0.0] = 0.0
    colors_display_hsv[colors_display_hsv > 1.0] = 1.0
    colors_display_hsv = cv2.cvtColor(colors_display_hsv, cv2.COLOR_HSV2RGB_FULL)
    writer.add_image(phase + '/Images/Color_' + str(idx),
                     np.moveaxis(colors_display_hsv, source=[0, 1, 2], destination=[1, 2, 0]), step)

    mean_depths_display = vutils.make_grid(scaled_mean_depth_maps_1, normalize=True, scale_each=True)
    mean_depths_display_hsv = cv2.applyColorMap(np.uint8(255 * np.moveaxis(mean_depths_display.data.cpu().numpy(),
                                                                           source=[0, 1, 2],
                                                                           destination=[2, 0, 1])),
                                                cv2.COLORMAP_HOT)
    mean_depths_display_hsv = cv2.cvtColor(mean_depths_display_hsv, cv2.COLOR_BGR2RGB)
    writer.add_image(phase + '/Images/Mean_Depth_' + str(idx),
                     np.moveaxis(mean_depths_display_hsv, source=[0, 1, 2], destination=[1, 2, 0]), step)

    std_depths_display = vutils.make_grid(scaled_std_depth_maps_1, normalize=True, scale_each=True)
    std_depths_display_hsv = cv2.applyColorMap(np.uint8(255 * np.moveaxis(std_depths_display.data.cpu().numpy(),
                                                                          source=[0, 1, 2], destination=[2, 0, 1])),
                                               cv2.COLORMAP_HOT)
    std_depths_display_hsv = cv2.cvtColor(std_depths_display_hsv, cv2.COLOR_BGR2RGB)
    writer.add_image(phase + '/Images/Std_Depth_' + str(idx),
                     np.moveaxis(std_depths_display_hsv, source=[0, 1, 2], destination=[1, 2, 0]), step)

    return colors_display_hsv, mean_depths_display_hsv, std_depths_display_hsv


def display_simulated_depth_map(idx, step, writer, simulated_depth_map_1):
    simulated_depths_display = vutils.make_grid(simulated_depth_map_1, normalize=True, scale_each=True)
    simulated_depths_display = cv2.applyColorMap(
        np.uint8(255 * np.moveaxis(simulated_depths_display.data.cpu().numpy(),
                                   source=[0, 1, 2], destination=[2, 0, 1])),
        cv2.COLORMAP_HOT)
    simulated_depths_display = cv2.cvtColor(simulated_depths_display, cv2.COLOR_BGR2RGB)
    writer.add_image('Training/Images/Simulated_Depth_' + str(idx),
                     np.moveaxis(simulated_depths_display, source=[0, 1, 2], destination=[1, 2, 0]), step)

    return simulated_depths_display


def display_testing_output(idx, step, writer, colors_1, scaled_mean_depth_maps_1, scaled_std_depth_maps_1):
    colors_display = vutils.make_grid(colors_1 * 0.5 + 0.5, normalize=False)
    colors_display_hsv = np.moveaxis(colors_display.data.cpu().numpy(),
                                     source=[0, 1, 2], destination=[2, 0, 1])
    colors_display_hsv[colors_display_hsv < 0.0] = 0.0
    colors_display_hsv[colors_display_hsv > 1.0] = 1.0
    colors_display_hsv = cv2.cvtColor(colors_display_hsv, cv2.COLOR_HSV2RGB_FULL)
    writer.add_image('Test/Images/Color_' + str(idx),
                     np.moveaxis(colors_display_hsv, source=[0, 1, 2], destination=[1, 2, 0]), step)

    mean_depths_display = vutils.make_grid(scaled_mean_depth_maps_1, normalize=True, scale_each=True)
    mean_depths_display_hsv = cv2.applyColorMap(np.uint8(255 * np.moveaxis(mean_depths_display.data.cpu().numpy(),
                                                                           source=[0, 1, 2],
                                                                           destination=[2, 0, 1])),
                                                cv2.COLORMAP_HOT)
    mean_depths_display_hsv = cv2.cvtColor(mean_depths_display_hsv, cv2.COLOR_BGR2RGB)
    writer.add_image('Test/Images/Mean_Depth_' + str(idx),
                     np.moveaxis(mean_depths_display_hsv, source=[0, 1, 2], destination=[1, 2, 0]), step)

    std_depths_display = vutils.make_grid(scaled_std_depth_maps_1, normalize=True, scale_each=True)
    std_depths_display_hsv = cv2.applyColorMap(np.uint8(255 * np.moveaxis(std_depths_display.data.cpu().numpy(),
                                                                          source=[0, 1, 2], destination=[2, 0, 1])),
                                               cv2.COLORMAP_HOT)
    std_depths_display_hsv = cv2.cvtColor(std_depths_display_hsv, cv2.COLOR_BGR2RGB)
    writer.add_image('Test/Images/Std_Depth_' + str(idx),
                     np.moveaxis(std_depths_display_hsv, source=[0, 1, 2], destination=[1, 2, 0]), step)

    return colors_display_hsv, mean_depths_display_hsv, std_depths_display_hsv


def display_feature_matching_map(idx, step, title, writer, feature_matching_heat_map, phase="Training"):
    batch_size, _, height, width = feature_matching_heat_map.shape
    feature_matching_heat_map = feature_matching_heat_map.view(batch_size, 1, height, width)
    heatmap_display = vutils.make_grid(feature_matching_heat_map, normalize=False, scale_each=True)
    heatmap_display = cv2.applyColorMap(np.uint8(255 * np.moveaxis(heatmap_display.data.cpu().numpy(),
                                                                   source=[0, 1, 2], destination=[2, 0, 1])),
                                        cv2.COLORMAP_HOT)
    heatmap_display = cv2.cvtColor(heatmap_display, cv2.COLOR_BGR2RGB)
    writer.add_image(phase + '/Images/' + title + str(idx),
                     np.moveaxis(heatmap_display,
                                 source=[0, 1, 2], destination=[1, 2, 0]), step)


def display_depth_goal(idx, step, writer, goal_depth_map_1, phase="Training"):
    depths_display = vutils.make_grid(goal_depth_map_1, normalize=True, scale_each=True)
    depths_display = cv2.applyColorMap(np.uint8(255 * np.moveaxis(depths_display.data.cpu().numpy(),
                                                                  source=[0, 1, 2], destination=[2, 0, 1])),
                                       cv2.COLORMAP_HOT)
    depths_display = cv2.cvtColor(depths_display, cv2.COLOR_BGR2RGB)
    writer.add_image(phase + '/Images/Goal_Depth_' + str(idx),
                     np.moveaxis(depths_display, source=[0, 1, 2], destination=[1, 2, 0]), step)
    return depths_display


def display_network_weights(depth_estimation_model_student, writer, step):
    for name, param in depth_estimation_model_student.named_parameters():
        writer.add_histogram("Weights/" + name, param.clone().cpu().data.numpy(), step)


def generate_training_output(colors_1, scaled_depth_maps_1, boundaries, intrinsic_matrices, is_hsv, epoch,
                             results_root):
    color_inputs_cpu = colors_1.data.cpu().numpy()
    pred_depths_cpu = scaled_depth_maps_1.data.cpu().numpy()
    boundaries_cpu = boundaries.data.cpu().numpy()
    intrinsics_cpu = intrinsic_matrices.data.cpu().numpy()
    color_imgs = []
    pred_depth_imgs = []

    for j in range(colors_1.shape[0]):
        color_img = color_inputs_cpu[j]
        pred_depth_img = pred_depths_cpu[j]

        color_img = np.moveaxis(color_img, source=[0, 1, 2], destination=[2, 0, 1])
        color_img = color_img * 0.5 + 0.5
        color_img[color_img < 0.0] = 0.0
        color_img[color_img > 1.0] = 1.0
        color_img = np.uint8(255 * color_img)
        if is_hsv:
            color_img = cv2.cvtColor(color_img, cv2.COLOR_HSV2BGR_FULL)

        pred_depth_img = np.moveaxis(pred_depth_img, source=[0, 1, 2], destination=[2, 0, 1])

        if j == 0:
            # Write point cloud
            boundary = boundaries_cpu[j]
            intrinsic = intrinsics_cpu[j]
            boundary = np.moveaxis(boundary, source=[0, 1, 2], destination=[2, 0, 1])
            point_cloud = point_cloud_from_depth(pred_depth_img, color_img, boundary,
                                                 intrinsic,
                                                 point_cloud_downsampling=1)
            write_point_cloud(
                str(results_root / "point_cloud_epoch_{epoch}_index_{index}.ply".format(epoch=epoch,
                                                                                        index=j)),
                point_cloud)

        color_img = cv2.resize(color_img, dsize=(300, 300))
        pred_depth_img = cv2.resize(pred_depth_img, dsize=(300, 300))
        color_imgs.append(color_img)

        if j == 0:
            histr = cv2.calcHist([pred_depth_img], [0], None, histSize=[100], ranges=[0, 1000])
            plt.plot(histr, color='b')
            plt.xlim([0, 40])
            plt.savefig(
                str(results_root / 'generated_depth_hist_{epoch}.jpg'.format(epoch=epoch)))
            plt.clf()
        display_depth_img = display_depth_map(pred_depth_img)
        pred_depth_imgs.append(display_depth_img)

    final_color = color_imgs[0]
    final_pred_depth = pred_depth_imgs[0]
    for j in range(colors_1.shape[0] - 1):
        final_color = cv2.hconcat((final_color, color_imgs[j + 1]))
        final_pred_depth = cv2.hconcat((final_pred_depth, pred_depth_imgs[j + 1]))

    final = cv2.vconcat((final_color, final_pred_depth))
    cv2.imwrite(str(results_root / 'generated_mask_{epoch}.jpg'.format(epoch=epoch)),
                final)


def generate_validation_output(idx, step, writer, colors_1, scaled_depth_maps_1, boundaries, intrinsic_matrices,
                               is_hsv,
                               results_root, which_bag):
    colors_display = vutils.make_grid(colors_1 * 0.5 + 0.5, normalize=False)
    colors_display_hsv = np.moveaxis(colors_display.data.cpu().numpy(),
                                     source=[0, 1, 2], destination=[2, 0, 1])
    colors_display_hsv[colors_display_hsv < 0.0] = 0.0
    colors_display_hsv[colors_display_hsv > 1.0] = 1.0
    colors_display_hsv = cv2.cvtColor(colors_display_hsv, cv2.COLOR_HSV2RGB_FULL)
    writer.add_image('Validation/Images/Color_' + str(idx),
                     np.moveaxis(colors_display_hsv, source=[0, 1, 2], destination=[1, 2, 0]), step)

    depths_display = vutils.make_grid(scaled_depth_maps_1, normalize=True, scale_each=True)
    depths_display_hsv = cv2.applyColorMap(np.uint8(255 * np.moveaxis(depths_display.data.cpu().numpy(),
                                                                      source=[0, 1, 2], destination=[2, 0, 1])),
                                           cv2.COLORMAP_HOT)
    depths_display_hsv = cv2.cvtColor(depths_display_hsv, cv2.COLOR_BGR2RGB)
    writer.add_image('Validation/Images/Depth_' + str(idx),
                     np.moveaxis(depths_display_hsv, source=[0, 1, 2], destination=[1, 2, 0]), step)

    color_inputs_cpu = colors_1.data.cpu().numpy()
    pred_depths_cpu = scaled_depth_maps_1.data.cpu().numpy()
    boundaries_cpu = boundaries.data.cpu().numpy()
    intrinsics_cpu = intrinsic_matrices.data.cpu().numpy()
    color_imgs = []
    pred_depth_imgs = []

    for j in range(colors_1.shape[0]):
        color_img = color_inputs_cpu[j]
        pred_depth_img = pred_depths_cpu[j]

        color_img = np.moveaxis(color_img, source=[0, 1, 2], destination=[2, 0, 1])
        color_img = color_img * 0.5 + 0.5
        color_img[color_img < 0.0] = 0.0
        color_img[color_img > 1.0] = 1.0
        color_img = np.uint8(255 * color_img)
        if is_hsv:
            color_img = cv2.cvtColor(color_img, cv2.COLOR_HSV2BGR_FULL)

        pred_depth_img = np.moveaxis(pred_depth_img, source=[0, 1, 2], destination=[2, 0, 1])

        if j == 0:
            # Write point cloud
            boundary = boundaries_cpu[j]
            intrinsic = intrinsics_cpu[j]
            boundary = np.moveaxis(boundary, source=[0, 1, 2], destination=[2, 0, 1])
            point_cloud = point_cloud_from_depth(pred_depth_img, color_img, boundary,
                                                 intrinsic,
                                                 point_cloud_downsampling=1)
            write_point_cloud(
                str(results_root / "point_cloud_step_{step}_index_{index}_bag_{bag}.ply".format(step=step,
                                                                                                index=j,
                                                                                                bag=which_bag)),
                point_cloud)

        color_imgs.append(color_img)
        display_depth_img = display_depth_map(pred_depth_img)
        pred_depth_imgs.append(display_depth_img)

    final_color = color_imgs[0]
    final_pred_depth = pred_depth_imgs[0]
    for j in range(colors_1.shape[0] - 1):
        final_color = cv2.hconcat((final_color, color_imgs[j + 1]))
        final_pred_depth = cv2.hconcat((final_pred_depth, pred_depth_imgs[j + 1]))

    final = cv2.vconcat((final_color, final_pred_depth))
    cv2.imwrite(str(results_root / 'generated_mask_step_{step}_bag_{bag}.jpg'.format(step=step, bag=which_bag)),
                final)
    return


def generate_test_output(idx, step, writer, colors_1, scaled_depth_maps_1, boundaries, intrinsic_matrices, is_hsv,
                         results_root, which_bag, color_mode=cv2.COLORMAP_HOT):
    colors_display = vutils.make_grid(colors_1 * 0.5 + 0.5, normalize=False)
    colors_display_hsv = np.moveaxis(colors_display.data.cpu().numpy(),
                                     source=[0, 1, 2], destination=[2, 0, 1])
    colors_display_hsv[colors_display_hsv < 0.0] = 0.0
    colors_display_hsv[colors_display_hsv > 1.0] = 1.0
    colors_display_hsv = cv2.cvtColor(colors_display_hsv, cv2.COLOR_HSV2RGB_FULL)
    writer.add_image('Test/Images/Color_' + str(idx),
                     np.moveaxis(colors_display_hsv, source=[0, 1, 2], destination=[1, 2, 0]), step)

    depths_display = vutils.make_grid(scaled_depth_maps_1, normalize=True, scale_each=True)
    depths_display_hsv = cv2.applyColorMap(np.uint8(255 * np.moveaxis(depths_display.data.cpu().numpy(),
                                                                      source=[0, 1, 2], destination=[2, 0, 1])),
                                           colormap=color_mode)
    depths_display_hsv = cv2.cvtColor(depths_display_hsv, cv2.COLOR_BGR2RGB)
    writer.add_image('Test/Images/Depth_' + str(idx),
                     np.moveaxis(depths_display_hsv, source=[0, 1, 2], destination=[1, 2, 0]), step)

    color_inputs_cpu = colors_1.data.cpu().numpy()
    pred_depths_cpu = scaled_depth_maps_1.data.cpu().numpy()
    boundaries_cpu = boundaries.data.cpu().numpy()
    intrinsics_cpu = intrinsic_matrices.data.cpu().numpy()
    color_imgs = []
    pred_depth_imgs = []

    for j in range(colors_1.shape[0]):
        color_img = color_inputs_cpu[j]
        pred_depth_img = pred_depths_cpu[j]

        color_img = np.moveaxis(color_img, source=[0, 1, 2], destination=[2, 0, 1])
        color_img = color_img * 0.5 + 0.5
        color_img[color_img < 0.0] = 0.0
        color_img[color_img > 1.0] = 1.0
        color_img = np.uint8(255 * color_img)
        if is_hsv:
            color_img = cv2.cvtColor(color_img, cv2.COLOR_HSV2BGR_FULL)

        pred_depth_img = np.moveaxis(pred_depth_img, source=[0, 1, 2], destination=[2, 0, 1])

        if j == 0:
            # Write point cloud
            boundary = boundaries_cpu[j]
            intrinsic = intrinsics_cpu[j]
            boundary = np.moveaxis(boundary, source=[0, 1, 2], destination=[2, 0, 1])
            point_cloud = point_cloud_from_depth(pred_depth_img, color_img, boundary,
                                                 intrinsic,
                                                 point_cloud_downsampling=1)
            write_point_cloud(
                str(results_root / "test_point_cloud_step_{step}_bag_{bag}.ply".format(step=step, bag=which_bag)),
                point_cloud)

        color_imgs.append(color_img)
        display_depth_img = display_depth_map(pred_depth_img, colormode=color_mode)
        pred_depth_imgs.append(display_depth_img)

    final_color = color_imgs[0]
    final_pred_depth = pred_depth_imgs[0]
    for j in range(colors_1.shape[0] - 1):
        final_color = cv2.hconcat((final_color, color_imgs[j + 1]))
        final_pred_depth = cv2.hconcat((final_pred_depth, pred_depth_imgs[j + 1]))

    final = cv2.vconcat((final_color, final_pred_depth))
    cv2.imwrite(str(results_root / 'generated_mask_step_{step}_bag_{bag}.jpg'.format(step=step, bag=which_bag)),
                final)
    return


def point_cloud_from_depth_and_initial_pose(depth_map, color_img, mask_img, intrinsic_matrix, translation, rotation,
                                            point_cloud_downsampling,
                                            min_threshold=None, max_threshold=None):
    point_clouds = []
    height, width, channel = color_img.shape

    f_x = intrinsic_matrix[0, 0]
    c_x = intrinsic_matrix[0, 2]
    f_y = intrinsic_matrix[1, 1]
    c_y = intrinsic_matrix[1, 2]

    z_min = -1
    z_max = -1

    for h in range(height):
        for w in range(width):
            if h % point_cloud_downsampling == 0 and w % point_cloud_downsampling == 0 and mask_img[h, w] > 0.5:
                z = depth_map[h, w]
                if z_min == -1:
                    z_min = z
                    z_max = z
                else:
                    z_min = min(z, z_min)
                    z_max = max(z, z_max)

    scale = 20.0 / (z_max - z_min)

    for h in range(height):
        for w in range(width):
            if h % point_cloud_downsampling == 0 and w % point_cloud_downsampling == 0 and mask_img[h, w] > 0.5:
                z = depth_map[h, w]
                x = (w - c_x) / f_x * z
                y = (h - c_y) / f_y * z
                position = np.array([x * scale, y * scale, z * scale], dtype=np.float32).reshape((3, 1))
                transformed_position = np.matmul(rotation, position) + translation.reshape((3, 1))

                r = color_img[h, w, 2]
                g = color_img[h, w, 1]
                b = color_img[h, w, 0]
                if max_threshold is not None and min_threshold is not None:
                    if np.max([r, g, b]) >= max_threshold and np.min([r, g, b]) <= min_threshold:
                        point_clouds.append(
                            (transformed_position[0], transformed_position[1], transformed_position[2],
                             np.uint8(r), np.uint8(g), np.uint8(b)))
                else:
                    point_clouds.append((transformed_position[0], transformed_position[1], transformed_position[2],
                                         np.uint8(r), np.uint8(g), np.uint8(b)))

    point_clouds = np.array(point_clouds, dtype='float32')
    point_clouds = np.reshape(point_clouds, (-1, 6))
    return point_clouds


def read_pose_messages_from_tracker(file_path):
    translation_array = []
    rotation_array = []
    with open(file_path, "r") as filestream:
        for count, line in enumerate(filestream):
            # Skip the header
            if count == 0:
                continue
            array = line.split(",")
            array = array[5:]
            array = np.array(array, dtype=np.float64)
            translation_array.append(array[:3])
            qx, qy, qz, qw = array[3:]
            rotation_matrix = quaternion_matrix([qw, qx, qy, qz])
            rotation_array.append(rotation_matrix[:3, :3])
    return translation_array, rotation_array


def write_test_output_with_initial_pose(results_root, colors_1, scaled_depth_maps_1, boundaries, intrinsic_matrices,
                                        is_hsv,
                                        image_indexes,
                                        translation_dict, rotation_dict, color_mode=cv2.COLORMAP_HOT):
    color_inputs_cpu = colors_1.data.cpu().numpy()
    pred_depths_cpu = (boundaries * scaled_depth_maps_1).data.cpu().numpy()
    boundaries_cpu = boundaries.data.cpu().numpy()
    intrinsics_cpu = intrinsic_matrices.data.cpu().numpy()

    for j in range(colors_1.shape[0]):
        print("processing {}...".format(image_indexes[j]))
        color_img = color_inputs_cpu[j]
        pred_depth_img = pred_depths_cpu[j]

        color_img = np.moveaxis(color_img, source=[0, 1, 2], destination=[2, 0, 1])
        color_img = color_img * 0.5 + 0.5
        color_img[color_img < 0.0] = 0.0
        color_img[color_img > 1.0] = 1.0
        color_img = np.uint8(255 * color_img)
        if is_hsv:
            color_img = cv2.cvtColor(color_img, cv2.COLOR_HSV2BGR_FULL)

        pred_depth_img = np.moveaxis(pred_depth_img, source=[0, 1, 2], destination=[2, 0, 1])

        # Write point cloud
        boundary = boundaries_cpu[j]
        intrinsic = intrinsics_cpu[j]
        boundary = np.moveaxis(boundary, source=[0, 1, 2], destination=[2, 0, 1])
        point_cloud = point_cloud_from_depth_and_initial_pose(pred_depth_img, color_img, boundary, intrinsic,
                                                              translation=translation_dict[image_indexes[j]],
                                                              rotation=rotation_dict[image_indexes[j]],
                                                              point_cloud_downsampling=1,
                                                              min_threshold=None, max_threshold=None)

        write_point_cloud(str(results_root / ("test_point_cloud_" + image_indexes[j] + ".ply")), point_cloud)
        cv2.imwrite(str(results_root / ("test_color_" + image_indexes[j] + ".jpg")), color_img)
        display_depth_img = display_depth_map(pred_depth_img, colormode=color_mode)
        cv2.imwrite(str(results_root / ("test_depth_" + image_indexes[j] + ".jpg")), display_depth_img)

    return


def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> numpy.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> numpy.allclose(M, numpy.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> numpy.allclose(M, numpy.diag([1, -1, -1, 1]))
    True

    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < np.finfo(float).eps * 4.0:
        return np.identity(4)
    q *= np.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
        [0.0, 0.0, 0.0, 1.0]])


def read_initial_pose_file(file_path):
    frame_index_array = []
    translation_dict = {}
    rotation_dict = {}

    with open(file_path, "r") as filestream:
        for line in filestream:
            array = line.split(", ")
            array = np.array(array, dtype=np.float64)
            frame_index_array.append(int(array[0]))
            translation_dict["{:08d}".format(int(array[0]))] = array[1:4]
            rotation_matrix = quaternion_matrix(array[4:])
            # flip y and z axes
            rotation_matrix[:3, 1] = -rotation_matrix[:3, 1]
            rotation_matrix[:3, 2] = -rotation_matrix[:3, 2]
            rotation_dict["{:08d}".format(int(array[0]))] = rotation_matrix[:3, :3]
    frame_index_array.sort()
    return frame_index_array, translation_dict, rotation_dict


def get_filenames_from_frame_indexes(bag_root, frame_index_array):
    test_image_list = []
    for index in frame_index_array:
        temp = list(bag_root.glob('*/*{:08d}.jpg'.format(index)))
        if len(temp) == 0:
            print(index)
        else:
            test_image_list.append(temp[0])
    test_image_list.sort()
    return test_image_list


def outlier_detection(i, epoch, sparse_flow_weight, sparse_flow_loss, display, flows_1, flows_from_depth_1,
                      flow_masks_1,
                      flows_2, flows_from_depth_2, flow_masks_2, folders, boundaries, scaled_depth_maps_1,
                      scaled_depth_maps_2, colors_1, colors_2, is_hsv):
    print("batch {:d} in epoch {:d} has large loss {:.5f}".format(i, epoch,
                                                                  sparse_flow_weight * sparse_flow_loss.item()))

    losses_1 = display(
        [flows_1, flows_from_depth_1, flow_masks_1])
    losses_2 = display(
        [flows_2, flows_from_depth_2, flow_masks_2])

    indice_1 = torch.argmax(losses_1, dim=0, keepdim=False)
    indice_2 = torch.argmax(losses_2, dim=0, keepdim=False)

    print("pair 1 max loss: {:.5f}, pair 2 max loss: {:.5f}".format(torch.max(losses_1).item(),
                                                                    torch.max(losses_2).item()))
    print(folders[indice_1.item()], folders[indice_2.item()])
    visualize_color_image("mask_sample_", boundaries, rebias=False, is_hsv=False,
                          idx=[indice_1.item(), indice_2.item()])

    visualize_color_image("original color_1_sample_", colors_1, rebias=True,
                          is_hsv=is_hsv, idx=[indice_1.item()])
    visualize_depth_map("depth_1_sample_", scaled_depth_maps_1, idx=[indice_1.item()])
    draw_hsv(flows_1, "sparse_flow_1_sample_", idx=[indice_1.item()])
    draw_hsv(flows_from_depth_1, "flow from depth_1_sample_", idx=[indice_1.item()])

    visualize_color_image("original color_2_sample_", colors_2, rebias=True,
                          is_hsv=is_hsv, idx=[indice_2.item()])
    visualize_depth_map("depth_2_sample_", scaled_depth_maps_2, idx=[indice_2.item()])
    draw_hsv(flows_2, "sparse_flow_2_sample_", idx=[indice_2.item()])
    draw_hsv(flows_from_depth_2, "flow from depth_2_sample_", idx=[indice_2.item()])
    cv2.waitKey()
    cv2.destroyAllWindows()


def outlier_detection_processing(failure_threshold, sparse_masked_l1_loss_detector, flows,
                                 flows_from_depth, flow_masks):
    failure_detection_loss = sparse_masked_l1_loss_detector(
        [flows, flows_from_depth, flow_masks])
    indexes = []
    for j in range(failure_detection_loss.shape[0]):
        if failure_detection_loss[j].item() > failure_threshold:
            indexes.append(j)
    return indexes, failure_detection_loss


def learn_from_teacher(boundaries, colors_1, colors_2, depth_estimation_model_teacher,
                       depth_estimation_model_student,
                       scale_invariant_loss):
    # Predicted depth from teacher model (where sparse signal can be easily propagated)
    goal_depth_maps_1 = depth_estimation_model_teacher(colors_1)
    goal_depth_maps_2 = depth_estimation_model_teacher(colors_2)
    # Predicted depth from student model
    predicted_depth_maps_1 = depth_estimation_model_student(colors_1)
    predicted_depth_maps_2 = depth_estimation_model_student(colors_2)

    abs_goal_depth_maps_1 = torch.abs(goal_depth_maps_1)
    abs_goal_depth_maps_2 = torch.abs(goal_depth_maps_2)

    abs_predicted_depth_maps_1 = torch.abs(predicted_depth_maps_1)
    abs_predicted_depth_maps_2 = torch.abs(predicted_depth_maps_2)

    loss = 0.5 * scale_invariant_loss(
        [abs_predicted_depth_maps_1, abs_goal_depth_maps_1, boundaries]) + \
           0.5 * scale_invariant_loss(
        [abs_predicted_depth_maps_2, abs_goal_depth_maps_2, boundaries])
    return loss, torch.abs(predicted_depth_maps_1), torch.abs(predicted_depth_maps_2), \
           torch.abs(goal_depth_maps_1), torch.abs(goal_depth_maps_2)


def learn_from_sfm(colors_1, colors_2, sparse_depths_1, sparse_depths_2, sparse_depth_masks_1, sparse_depth_masks_2,
                   depth_estimation_model_student, depth_scaling_layer, sparse_flow_weight, flow_from_depth_layer,
                   boundaries,
                   translations, rotations, intrinsic_matrices, translations_inverse, rotations_inverse,
                   flow_masks_1, flow_masks_2, flows_1, flows_2, enable_failure_detection,
                   sparse_masked_l1_loss, depth_consistency_weight, depth_warping_layer, masked_log_l2_loss,
                   batch, epoch, failure_threshold, sparse_masked_l1_loss_detector, epoch_failure_sequences,
                   folders, batch_size, visualize, scale_std_loss_weight):
    # Predicted depth from student model
    predicted_depth_maps_1 = depth_estimation_model_student(colors_1)
    predicted_depth_maps_2 = depth_estimation_model_student(colors_2)

    # print(torch.min(predicted_depth_maps_1), torch.min(predicted_depth_maps_2))
    scaled_depth_maps_1, normalized_scale_std_1 = depth_scaling_layer(
        [torch.abs(predicted_depth_maps_1), sparse_depths_1, sparse_depth_masks_1])
    scaled_depth_maps_2, normalized_scale_std_2 = depth_scaling_layer(
        [torch.abs(predicted_depth_maps_2), sparse_depths_2, sparse_depth_masks_2])

    scale_std_loss = 0.5 * normalized_scale_std_1 + 0.5 * normalized_scale_std_2
    flows_from_depth_1 = torch.tensor(0.0).float().cuda()
    flows_from_depth_2 = torch.tensor(0.0).float().cuda()
    depth_consistency_loss = torch.tensor(0.0).float().cuda()
    sparse_flow_loss = torch.tensor(0.0).float().cuda()
    warped_depth_maps_2_to_1 = 0.0
    warped_depth_maps_1_to_2 = 0.0

    if sparse_flow_weight > 0.0:
        # Sparse optical flow loss
        # Flow maps calculated using predicted dense depth maps and camera poses
        # should agree with the sparse flows of feature points from SfM
        flows_from_depth_1 = flow_from_depth_layer(
            [scaled_depth_maps_1, boundaries, translations, rotations,
             intrinsic_matrices])
        flows_from_depth_2 = flow_from_depth_layer(
            [scaled_depth_maps_2, boundaries, translations_inverse, rotations_inverse,
             intrinsic_matrices])
        flow_masks_1 = flow_masks_1 * boundaries
        flow_masks_2 = flow_masks_2 * boundaries
        flows_1 = flows_1 * boundaries
        flows_2 = flows_2 * boundaries
        flows_from_depth_1 = flows_from_depth_1 * boundaries
        flows_from_depth_2 = flows_from_depth_2 * boundaries
        # If we do not try to detect any failure case from SfM
        if not enable_failure_detection:
            sparse_flow_loss = 0.5 * sparse_masked_l1_loss(
                [flows_1, flows_from_depth_1, flow_masks_1]) + \
                               0.5 * sparse_masked_l1_loss(
                [flows_2, flows_from_depth_2, flow_masks_2])

    if depth_consistency_weight > 0.0:
        # Depth consistency loss
        warped_depth_maps_2_to_1, intersect_masks_1 = depth_warping_layer(
            [scaled_depth_maps_1, scaled_depth_maps_2, boundaries, translations, rotations,
             intrinsic_matrices])
        warped_depth_maps_1_to_2, intersect_masks_2 = depth_warping_layer(
            [scaled_depth_maps_2, scaled_depth_maps_1, boundaries, translations_inverse,
             rotations_inverse,
             intrinsic_matrices])
        depth_consistency_loss = 0.5 * masked_log_l2_loss(
            [scaled_depth_maps_1, warped_depth_maps_2_to_1, intersect_masks_1, translations]) + \
                                 0.5 * masked_log_l2_loss(
            [scaled_depth_maps_2, warped_depth_maps_1_to_2, intersect_masks_2, translations])
        if visualize:
            visualize_color_image("color_1_sample_", colors_1, rebias=True, is_hsv=True, idx=[0, ])
            visualize_color_image("color_2_sample_", colors_2, rebias=True, is_hsv=True, idx=[0, ])
            min_list, max_list = visualize_depth_map("depth_1_sample_", scaled_depth_maps_1 * boundaries, idx=[0, ],
                                                     color_mode=cv2.COLORMAP_JET)
            visualize_depth_map("depth_2_sample_", scaled_depth_maps_2 * boundaries, min_value_=min_list,
                                max_value_=max_list, idx=[0, ], color_mode=cv2.COLORMAP_JET)
            # visualize_depth_map("depth_2_sample_", scaled_depth_maps_2 * boundaries)
            visualize_depth_map("2_to_1_depth_sample_", intersect_masks_1 * warped_depth_maps_2_to_1,
                                min_value_=min_list, max_value_=max_list, idx=[0, ], color_mode=cv2.COLORMAP_JET)
            visualize_depth_map("1_to_2_depth_sample_", intersect_masks_2 * warped_depth_maps_1_to_2,
                                min_value_=min_list, max_value_=max_list, idx=[0, ], color_mode=cv2.COLORMAP_JET)
            draw_hsv(flows_from_depth_1, "flow1_sample_", idx=[0, ])
            draw_hsv(flows_from_depth_2, "flow2_sample_", idx=[0, ])
            cv2.waitKey()
    loss = 0.0
    # Bootstrapping data cleaning method
    if enable_failure_detection:
        failure_indexes_1, sparse_flow_losses_1 = outlier_detection_processing(failure_threshold,
                                                                               sparse_masked_l1_loss_detector,
                                                                               flows_1,
                                                                               flows_from_depth_1,
                                                                               flow_masks_1)
        failure_indexes_2, sparse_flow_losses_2 = outlier_detection_processing(failure_threshold,
                                                                               sparse_masked_l1_loss_detector,
                                                                               flows_2,
                                                                               flows_from_depth_2,
                                                                               flow_masks_2)
        for index in failure_indexes_1:
            epoch_failure_sequences[folders[index]] = 1
            sparse_flow_losses_1[index] = torch.tensor(0.0).float().cuda()

        for index in failure_indexes_2:
            epoch_failure_sequences[folders[index]] = 1
            sparse_flow_losses_2[index] = torch.tensor(0.0).float().cuda()

        if batch_size != len(failure_indexes_1) and batch_size != len(failure_indexes_2):
            sparse_flow_loss = torch.tensor(0.5).float().cuda() * (
                    torch.sum(sparse_flow_losses_1) / torch.tensor(
                batch_size - len(failure_indexes_1)).float().cuda() + torch.sum(
                sparse_flow_losses_2) / torch.tensor(
                batch_size - len(failure_indexes_2)).float().cuda())
        else:
            sparse_flow_loss = torch.tensor(0.5).float().cuda() * torch.mean(sparse_flow_losses_1) + \
                               torch.tensor(0.5).float().cuda() * torch.mean(sparse_flow_losses_2)

        loss = depth_consistency_weight * depth_consistency_loss + sparse_flow_weight * sparse_flow_loss + scale_std_loss_weight * scale_std_loss
    else:
        loss = depth_consistency_weight * depth_consistency_loss + sparse_flow_weight * sparse_flow_loss + scale_std_loss_weight * scale_std_loss

    return loss, scaled_depth_maps_1, scaled_depth_maps_2, epoch_failure_sequences, \
           depth_consistency_loss, sparse_flow_loss, scale_std_loss, warped_depth_maps_2_to_1, warped_depth_maps_1_to_2, predicted_depth_maps_1, sparse_flow_losses_1, sparse_flow_losses_2


def save_student_model(model_root, depth_estimation_model_student, optimizer, epoch,
                       step, failure_sequences, model_path_student, validation_losses, best_validation_losses,
                       save_best_only):
    model_path_epoch_student = model_root / 'checkpoint_student_model_epoch_{epoch}.pt'.format(epoch=epoch)
    validation_losses = np.array(validation_losses)
    best_validation_losses = np.array(best_validation_losses)

    # Checkpoint model
    save_model(model=depth_estimation_model_student, optimizer=optimizer,
               epoch=epoch + 1, step=step,
               model_path=model_path_epoch_student, failure_sequences=failure_sequences,
               validation_loss=validation_losses)

    # Best model
    # If we use the validation loss to select our model
    if save_best_only:
        # Save best validation loss model
        if calculate_outlier_robust_validation_loss(validation_losses, best_validation_losses) < 0.0:
            print("Found better model in terms of validation loss: {:.5f}".format(np.mean(validation_losses)))
            save_model(model=depth_estimation_model_student, optimizer=optimizer,
                       epoch=epoch + 1, step=step,
                       model_path=model_path_student, failure_sequences=failure_sequences,
                       validation_loss=validation_losses)
            return validation_losses
        else:
            return best_validation_losses

    else:
        save_model(model=depth_estimation_model_student, optimizer=optimizer,
                   epoch=epoch + 1, step=step,
                   model_path=model_path_student, failure_sequences=failure_sequences,
                   validation_loss=validation_losses)
        return validation_losses


def save_teacher_model(model_root, depth_estimation_model_teacher, optimizer, epoch,
                       step, failure_sequences, model_path_teacher, validation_losses, best_validation_losses,
                       save_best_only):
    model_path_epoch_teacher = model_root / 'checkpoint_teacher_model_epoch_{epoch}.pt'.format(epoch=epoch)
    validation_losses = np.array(validation_losses)
    best_validation_losses = np.array(best_validation_losses)

    # Checkpoint model
    save_model(model=depth_estimation_model_teacher, optimizer=optimizer,
               epoch=epoch + 1, step=step,
               model_path=model_path_epoch_teacher, failure_sequences=failure_sequences,
               validation_loss=validation_losses)
    # Best model
    # If we use the validation loss to select our model
    if save_best_only:
        # Save best validation loss model
        if calculate_outlier_robust_validation_loss(validation_losses, best_validation_losses) < 0.0:
            print("Found better model in terms of validation loss: {:.5f}".format(np.mean(validation_losses)))
            save_model(model=depth_estimation_model_teacher, optimizer=optimizer,
                       epoch=epoch + 1, step=step,
                       model_path=model_path_teacher, failure_sequences=failure_sequences,
                       validation_loss=validation_losses)
            return validation_losses
        else:
            return best_validation_losses

    else:
        save_model(model=depth_estimation_model_teacher, optimizer=optimizer,
                   epoch=epoch + 1, step=step,
                   model_path=model_path_teacher, failure_sequences=failure_sequences,
                   validation_loss=validation_losses)
        return validation_losses


def network_validation(writer, validation_loader, batch_size, epoch, depth_estimation_model_student, device,
                       depth_scaling_layer,
                       sparse_flow_weight, flow_from_depth_layer, sparse_masked_l1_loss, depth_consistency_weight,
                       masked_log_l2_loss,
                       is_hsv, depth_warping_layer, results_root, tq, which_bag):
    # Validation
    # Variable initialization
    depth_consistency_loss = torch.tensor(0.0).float().cuda()
    sparse_flow_loss = torch.tensor(0.0).float().cuda()
    scale_std_loss = torch.tensor(0.0).float().cuda()
    validation_losses = []
    validation_sparse_flow_losses = []
    validation_depth_consistency_losses = []
    for param in depth_estimation_model_student.parameters():
        param.requires_grad = False

    sample_batch = np.random.randint(low=0, high=len(validation_loader))
    for batch, (
            colors_1, colors_2, sparse_depths_1, sparse_depths_2, sparse_depth_masks_1,
            sparse_depth_masks_2,
            flows_1,
            flows_2, flow_masks_1, flow_masks_2, boundaries, rotations,
            rotations_inverse, translations, translations_inverse, intrinsic_matrices,
            folders) in enumerate(
        validation_loader):

        colors_1, colors_2, \
        sparse_depths_1, sparse_depths_2, sparse_depth_masks_1, sparse_depth_masks_2, flows_1, flows_2, flow_masks_1, \
        flow_masks_2, \
        boundaries, rotations, rotations_inverse, translations, translations_inverse, intrinsic_matrices = \
            colors_1.to(device), colors_2.to(device), \
            sparse_depths_1.to(device), sparse_depths_2.to(device), \
            sparse_depth_masks_1.to(device), sparse_depth_masks_2.to(device), flows_1.to(
                device), flows_2.to(
                device), flow_masks_1.to(device), flow_masks_2.to(device), \
            boundaries.to(device), rotations.to(device), \
            rotations_inverse.to(device), translations.to(device), translations_inverse.to(
                device), intrinsic_matrices.to(device)

        # Binarize the boundaries
        boundaries = torch.where(boundaries >= torch.tensor(0.9).float().cuda(),
                                 torch.tensor(1.0).float().cuda(), torch.tensor(0.0).float().cuda())
        # Remove invalid regions of color images
        colors_1 = boundaries * colors_1
        colors_2 = boundaries * colors_2

        # Predicted depth from student model
        predicted_depth_maps_1 = depth_estimation_model_student(colors_1)
        predicted_depth_maps_2 = depth_estimation_model_student(colors_2)
        scaled_depth_maps_1, normalized_scale_std_1 = depth_scaling_layer(
            [torch.abs(predicted_depth_maps_1), sparse_depths_1, sparse_depth_masks_1])
        scaled_depth_maps_2, normalized_scale_std_2 = depth_scaling_layer(
            [torch.abs(predicted_depth_maps_2), sparse_depths_2, sparse_depth_masks_2])

        if sparse_flow_weight > 0.0:
            # Sparse optical flow loss
            # Optical flow maps calculated using predicted dense depth maps and camera poses
            # should agree with the sparse optical flows of feature points from SfM
            flows_from_depth_1 = flow_from_depth_layer(
                [scaled_depth_maps_1, boundaries, translations, rotations,
                 intrinsic_matrices])
            flows_from_depth_2 = flow_from_depth_layer(
                [scaled_depth_maps_2, boundaries, translations_inverse, rotations_inverse,
                 intrinsic_matrices])
            flow_masks_1 = flow_masks_1 * boundaries
            flow_masks_2 = flow_masks_2 * boundaries
            flows_1 = flows_1 * boundaries
            flows_2 = flows_2 * boundaries
            flows_from_depth_1 = flows_from_depth_1 * boundaries
            flows_from_depth_2 = flows_from_depth_2 * boundaries
            # If we do not try to detect any failure case from SfM
            sparse_flow_loss = 0.5 * sparse_masked_l1_loss(
                [flows_1, flows_from_depth_1, flow_masks_1]) + \
                               0.5 * sparse_masked_l1_loss(
                [flows_2, flows_from_depth_2, flow_masks_2])

        if depth_consistency_weight > 0.0:
            # Depth consistency loss
            warped_depth_maps_2_to_1, intersect_masks_1 = depth_warping_layer(
                [scaled_depth_maps_1, scaled_depth_maps_2, boundaries, translations, rotations,
                 intrinsic_matrices])
            warped_depth_maps_1_to_2, intersect_masks_2 = depth_warping_layer(
                [scaled_depth_maps_2, scaled_depth_maps_1, boundaries, translations_inverse,
                 rotations_inverse,
                 intrinsic_matrices])
            depth_consistency_loss = 0.5 * masked_log_l2_loss(
                [scaled_depth_maps_1, warped_depth_maps_2_to_1, intersect_masks_1, translations]) + \
                                     0.5 * masked_log_l2_loss(
                [scaled_depth_maps_2, warped_depth_maps_1_to_2, intersect_masks_2, translations])

        loss = depth_consistency_weight * depth_consistency_loss + sparse_flow_weight * sparse_flow_loss

        # Avoid the effects of nan samples
        if not np.isnan(loss.item()):
            validation_losses.append(loss.item())
            validation_sparse_flow_losses.append(sparse_flow_weight * sparse_flow_loss.item())
            validation_depth_consistency_losses.append(
                depth_consistency_weight * depth_consistency_loss.item())
            tq.set_postfix(loss='{:.5f} {:.5f}'.format(np.mean(validation_losses), loss.item()),
                           loss_depth_consistency='{:.5f} {:.5f}'.format(
                               np.mean(validation_depth_consistency_losses),
                               depth_consistency_weight * depth_consistency_loss.item()),
                           loss_sparse_flow='{:.5f} {:.5f}'.format(np.mean(validation_sparse_flow_losses),
                                                                   sparse_flow_weight * sparse_flow_loss.item()))
        tq.update(batch_size)

        if batch == sample_batch:
            generate_validation_output(1, epoch, writer, colors_1, scaled_depth_maps_1 * boundaries, boundaries,
                                       intrinsic_matrices,
                                       is_hsv, results_root, which_bag)

    # TensorboardX
    writer.add_scalars('Validation', {'overall': np.mean(validation_losses),
                                      'depth consistency': np.mean(validation_depth_consistency_losses),
                                      'sparse opt': np.mean(validation_sparse_flow_losses)}, epoch)

    return np.mean(validation_losses), validation_losses


def calculate_outlier_robust_validation_loss(validation_losses, previous_validation_losses):
    if len(validation_losses) == len(previous_validation_losses):
        differences = validation_losses - previous_validation_losses

        positive = np.sum(np.sum(np.int32(differences > 0.0)) * (differences > 0.0) * differences)
        negative = np.sum(np.sum(np.int32(differences < 0.0)) * (differences < 0.0) * differences)
        return positive + negative
    elif len(validation_losses) > len(previous_validation_losses):
        return -1.0
    else:
        return 1.0


def read_pose_corresponding_image_indexes(file_path):
    pose_corresponding_video_frame_index_array = []
    with open(file_path, "r") as filestream:
        for pose_index, line in enumerate(filestream):
            array = line.split(", ")
            array = np.array(array, dtype=np.float32)
            pose_corresponding_video_frame_index_array.append(int(array[0]))
    pose_corresponding_video_frame_index_array = np.array(pose_corresponding_video_frame_index_array,
                                                          dtype=np.float32)
    return pose_corresponding_video_frame_index_array


def read_pose_corresponding_image_indexes_and_time_difference(file_path):
    pose_corresponding_video_frame_index_array = []
    pose_corresponding_video_frame_time_difference_array = []
    with open(file_path, "r") as filestream:
        for pose_index, line in enumerate(filestream):
            array = line.split(", ")
            array = np.array(array, dtype=np.float32)
            pose_corresponding_video_frame_index_array.append(int(array[0]))
            pose_corresponding_video_frame_time_difference_array.append(int(array[1]))
    pose_corresponding_video_frame_index_array = np.array(pose_corresponding_video_frame_index_array,
                                                          dtype=np.int32)
    pose_corresponding_video_frame_time_difference_array = np.array(
        pose_corresponding_video_frame_time_difference_array, dtype=np.int32)
    return pose_corresponding_video_frame_index_array, pose_corresponding_video_frame_time_difference_array


def synchronize_selected_calibration_poses(root):
    pose_messages_path = root / "poses"
    translation_array_EM, rotation_array_EM = read_pose_messages_from_tracker(str(pose_messages_path))

    pose_image_indexes_path = root / "pose_corresponding_image_indexes"
    pose_corresponding_video_frame_index_array = read_pose_corresponding_image_indexes(str(pose_image_indexes_path))

    selected_calibration_image_name_list = list(root.glob('*.jpg'))

    # Find the most likely camera position
    for calibration_image_name in selected_calibration_image_name_list:
        calibration_image_name = str(calibration_image_name)
        difference_array = pose_corresponding_video_frame_index_array.astype(np.int32) - int(
            calibration_image_name[-12:-4])
        # Find if there are some zeros in it
        zero_indexes, = np.where(difference_array == 0)

        translation = np.zeros((3,), dtype=np.float64)
        rotation = np.zeros((3, 3), dtype=np.float64)
        # Average over these corresponding EM positions
        if zero_indexes.size != 0:
            flag = ""
            sum_count = 0
            for count, zero_index in enumerate(zero_indexes):
                translation += translation_array_EM[zero_index]
                rotation += rotation_array_EM[zero_index]
                sum_count = count + 1.0
            translation /= sum_count
            # print("previous", rotation / sum_count)
            if sum_count > 1.0:
                rotation = rotation_array_EM[zero_indexes[0]]
                # rotation = average_rotation(rotation / sum_count)
                # print("averaged", rotation)
        # Find the closest EM positions and use that instead
        else:
            min_indexes = np.argmin(np.abs(difference_array))
            flag = ""
            # If the closest frame are too far away, raise an error for bug inspection
            if np.amin(np.abs(difference_array)) > 10:
                flag = "bad"
                print("no best matches available for image {}".format(calibration_image_name))
                # raise OSError

            if hasattr(min_indexes, "__len__"):
                # Average over all the corresponding EM positions
                sum_count = 0
                for count, min_index in enumerate(min_indexes):
                    translation += translation_array_EM[min_index]
                    rotation += rotation_array_EM[min_index]
                    sum_count = count + 1.0
                translation /= sum_count
                # print("previous", rotation / sum_count)
                if sum_count > 1.0:
                    rotation = rotation_array_EM[min_indexes[0]]
                    # rotation = average_rotation(rotation / sum_count)
                    # print("averaged", rotation)
            else:
                translation = translation_array_EM[min_indexes]
                rotation = rotation_array_EM[min_indexes]

        with open(calibration_image_name[:-4] + flag + ".coords", "w") as filestream:
            for i in range(3):
                filestream.write("{:.5f},".format(translation[i]))
            for i in range(3):
                for j in range(3):
                    if i != 2 or j != 2:
                        filestream.write("{:.5f},".format(rotation[i][j]))
                    else:
                        filestream.write("{:.5f}\n".format(rotation[i][j]))
    return


def synchronize_image_and_poses(root, tolerance_threshold=1.0e6):
    pose_messages_path = root / "bags" / "poses_calibration"
    translation_array_EM, rotation_array_EM = read_pose_messages_from_tracker(str(pose_messages_path))

    pose_image_indexes_path = root / "bags" / "pose_corresponding_image_indexes_calibration"
    pose_corresponding_video_frame_index_array, pose_corresponding_video_frame_time_difference_array = \
        read_pose_corresponding_image_indexes_and_time_difference(str(pose_image_indexes_path))

    best_matches_pose_indexes = np.where(pose_corresponding_video_frame_time_difference_array < tolerance_threshold)
    best_matches_pose_indexes = best_matches_pose_indexes[0]
    selected_video_frame_index_array = pose_corresponding_video_frame_index_array[best_matches_pose_indexes]

    selected_calibration_root = root / "selected_calibration_images"
    calibration_root = root / "calibration_images"
    try:
        selected_calibration_root.mkdir(mode=0o777, parents=True)
    except OSError:
        pass

    for ori_index, selected_video_frame_index in enumerate(selected_video_frame_index_array):

        dest = selected_calibration_root / "{:08d}.jpg".format(selected_video_frame_index)
        if not dest.exists():
            shutil.copyfile(str(calibration_root / "{:08d}.jpg".format(selected_video_frame_index)),
                            str(dest))

        translation = translation_array_EM[best_matches_pose_indexes[ori_index]]
        rotation = rotation_array_EM[best_matches_pose_indexes[ori_index]]
        with open(str(selected_calibration_root / "{:08d}.coords".format(selected_video_frame_index)),
                  "w") as filestream:
            for i in range(3):
                filestream.write("{:.5f},".format(translation[i]))
            for i in range(3):
                for j in range(3):
                    if i != 2 or j != 2:
                        filestream.write("{:.5f},".format(rotation[i][j]))
                    else:
                        filestream.write("{:.5f}\n".format(rotation[i][j]))

    return


def read_camera_to_tcp_transform(root):
    transform = np.zeros((3, 4), dtype=np.float)
    with open(str(root / "camera_to_tcp"), "r") as filestream:
        for line in filestream:
            temp = line.split(" ")
            temp = np.array(temp, dtype=np.float)

    for i in range(3):
        for j in range(4):
            transform[i, j] = temp[4 * i + j]
    return transform[:, :3], transform[:, 3].reshape((3, 1))


def get_parent_folder_names(root, id_range):
    folder_list = []
    for i in range(id_range[0], id_range[1]):
        folder_list += list(root.glob('*' + str(i) + '/_start*/'))

    folder_list.sort()
    return folder_list


def get_file_names_in_sequence(sequence_root, suffix):
    path = sequence_root / 'visible_view_indexes{}'.format(suffix)
    if not path.exists():
        return []

    visible_view_indexes = read_visible_view_indexes(sequence_root, suffix)
    filenames = []
    for index in visible_view_indexes:
        filenames.append(sequence_root / "{:08d}.jpg".format(index))
    return filenames


def get_all_color_image_names_in_sequence(sequence_root):
    _, view_indexes = read_selected_indexes(sequence_root)
    filenames = []
    for index in view_indexes:
        filenames.append(sequence_root / "{:08d}.jpg".format(index))
    return filenames


def overlapping_visible_view_indexes_per_point(visible_view_indexes_per_point, visible_interval):
    temp_array = np.copy(visible_view_indexes_per_point)
    view_count = visible_view_indexes_per_point.shape[1]
    for i in range(view_count):
        visible_view_indexes_per_point[:, i] = \
            np.sum(temp_array[:, max(0, i - visible_interval):min(view_count, i + visible_interval)], axis=1)

    return visible_view_indexes_per_point


def write_scaled_point_cloud(path, point_cloud, scale):
    point_clouds_list = []
    for i in range(len(point_cloud)):
        point = point_cloud[i]
        point_clouds_list.append(
            (point[0] / scale, point[1] / scale, point[2] / scale))

    vertex = np.array(point_clouds_list,
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], text=True).write(path)
    return


def global_scale_estimation(extrinsics, point_cloud):
    max_bound = np.zeros((3,), dtype=np.float32)
    min_bound = np.zeros((3,), dtype=np.float32)

    for i, extrinsic in enumerate(extrinsics):
        if i == 0:
            max_bound = extrinsic[:3, 3]
            min_bound = extrinsic[:3, 3]
        else:
            temp = extrinsic[:3, 3]
            max_bound = np.maximum(max_bound, temp)
            min_bound = np.minimum(min_bound, temp)

    norm_1 = np.linalg.norm(max_bound - min_bound, ord=2)

    max_bound = np.zeros((3,), dtype=np.float32)
    min_bound = np.zeros((3,), dtype=np.float32)
    for i, point in enumerate(point_cloud):
        if i == 0:
            max_bound = np.asarray(point[:3], dtype=np.float32)
            min_bound = np.asarray(point[:3], dtype=np.float32)
        else:
            temp = np.asarray(point[:3], dtype=np.float32)
            if np.any(np.isnan(temp)):
                continue
            max_bound = np.maximum(max_bound, temp)
            min_bound = np.minimum(min_bound, temp)

    norm_2 = np.linalg.norm(max_bound - min_bound, ord=2)

    return max(1.0, max(norm_1, norm_2))


def surface_mesh_global_scale(surface_mesh):
    max_bound = np.max(surface_mesh.vertices, axis=0)
    min_bound = np.min(surface_mesh.vertices, axis=0)

    return np.linalg.norm(max_bound - min_bound, ord=2), np.linalg.norm(min_bound, ord=2), np.abs(
        max_bound[2] - min_bound[0])


def generate_heatmap_from_locations(feature_2D_locations, height, width, sigma):
    sample_size, _ = feature_2D_locations.shape

    feature_2D_locations = np.reshape(feature_2D_locations, (sample_size, 4))

    source_heatmaps = []
    target_heatmaps = []

    sigma_2 = sigma ** 2
    for i in range(sample_size):
        x = feature_2D_locations[i, 0]
        y = feature_2D_locations[i, 1]

        x_2 = feature_2D_locations[i, 2]
        y_2 = feature_2D_locations[i, 3]

        y_grid, x_grid = np.meshgrid(np.arange(height), np.arange(width), sparse=False, indexing='ij')

        source_grid_x = x_grid - x
        source_grid_y = y_grid - y

        target_grid_x = x_grid - x_2
        target_grid_y = y_grid - y_2

        heatmap = np.exp(-(source_grid_x ** 2 + source_grid_y ** 2) / (2.0 * sigma_2))
        heatmap_2 = np.exp(-(target_grid_x ** 2 + target_grid_y ** 2) / (2.0 * sigma_2))

        source_heatmaps.append(heatmap)
        target_heatmaps.append(heatmap_2)

    source_heatmaps = np.asarray(source_heatmaps, dtype=np.float32).reshape((sample_size, height, width))
    target_heatmaps = np.asarray(target_heatmaps, dtype=np.float32).reshape((sample_size, height, width))

    return source_heatmaps, target_heatmaps


# Checks if a matrix is a valid rotation matrix.
def is_rotation(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotation_to_euler_angles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        gamma = math.atan2(R[2, 1], R[2, 2])
        beta = math.atan2(-R[2, 0], sy)
        alpha = math.atan2(R[1, 0], R[0, 0])
    else:
        gamma = math.atan2(R[0, 1], R[1, 1])
        beta = np.pi / 2.0
        alpha = 0
    return np.array([alpha, beta, gamma])


def rotation_matrix_from_rpy(rpy):
    cos_alpha = torch.cos(rpy[:, 0]).view(-1, 1)
    cos_beta = torch.cos(rpy[:, 1]).view(-1, 1)
    cos_gamma = torch.cos(rpy[:, 2]).view(-1, 1)

    sin_alpha = torch.sin(rpy[:, 0]).view(-1, 1)
    sin_beta = torch.sin(rpy[:, 1]).view(-1, 1)
    sin_gamma = torch.sin(rpy[:, 2]).view(-1, 1)

    # alpha-x: yaw, beta-y: pitch, gamma-z: roll
    temp = [None for i in range(9)]

    temp[0] = cos_alpha * cos_beta
    temp[1] = cos_alpha * sin_beta * sin_gamma - sin_alpha * cos_gamma
    temp[2] = cos_alpha * sin_beta * cos_gamma + sin_alpha * sin_gamma
    temp[3] = sin_alpha * cos_beta
    temp[4] = sin_alpha * sin_beta * sin_gamma + cos_alpha * cos_gamma
    temp[5] = sin_alpha * sin_beta * cos_gamma - cos_alpha * sin_gamma
    temp[6] = -sin_beta
    temp[7] = cos_beta * sin_gamma
    temp[8] = cos_beta * cos_gamma

    rotation_matrix = torch.cat(temp, dim=1)
    rotation_matrix = rotation_matrix.view(-1, 3, 3)
    return rotation_matrix


def images_warping(images, source_coord_w_flat, source_coord_h_flat, padding_mode="zeros"):
    batch_num, channels, image_h, image_w = images.shape
    warped_images_flat = _bilinear_interpolate(images.permute(0, 2, 3, 1), x=source_coord_w_flat,
                                               y=source_coord_h_flat, padding_mode=padding_mode)
    warped_images = warped_images_flat.view(batch_num, image_h, image_w, channels).permute(0, 3, 1, 2)
    return warped_images


def images_rotation_coordinates_calculate(thetas, image_h, image_w):
    # B x 1 x 1
    cos_theta = torch.cos(thetas).view(-1, 1, 1)
    sin_theta = torch.sin(thetas).view(-1, 1, 1)

    image_center_h = torch.tensor(np.floor(image_h / 2.0)).float().cuda()
    image_center_w = torch.tensor(np.floor(image_w / 2.0)).float().cuda()

    h_grid, w_grid = torch.meshgrid(
        [torch.arange(start=0, end=image_h, dtype=torch.float32).cuda(),
         torch.arange(start=0, end=image_w, dtype=torch.float32).cuda()])

    # 1 x H x W
    h_grid = h_grid.view(1, image_h, image_w)
    w_grid = w_grid.view(1, image_h, image_w)

    # B x H x W
    source_coord_w = cos_theta * (w_grid - image_center_w) + \
                     sin_theta * (h_grid - image_center_h) + image_center_w
    source_coord_h = -sin_theta * (w_grid - image_center_w) + \
                     cos_theta * (h_grid - image_center_h) + image_center_h

    source_coord_h_flat = source_coord_h.view(-1)
    source_coord_w_flat = source_coord_w.view(-1)

    return source_coord_h_flat, source_coord_w_flat


def images_horizontal_flipping_coordinates_calculation(batch_size, image_h, image_w):
    h_grid, w_grid = torch.meshgrid(
        [torch.arange(start=0, end=image_h, dtype=torch.float32).cuda(),
         torch.arange(start=image_w - 1, end=-1, step=-1, dtype=torch.float32).cuda()])

    # 1 x H x W
    h_grid = h_grid.view(1, image_h, image_w)
    w_grid = w_grid.view(1, image_h, image_w)

    source_coord_h_flat = h_grid.expand(batch_size, -1, -1).contiguous().view(-1)
    source_coord_w_flat = w_grid.expand(batch_size, -1, -1).contiguous().view(-1)

    return source_coord_h_flat, source_coord_w_flat


def images_vertical_flipping_coordinates_calculation(batch_size, image_h, image_w):
    h_grid, w_grid = torch.meshgrid(
        [torch.arange(start=image_h - 1, end=-1, step=-1, dtype=torch.float32).cuda(),
         torch.arange(start=0, end=image_w, dtype=torch.float32).cuda()])

    # 1 x H x W
    h_grid = h_grid.view(1, image_h, image_w)
    w_grid = w_grid.view(1, image_h, image_w)

    source_coord_h_flat = h_grid.expand(batch_size, -1, -1).contiguous().view(-1)
    source_coord_w_flat = w_grid.expand(batch_size, -1, -1).contiguous().view(-1)

    return source_coord_h_flat, source_coord_w_flat


def _bilinear_interpolate(im, x, y, padding_mode="zeros"):
    num_batch, height, width, channels = im.shape
    # Range [-1, 1]
    grid = torch.cat([torch.tensor(2.0).float().cuda() *
                      (x.view(num_batch, height, width, 1) / torch.tensor(width).float().cuda())
                      - torch.tensor(1.0).float().cuda(), torch.tensor(2.0).float().cuda() * (
                              y.view(num_batch, height, width, 1) / torch.tensor(height).float().cuda()) - torch.tensor(
        1.0).float().cuda()], dim=-1)

    return torch.nn.functional.grid_sample(input=im.permute(0, 3, 1, 2), grid=grid, mode='bilinear',
                                           padding_mode=padding_mode).permute(0, 2, 3, 1)


def convert_depth_map_to_point_cloud(depth_map, valid_mask, cam_pose, cam_intr, downsampling=5):
    valid_mask = valid_mask.reshape((-1))
    valid_indexes = np.where(valid_mask > 0.9)
    x_grid, y_grid = np.meshgrid(np.arange(depth_map.shape[1]), np.arange(depth_map.shape[0]))
    y_grid = y_grid.reshape((-1))
    x_grid = x_grid.reshape((-1))
    z_coord = depth_map.reshape((-1))

    fx = cam_intr[0, 0]
    fy = cam_intr[1, 1]
    cx = cam_intr[0, 2]
    cy = cam_intr[1, 2]

    x_coord = ((x_grid - cx) * z_coord) / fx
    y_coord = ((y_grid - cy) * z_coord) / fy
    ones = np.ones_like(x_coord)

    x_coord = x_coord[valid_indexes]
    y_coord = y_coord[valid_indexes]
    z_coord = z_coord[valid_indexes]
    ones = ones[valid_indexes]

    coord_3D = np.ones((4, x_coord.shape[0]), dtype=np.float32)
    coord_3D[0, :] = x_coord
    coord_3D[1, :] = y_coord
    coord_3D[2, :] = z_coord
    world_coord_3D = np.matmul(cam_pose, coord_3D)
    world_coord_3D /= (world_coord_3D[3, :]).reshape((1, -1))
    return world_coord_3D[:3, ::downsampling]


def write_point_cloud_position_only(path, point_cloud):
    point_clouds_list = []
    for i in range(point_cloud.shape[1]):
        point_clouds_list.append(
            (point_cloud[0, i], point_cloud[1, i], point_cloud[2, i]))

    vertex = np.array(point_clouds_list,
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], text=True).write(path)
    return


def hamming_distance(a, b):
    r = (1 << np.arange(8))[:, None]
    return np.count_nonzero((np.bitwise_xor(a, b) & r) != 0)


def orb_feature_detection(img):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # # Initiate STAR detector
    # orb = cv2.KAZE_create()
    #
    # # find the keypoints with ORB
    # kp, des = orb.detectAndCompute(img, None)
    #
    # # # compute the descriptors with ORB
    # # kp, des = orb.compute(img, kp)
    #
    # # draw only keypoints location,not size and orientation
    # cv2.drawKeypoints(img, kp, color=(0, 255, 0), flags=0, outImage=img)
    # sift = cv2.xfeatures2d.SIFT_create()
    # sift = cv2.xfeatures2d.SIFT_create()
    # (kps, descs) = sift.detectAndCompute(gray, None)
    # print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
    # display_img = cv2.drawKeypoints(gray, kps)

    # Initiate FAST object with default values
    # fast = cv2.FastFeatureDetector_create()
    orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0, WTA_K=2,
                         scoreType=cv2.ORB_HARRIS_SCORE, patchSize=101, fastThreshold=10)
    kp, des = orb.detectAndCompute(img, None)

    # kp, des = sift.detectAndCompute(img, None)
    # find and draw the keypoints
    # kp = fast.detect(img, None)
    # cv2.drawKeypoints(img, kp, color=(255, 0, 0), outImage=img)
    # # Print all default params
    # print "Threshold: ", fast.getInt('threshold')
    # print "nonmaxSuppression: ", fast.getBool('nonmaxSuppression')
    # print "neighborhood: ", fast.getInt('type')
    # print "Total Keypoints with nonmaxSuppression: ", len(kp)
    # cv2.imshow("kp", img)
    # return img,
    return kp, des
    #
    # # create BFMatcher object
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    #
    # # Match descriptors.
    # matches = bf.match(des1, des2)
    #
    # # Sort them in the order of their distance.
    # matches = sorted(matches, key=lambda x: x.distance)
    #
    # # Draw first 10 matches.
    # img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], flags=2)
    #
    # plt.imshow(img3), plt.show()


def feature_matching_sparse_sift_vicinity(color_1, color_2, display_number, kps_1, kps_2, des_1, des_2,
                                          cross_check_distance,
                                          display_matches=False):
    # H x W x 3
    color_1 = np.moveaxis(color_1, source=[0, 1, 2], destination=[2, 0, 1])
    color_2 = np.moveaxis(color_2, source=[0, 1, 2], destination=[2, 0, 1])

    # Extract corner points
    color_1 = np.uint8(255 * (color_1 * 0.5 + 0.5))
    color_2 = np.uint8(255 * (color_2 * 0.5 + 0.5))

    # 1 is query, 2 is train
    bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)
    sift_matches = bf.knnMatch(des_1, des_2, k=1)
    reverse_sift_matches = bf.knnMatch(des_2, des_1, k=1)

    sift_good = []
    for kp, m in zip(kps_1, sift_matches):
        twice_kp = kps_1[reverse_sift_matches[m[0].trainIdx][0].trainIdx]
        if np.sqrt((twice_kp.pt[0] - kp.pt[0]) ** 2 + (twice_kp.pt[1] - kp.pt[1]) ** 2) < cross_check_distance:
            sift_good.append(m[0])

    def func(x):
        return x.distance

    if display_matches:
        if display_number < len(sift_good):
            sift_good = sorted(sift_good, key=func)
        display_matches_sift = cv2.drawMatches(color_1, kps_1, color_2, kps_2, sift_good[:display_number], flags=2,
                                               outImg=None)
        return display_matches_sift
    else:
        return sift_good


def feature_matching_sparse_sift(color_1, color_2, display_number, kps_1, kps_2, des_1, des_2, display_matches=False):
    # 1 is query, 2 is train
    bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)
    sift_matches = bf.knnMatch(des_1, des_2, k=1)
    sift_good = []
    for m in sift_matches:
        if len(m) != 0:
            sift_good.append(m[0])

    def func(x):
        return x.distance

    if display_matches:
        # H x W x 3
        color_1 = np.moveaxis(color_1, source=[0, 1, 2], destination=[2, 0, 1])
        color_2 = np.moveaxis(color_2, source=[0, 1, 2], destination=[2, 0, 1])

        # Extract corner points
        color_1 = np.uint8(255 * (color_1 * 0.5 + 0.5))
        color_2 = np.uint8(255 * (color_2 * 0.5 + 0.5))
        if display_number < len(sift_good):
            sift_good = sorted(sift_good, key=func)
        display_matches_sift = cv2.drawMatches(color_1, kps_1, color_2, kps_2, sift_good[:display_number], flags=2,
                                               outImg=None)
        return display_matches_sift
    else:
        return sift_good


def feature_matching_sparse_dl(color_1, color_2, rough_feature_map_1, rough_feature_map_2, fine_feature_map_1,
                               fine_feature_map_2,
                               display_number, kps_1, kps_2, des_1, des_2, display_matches=False):
    # Concatenate the features together to approximate the feature map multiplication
    feature_map_1 = np.concatenate([rough_feature_map_1, fine_feature_map_1], axis=0)
    feature_map_2 = np.concatenate([rough_feature_map_2, fine_feature_map_2], axis=0)
    feature_length, height, width = feature_map_1.shape

    # H x W x C
    feature_map_1 = np.moveaxis(feature_map_1, source=[0, 1, 2], destination=[2, 0, 1])
    feature_map_2 = np.moveaxis(feature_map_2, source=[0, 1, 2], destination=[2, 0, 1])

    # H x W x 3
    color_1 = np.moveaxis(color_1, source=[0, 1, 2], destination=[2, 0, 1])
    color_2 = np.moveaxis(color_2, source=[0, 1, 2], destination=[2, 0, 1])

    # Extract corner points
    color_1 = cv2.cvtColor(np.uint8(255 * (color_1 * 0.5 + 0.5)), cv2.COLOR_HSV2BGR_FULL)
    color_2 = cv2.cvtColor(np.uint8(255 * (color_2 * 0.5 + 0.5)), cv2.COLOR_HSV2BGR_FULL)

    # Extract keypoints and calculate the distance against the entire other dense feature map to find the best correspondence
    feature_map_1 = feature_map_1.reshape((-1, feature_length))
    feature_map_2 = feature_map_2.reshape((-1, feature_length))

    indexes_1 = []
    indexes_2 = []
    for point in kps_1:
        indexes_1.append(np.round(point.pt[0]) + np.round(point.pt[1]) * width)

    for point in kps_2:
        indexes_2.append(np.round(point.pt[0]) + np.round(point.pt[1]) * width)

    # N x C
    selected_features_1 = feature_map_1[np.array(indexes_1, dtype=np.int32), :].reshape(
        (len(indexes_1), feature_length))
    selected_features_2 = feature_map_2[np.array(indexes_2, dtype=np.int32), :].reshape(
        (len(indexes_2), feature_length))

    # 1 is query, 2 is train
    bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)
    ai_matches = bf.knnMatch(selected_features_1, selected_features_2, k=1)

    if display_matches:
        ai_good = []
        for m in ai_matches:
            if len(m) != 0:
                ai_good.append(m[0])

        if display_number < len(ai_good):
            ai_good = sorted(ai_good, key=lambda x: x.distance)
        display_matches_ai = cv2.drawMatches(color_1, kps_1, color_2, kps_2, ai_good[:display_number], flags=2,
                                             outImg=None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        orb_matches = bf.knnMatch(des_1, des_2, k=1)
        orb_good = []
        for m in orb_matches:
            if len(m) != 0:
                orb_good.append(m[0])

        if display_number < len(orb_good):
            orb_good = sorted(orb_good, key=lambda x: x.distance)
        display_matches_orb = cv2.drawMatches(color_1, kps_1, color_2, kps_2, orb_good[:display_number], flags=2,
                                              outImg=None)
        return display_matches_ai, display_matches_orb
    else:
        ai_good = []
        for m in ai_matches:
            if len(m) != 0:
                ai_good.append(m[0])
        return ai_good


def knn_dense_feature_matching(color_1, color_2, kdt_1, kdt_2, rough_feature_map_1, rough_feature_map_2,
                               fine_feature_map_1,
                               fine_feature_map_2, kps_1, kps_1D_1, display_number, cross_check_distance,
                               display_matches=False, gpu_id=0):
    feature_map_1 = torch.cat([rough_feature_map_1, fine_feature_map_1], dim=0)
    feature_map_2 = torch.cat([rough_feature_map_2, fine_feature_map_2], dim=0)

    feature_length, height, width = feature_map_1.shape
    # Extend 1D locations to B x C x Sampling_size
    keypoint_number = len(kps_1D_1)
    source_1D_locations = torch.from_numpy(kps_1D_1).long().cuda(gpu_id).view(1, keypoint_number)

    # Sampled rough locator feature vectors
    sampled_source_feature_vectors = torch.gather(
        feature_map_1.view(feature_length, height * width), 1,
        source_1D_locations.expand(feature_length, -1).long())
    sampled_source_feature_vectors = sampled_source_feature_vectors.view(feature_length,
                                                                         keypoint_number).permute(1, 0).view(
        keypoint_number,
        feature_length)

    # Extract the feature vectors associated with the target 1D locations obtained from KDTree NN searching
    target_1D_locations = torch.from_numpy(kdt_2.query(sampled_source_feature_vectors.data.cpu().numpy(),
                                                       k=1, return_distance=False)).long().cuda(gpu_id).view(1,
                                                                                                             keypoint_number)
    sampled_target_feature_vectors = torch.gather(
        feature_map_2.view(feature_length, height * width), 1,
        target_1D_locations.long().expand(feature_length, -1))

    sampled_target_feature_vectors = sampled_target_feature_vectors.view(feature_length,
                                                                         keypoint_number).permute(1, 0).view(
        keypoint_number,
        feature_length)

    detected_source_1D_locations = torch.from_numpy(kdt_1.query(sampled_target_feature_vectors.data.cpu().numpy(), k=1,
                                                                return_distance=False)).cuda(gpu_id).float().view(
        keypoint_number, 1)

    source_1D_locations = source_1D_locations.float().view(keypoint_number, 1)
    source_keypoint_2D_locations = torch.cat(
        [torch.fmod(source_1D_locations, width),
         torch.floor(source_1D_locations / width)],
        dim=1).view(keypoint_number, 2).float()

    detected_source_keypoint_2D_locations = torch.cat(
        [torch.fmod(detected_source_1D_locations, width),
         torch.floor(detected_source_1D_locations / width)],
        dim=1).view(keypoint_number, 2).float()

    cross_check_correspondence_distances = torch.norm(
        source_keypoint_2D_locations - detected_source_keypoint_2D_locations, dim=1, p=2).view(keypoint_number)
    valid_correspondence_indexes = torch.nonzero(cross_check_correspondence_distances < cross_check_distance).view(
        -1)
    valid_target_1D_locations = torch.gather(target_1D_locations.view(-1),
                                             0, valid_correspondence_indexes.long())
    valid_target_1D_locations_cpu = valid_target_1D_locations.data.cpu().numpy()

    detected_target_keypoints = []
    for index in valid_target_1D_locations_cpu:
        detected_target_keypoints.append(
            cv2.KeyPoint(x=float(np.floor(index % width)), y=float(np.floor(index / width)), _size=1.0))

    valid_cross_check_correspondence_distances = torch.gather(cross_check_correspondence_distances.view(-1),
                                                              0, valid_correspondence_indexes.long())
    valid_cross_check_correspondence_distances_cpu = valid_cross_check_correspondence_distances.data.cpu().numpy()
    valid_correspondence_indexes_cpu = valid_correspondence_indexes.data.cpu().numpy()
    matches = []
    for i, query_index in enumerate(valid_correspondence_indexes_cpu):
        matches.append(cv2.DMatch(_queryIdx=query_index, _trainIdx=i,
                                  _distance=valid_cross_check_correspondence_distances_cpu[i]))
    matches = sorted(matches, key=lambda x: x.distance, reverse=True)

    if display_matches:
        color_1 = np.moveaxis(color_1, source=[0, 1, 2], destination=[2, 0, 1])
        color_2 = np.moveaxis(color_2, source=[0, 1, 2], destination=[2, 0, 1])

        # Extract corner points
        color_1 = cv2.cvtColor(np.uint8(255 * (color_1 * 0.5 + 0.5)), cv2.COLOR_HSV2BGR_FULL)
        color_2 = cv2.cvtColor(np.uint8(255 * (color_2 * 0.5 + 0.5)), cv2.COLOR_HSV2BGR_FULL)

        display_matches_ai = cv2.drawMatches(color_1, kps_1, color_2, detected_target_keypoints,
                                             matches[:display_number],
                                             flags=2, outImg=None)
        return display_matches_ai


def KD_tree_generation(rough_feature_map, fine_feature_map):
    feature_map = torch.cat([rough_feature_map, fine_feature_map], dim=0)
    feature_length, height, width = feature_map.shape
    feature_vectors = feature_map.view(feature_length, height * width).permute(1, 0).view(height * width,
                                                                                          feature_length)
    kdt = KDTree(feature_vectors.data.numpy(), leaf_size=30, metric='euclidean')
    return kdt


# def feature_matching(color_1, color_2, rough_feature_map_1, rough_feature_map_2, fine_feature_map_1, fine_feature_map_2,
#                      boundary, kps_1D_1, des_1, des_2,
#                      scale, threshold, display_number, cross_check_distance,
#                      kps_1=None, kps_2=None, display_matches=False, gpu_id=0):
#     # Color image 3 x H x W
#     # Feature map C x H x W
#     rough_feature_length, height, width = rough_feature_map_1.shape
#     fine_feature_length, height, width = fine_feature_map_1.shape
#
#     # Extend 1D locations to B x C x Sampling_size
#     keypoint_number = len(kps_1D_1)
#     rough_source_feature_1D_locations = torch.from_numpy(kps_1D_1).long().cuda(gpu_id).view(
#         1, 1,
#         keypoint_number).expand(
#         -1, rough_feature_length, -1)
#
#     # Sampled rough locator feature vectors
#     sampled_rough_feature_vectors = torch.gather(
#         rough_feature_map_1.view(1, rough_feature_length, height * width), 2,
#         rough_source_feature_1D_locations.long())
#     sampled_rough_feature_vectors = sampled_rough_feature_vectors.view(1, rough_feature_length,
#                                                                        keypoint_number,
#                                                                        1,
#                                                                        1).permute(0, 2, 1, 3,
#                                                                                   4).view(1,
#                                                                                           keypoint_number,
#                                                                                           rough_feature_length,
#                                                                                           1, 1)
#
#     rough_filter_response_map = torch.nn.functional.conv2d(
#         input=rough_feature_map_2.view(1, rough_feature_length, height, width),
#         weight=sampled_rough_feature_vectors.view(keypoint_number,
#                                                   rough_feature_length,
#                                                   1, 1), padding=0)
#
#     # 1 x Sampling_size x H x W
#     rough_filter_response_map = 0.5 * rough_filter_response_map + 0.5
#     rough_filter_response_map = torch.exp(scale * (rough_filter_response_map - threshold))
#     rough_filter_response_map = rough_filter_response_map / torch.sum(rough_filter_response_map,
#                                                                       dim=(2, 3),
#                                                                       keepdim=True)
#
#     # # Sampled texture matcher feature vectors
#     fine_source_feature_1D_locations = torch.from_numpy(kps_1D_1).cuda(gpu_id).long().view(1, 1,
#                                                                                            keypoint_number).expand(
#         -1, fine_feature_length, -1)
#     sampled_fine_feature_vectors = torch.gather(
#         fine_feature_map_1.view(1, fine_feature_length, height * width), 2,
#         fine_source_feature_1D_locations.long())
#     sampled_fine_feature_vectors = sampled_fine_feature_vectors.view(1, fine_feature_length,
#                                                                      keypoint_number, 1,
#                                                                      1).permute(0, 2, 1, 3, 4).view(
#         1, keypoint_number,
#         fine_feature_length,
#         1, 1)
#     fine_filter_response_map = torch.nn.functional.conv2d(
#         input=fine_feature_map_2.view(1, fine_feature_length, height, width),
#         weight=sampled_fine_feature_vectors.view(keypoint_number,
#                                                  fine_feature_length,
#                                                  1, 1), padding=0)
#     # 1 x Sampling_size x H x W
#     fine_filter_response_map = 0.5 * fine_filter_response_map + 0.5
#     fine_filter_response_map = torch.exp(
#         scale * (fine_filter_response_map - threshold)) * boundary.view(1, 1, height, width).expand(
#         -1, keypoint_number, -1, -1)
#     fine_filter_response_map = fine_filter_response_map / torch.sum(fine_filter_response_map,
#                                                                     dim=(2, 3), keepdim=True)
#
#     merged_response_map = rough_filter_response_map * fine_filter_response_map
#     max_reponses, max_indexes = torch.max(merged_response_map.view(keypoint_number, -1), dim=1,
#                                           keepdim=False)
#     # query is 1 and train is 2 here
#     selected_detected_1D_locations_2 = max_indexes.view(-1)
#     selected_max_responses = max_reponses.view(-1)
#     # Do cross check
#     rough_feature_1D_locations_2 = selected_detected_1D_locations_2.long().view(
#         1, 1, -1).expand(-1, rough_feature_length, -1)
#     keypoint_number_2 = keypoint_number
#
#     # Sampled rough locator feature vectors
#     sampled_rough_feature_vectors_2 = torch.gather(
#         rough_feature_map_2.view(1, rough_feature_length, height * width), 2,
#         rough_feature_1D_locations_2.long())
#     sampled_rough_feature_vectors_2 = sampled_rough_feature_vectors_2.view(1, rough_feature_length,
#                                                                            keypoint_number_2,
#                                                                            1,
#                                                                            1).permute(0, 2, 1, 3,
#                                                                                       4).view(1,
#                                                                                               keypoint_number_2,
#                                                                                               rough_feature_length,
#                                                                                               1, 1)
#
#     rough_filter_response_map_2 = torch.nn.functional.conv2d(
#         input=rough_feature_map_1.view(1, rough_feature_length, height, width),
#         weight=sampled_rough_feature_vectors_2.view(keypoint_number_2,
#                                                     rough_feature_length,
#                                                     1, 1), padding=0)
#
#     # 1 x Sampling_size x H x W
#     rough_filter_response_map_2 = 0.5 * rough_filter_response_map_2 + 0.5
#     rough_filter_response_map_2 = torch.exp(scale * (rough_filter_response_map_2 - threshold))
#     rough_filter_response_map_2 = rough_filter_response_map_2 / torch.sum(
#         rough_filter_response_map_2,
#         dim=(2, 3),
#         keepdim=True)
#     # Sampled texture matcher feature vectors
#     fine_source_feature_1D_locations_2 = selected_detected_1D_locations_2.long().cuda(gpu_id).view(
#         1, 1, -1).expand(-1, fine_feature_length, -1)
#     sampled_fine_feature_vectors_2 = torch.gather(
#         fine_feature_map_2.view(1, fine_feature_length, height * width), 2,
#         fine_source_feature_1D_locations_2.long())
#     sampled_fine_feature_vectors_2 = sampled_fine_feature_vectors_2.view(1, fine_feature_length,
#                                                                          keypoint_number_2, 1,
#                                                                          1).permute(0, 2, 1, 3,
#                                                                                     4).view(
#         1, keypoint_number_2,
#         fine_feature_length,
#         1, 1)
#     fine_filter_response_map_2 = torch.nn.functional.conv2d(
#         input=fine_feature_map_1.view(1, fine_feature_length, height, width),
#         weight=sampled_fine_feature_vectors_2.view(keypoint_number_2,
#                                                    fine_feature_length,
#                                                    1, 1), padding=0)
#     # 1 x Sampling_size x H x W
#     fine_filter_response_map_2 = 0.5 * fine_filter_response_map_2 + 0.5
#     fine_filter_response_map_2 = torch.exp(
#         scale * (fine_filter_response_map_2 - threshold)) * boundary.view(1, 1, height,
#                                                                           width).expand(
#         -1, keypoint_number_2, -1, -1)
#     fine_filter_response_map_2 = fine_filter_response_map_2 / torch.sum(fine_filter_response_map_2,
#                                                                         dim=(2, 3), keepdim=True)
#
#     merged_response_map_2 = rough_filter_response_map_2 * fine_filter_response_map_2
#     max_reponses_2, max_indexes_2 = torch.max(merged_response_map_2.view(keypoint_number_2, -1),
#                                               dim=1,
#                                               keepdim=False)
#
#     keypoint_1D_locations_1 = torch.from_numpy(np.asarray(kps_1D_1)).float().cuda(gpu_id).view(
#         keypoint_number, 1)
#     keypoint_2D_locations_1 = torch.cat(
#         [torch.fmod(keypoint_1D_locations_1, width),
#          torch.floor(keypoint_1D_locations_1 / width)],
#         dim=1).view(keypoint_number, 2).float()
#
#     detected_keypoint_1D_locations_1 = max_indexes_2.float().view(keypoint_number, 1)
#     detected_keypoint_2D_locations_1 = torch.cat(
#         [torch.fmod(detected_keypoint_1D_locations_1, width),
#          torch.floor(detected_keypoint_1D_locations_1 / width)],
#         dim=1).view(keypoint_number, 2).float()
#
#     # We will accept the feature matches if the max indexes here is not far away from the original key point location from ORB
#     cross_check_correspondence_distances = torch.norm(
#         keypoint_2D_locations_1 - detected_keypoint_2D_locations_1, dim=1, p=2).view(
#         keypoint_number)
#     valid_correspondence_indexes = torch.nonzero(cross_check_correspondence_distances < cross_check_distance).view(
#         -1)
#
#     if valid_correspondence_indexes.shape[0] == 0:
#         return None
#
#     valid_detected_1D_locations_2 = torch.gather(selected_detected_1D_locations_2.long().view(-1),
#                                                  0, valid_correspondence_indexes.long())
#     valid_max_responses = torch.gather(selected_max_responses.view(-1), 0,
#                                        valid_correspondence_indexes.long())
#
#     valid_detected_1D_locations_2_cpu = valid_detected_1D_locations_2.data.cpu().numpy()
#     valid_max_responses_cpu = valid_max_responses.data.cpu().numpy()
#     valid_correspondence_indexes_cpu = valid_correspondence_indexes.data.cpu().numpy()
#
#     if display_matches:
#         detected_keypoints_2 = []
#         for index in valid_detected_1D_locations_2_cpu:
#             detected_keypoints_2.append(
#                 cv2.KeyPoint(x=float(np.floor(index % width)), y=float(np.floor(index / width)), _size=1.0))
#
#         matches = []
#         for idx, (query_index, response) in enumerate(
#                 zip(valid_correspondence_indexes_cpu, valid_max_responses_cpu)):
#             matches.append(cv2.DMatch(_queryIdx=query_index, _trainIdx=idx, _distance=response))
#         matches = sorted(matches, key=lambda x: x.distance, reverse=True)
#
#         color_1 = np.moveaxis(color_1, source=[0, 1, 2], destination=[2, 0, 1])
#         color_2 = np.moveaxis(color_2, source=[0, 1, 2], destination=[2, 0, 1])
#
#         # Extract corner points
#         color_1 = cv2.cvtColor(np.uint8(255 * (color_1 * 0.5 + 0.5)), cv2.COLOR_HSV2BGR_FULL)
#         color_2 = cv2.cvtColor(np.uint8(255 * (color_2 * 0.5 + 0.5)), cv2.COLOR_HSV2BGR_FULL)
#
#         display_matches_ai = cv2.drawMatches(color_1, kps_1, color_2, detected_keypoints_2, matches[:display_number],
#                                              flags=2, outImg=None)
#         bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#         orb_matches = bf.knnMatch(des_1, des_2, k=1)
#         good = []
#         for m in orb_matches:
#             if len(m) != 0:
#                 good.append(m[0])
#
#         good = sorted(good, key=lambda x: x.distance)
#         display_matches_orb = cv2.drawMatches(color_1, kps_1, color_2, kps_2, good[:display_number], flags=2,
#                                               outImg=None)
#         return display_matches_ai, display_matches_orb
#     else:
#         detected_keypoints_2 = np.zeros((len(valid_max_responses_cpu), 2))
#         for idx, index in enumerate(valid_detected_1D_locations_2_cpu):
#             detected_keypoints_2[idx][0] = float(np.floor(index % width))
#             detected_keypoints_2[idx][1] = float(np.floor(index / width))
#
#         keypoints_1 = np.zeros((len(kps_1D_1), 2))
#         for idx, index in enumerate(kps_1D_1):
#             keypoints_1[idx][0] = float(np.floor(index % width))
#             keypoints_1[idx][1] = float(np.floor(index / width))
#
#         matches = np.zeros((len(valid_max_responses_cpu), 3))
#         for idx, (query_index, response) in enumerate(
#                 zip(valid_correspondence_indexes_cpu, valid_max_responses_cpu)):
#             matches[idx][0] = query_index
#             matches[idx][1] = idx
#             matches[idx][2] = response
#         return matches, keypoints_1, detected_keypoints_2

# def feature_matching(color_1, color_2, rough_feature_map_1, rough_feature_map_2, fine_feature_map_1, fine_feature_map_2,
#                      boundary, kps_1D_1, des_1, des_2,
#                      scale, threshold, cross_check_distance,
#                      kps_1, kps_2, display_matches, gpu_id=0):
#     with torch.no_grad():
#         # Color image 3 x H x W
#         # Feature map C x H x W
#         rough_feature_length, height, width = rough_feature_map_1.shape
#         fine_feature_length, height, width = fine_feature_map_1.shape
#
#         # Extend 1D locations to B x C x Sampling_size
#         keypoint_number = len(kps_1D_1)
#         rough_source_feature_1D_locations = torch.from_numpy(kps_1D_1).long().cuda(gpu_id).view(
#             1, 1,
#             keypoint_number).expand(
#             -1, rough_feature_length, -1)
#
#         # Sampled rough locator feature vectors
#         sampled_rough_feature_vectors = torch.gather(
#             rough_feature_map_1.view(1, rough_feature_length, height * width), 2,
#             rough_source_feature_1D_locations.long())
#         sampled_rough_feature_vectors = sampled_rough_feature_vectors.view(1, rough_feature_length,
#                                                                            keypoint_number,
#                                                                            1,
#                                                                            1).permute(0, 2, 1, 3,
#                                                                                       4).view(1,
#                                                                                               keypoint_number,
#                                                                                               rough_feature_length,
#                                                                                               1, 1)
#
#         rough_filter_response_map = torch.nn.functional.conv2d(
#             input=rough_feature_map_2.view(1, rough_feature_length, height, width),
#             weight=sampled_rough_feature_vectors.view(keypoint_number,
#                                                       rough_feature_length,
#                                                       1, 1), padding=0)
#
#         # 1 x Sampling_size x H x W
#         rough_filter_response_map = 0.5 * rough_filter_response_map + 0.5
#         rough_filter_response_map = torch.exp(scale * (rough_filter_response_map - threshold))
#         rough_filter_response_map = rough_filter_response_map / torch.sum(rough_filter_response_map,
#                                                                           dim=(2, 3),
#                                                                           keepdim=True)
#         # Cleaning used variables to save space
#         del sampled_rough_feature_vectors
#         del rough_source_feature_1D_locations
#
#         # # Sampled texture matcher feature vectors
#         fine_source_feature_1D_locations = torch.from_numpy(kps_1D_1).cuda(gpu_id).long().view(1, 1,
#                                                                                                keypoint_number).expand(
#             -1, fine_feature_length, -1)
#         sampled_fine_feature_vectors = torch.gather(
#             fine_feature_map_1.view(1, fine_feature_length, height * width), 2,
#             fine_source_feature_1D_locations.long())
#         sampled_fine_feature_vectors = sampled_fine_feature_vectors.view(1, fine_feature_length,
#                                                                          keypoint_number, 1,
#                                                                          1).permute(0, 2, 1, 3, 4).view(
#             1, keypoint_number,
#             fine_feature_length,
#             1, 1)
#         fine_filter_response_map = torch.nn.functional.conv2d(
#             input=fine_feature_map_2.view(1, fine_feature_length, height, width),
#             weight=sampled_fine_feature_vectors.view(keypoint_number,
#                                                      fine_feature_length,
#                                                      1, 1), padding=0)
#         # 1 x Sampling_size x H x W
#         fine_filter_response_map = 0.5 * fine_filter_response_map + 0.5
#         fine_filter_response_map = torch.exp(
#             scale * (fine_filter_response_map - threshold)) * boundary.view(1, 1, height, width).expand(
#             -1, keypoint_number, -1, -1)
#         fine_filter_response_map = fine_filter_response_map / torch.sum(fine_filter_response_map,
#                                                                         dim=(2, 3), keepdim=True)
#
#         # Cleaning used variables to save space
#         del fine_source_feature_1D_locations
#         del sampled_fine_feature_vectors
#
#         merged_response_map = rough_filter_response_map * fine_filter_response_map
#         max_reponses, max_indexes = torch.max(merged_response_map.view(keypoint_number, -1), dim=1,
#                                               keepdim=False)
#
#         # Cleaning used variables to save space
#         del rough_filter_response_map
#         del fine_filter_response_map
#
#         # query is 1 and train is 2 here
#         selected_detected_1D_locations_2 = max_indexes.view(-1)
#         selected_max_responses = max_reponses.view(-1)
#         # Do cross check
#         rough_feature_1D_locations_2 = selected_detected_1D_locations_2.long().view(
#             1, 1, -1).expand(-1, rough_feature_length, -1)
#         keypoint_number_2 = keypoint_number
#
#         # Sampled rough locator feature vectors
#         sampled_rough_feature_vectors_2 = torch.gather(
#             rough_feature_map_2.view(1, rough_feature_length, height * width), 2,
#             rough_feature_1D_locations_2.long())
#         sampled_rough_feature_vectors_2 = sampled_rough_feature_vectors_2.view(1, rough_feature_length,
#                                                                                keypoint_number_2,
#                                                                                1,
#                                                                                1).permute(0, 2, 1, 3,
#                                                                                           4).view(1,
#                                                                                                   keypoint_number_2,
#                                                                                                   rough_feature_length,
#                                                                                                   1, 1)
#
#         rough_filter_response_map_2 = torch.nn.functional.conv2d(
#             input=rough_feature_map_1.view(1, rough_feature_length, height, width),
#             weight=sampled_rough_feature_vectors_2.view(keypoint_number_2,
#                                                         rough_feature_length,
#                                                         1, 1), padding=0)
#
#         # 1 x Sampling_size x H x W
#         rough_filter_response_map_2 = 0.5 * rough_filter_response_map_2 + 0.5
#         rough_filter_response_map_2 = torch.exp(scale * (rough_filter_response_map_2 - threshold))
#         rough_filter_response_map_2 = rough_filter_response_map_2 / torch.sum(
#             rough_filter_response_map_2,
#             dim=(2, 3),
#             keepdim=True)
#
#         del rough_feature_1D_locations_2
#         del sampled_rough_feature_vectors_2
#
#         # Sampled texture matcher feature vectors
#         fine_source_feature_1D_locations_2 = selected_detected_1D_locations_2.long().cuda(gpu_id).view(
#             1, 1, -1).expand(-1, fine_feature_length, -1)
#         sampled_fine_feature_vectors_2 = torch.gather(
#             fine_feature_map_2.view(1, fine_feature_length, height * width), 2,
#             fine_source_feature_1D_locations_2.long())
#         sampled_fine_feature_vectors_2 = sampled_fine_feature_vectors_2.view(1, fine_feature_length,
#                                                                              keypoint_number_2, 1,
#                                                                              1).permute(0, 2, 1, 3,
#                                                                                         4).view(
#             1, keypoint_number_2,
#             fine_feature_length,
#             1, 1)
#         fine_filter_response_map_2 = torch.nn.functional.conv2d(
#             input=fine_feature_map_1.view(1, fine_feature_length, height, width),
#             weight=sampled_fine_feature_vectors_2.view(keypoint_number_2,
#                                                        fine_feature_length,
#                                                        1, 1), padding=0)
#         # 1 x Sampling_size x H x W
#         fine_filter_response_map_2 = 0.5 * fine_filter_response_map_2 + 0.5
#         fine_filter_response_map_2 = torch.exp(
#             scale * (fine_filter_response_map_2 - threshold)) * boundary.view(1, 1, height,
#                                                                               width).expand(
#             -1, keypoint_number_2, -1, -1)
#         fine_filter_response_map_2 = fine_filter_response_map_2 / torch.sum(fine_filter_response_map_2,
#                                                                             dim=(2, 3), keepdim=True)
#
#         del fine_source_feature_1D_locations_2
#         del sampled_fine_feature_vectors_2
#
#         merged_response_map_2 = rough_filter_response_map_2 * fine_filter_response_map_2
#         max_reponses_2, max_indexes_2 = torch.max(merged_response_map_2.view(keypoint_number_2, -1),
#                                                   dim=1,
#                                                   keepdim=False)
#
#         del rough_filter_response_map_2
#         del fine_filter_response_map_2
#
#         keypoint_1D_locations_1 = torch.from_numpy(np.asarray(kps_1D_1)).float().cuda(gpu_id).view(
#             keypoint_number, 1)
#         keypoint_2D_locations_1 = torch.cat(
#             [torch.fmod(keypoint_1D_locations_1, width),
#              torch.floor(keypoint_1D_locations_1 / width)],
#             dim=1).view(keypoint_number, 2).float()
#
#         detected_keypoint_1D_locations_1 = max_indexes_2.float().view(keypoint_number, 1)
#         detected_keypoint_2D_locations_1 = torch.cat(
#             [torch.fmod(detected_keypoint_1D_locations_1, width),
#              torch.floor(detected_keypoint_1D_locations_1 / width)],
#             dim=1).view(keypoint_number, 2).float()
#
#         # We will accept the feature matches if the max indexes here is not far away from the original key point location from ORB
#         cross_check_correspondence_distances = torch.norm(
#             keypoint_2D_locations_1 - detected_keypoint_2D_locations_1, dim=1, p=2).view(
#             keypoint_number)
#         valid_correspondence_indexes = torch.nonzero(cross_check_correspondence_distances < cross_check_distance).view(
#             -1)
#
#         if valid_correspondence_indexes.shape[0] == 0:
#             return None
#
#         valid_detected_1D_locations_2 = torch.gather(selected_detected_1D_locations_2.long().view(-1),
#                                                      0, valid_correspondence_indexes.long())
#         valid_max_responses = torch.gather(selected_max_responses.view(-1), 0,
#                                            valid_correspondence_indexes.long())
#
#         valid_detected_1D_locations_2 = valid_detected_1D_locations_2.data.cpu().numpy()
#         valid_max_responses = valid_max_responses.data.cpu().numpy()
#         valid_correspondence_indexes = valid_correspondence_indexes.data.cpu().numpy()
#
#         if display_matches:
#             detected_keypoints_2 = []
#             for index in valid_detected_1D_locations_2:
#                 detected_keypoints_2.append(
#                     cv2.KeyPoint(x=float(np.floor(index % width)), y=float(np.floor(index / width)), _size=1.0))
#
#             matches = []
#             for idx, (query_index, response) in enumerate(
#                     zip(valid_correspondence_indexes, valid_max_responses)):
#                 matches.append(cv2.DMatch(_queryIdx=query_index, _trainIdx=idx, _distance=response))
#             # matches = sorted(matches, key=lambda x: x.distance, reverse=True)
#
#             color_1 = np.moveaxis(color_1, source=[0, 1, 2], destination=[2, 0, 1])
#             color_2 = np.moveaxis(color_2, source=[0, 1, 2], destination=[2, 0, 1])
#
#             # Extract corner points
#             color_1 = cv2.cvtColor(np.uint8(255 * (color_1 * 0.5 + 0.5)), cv2.COLOR_HSV2BGR_FULL)
#             color_2 = cv2.cvtColor(np.uint8(255 * (color_2 * 0.5 + 0.5)), cv2.COLOR_HSV2BGR_FULL)
#
#             display_matches_ai = cv2.drawMatches(color_1, kps_1, color_2, detected_keypoints_2, matches,
#                                                  flags=2, outImg=None)
#             bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#             orb_matches = bf.knnMatch(des_1, des_2, k=1)
#             good = []
#             for m in orb_matches:
#                 if len(m) != 0:
#                     good.append(m[0])
#
#             # good = sorted(good, key=lambda x: x.distance)
#             display_matches_orb = cv2.drawMatches(color_1, kps_1, color_2, kps_2, good, flags=2,
#                                                   outImg=None)
#             return display_matches_ai, display_matches_orb
#         else:
#             detected_keypoints_2 = np.zeros((len(valid_max_responses), 2))
#             for idx, index in enumerate(valid_detected_1D_locations_2):
#                 detected_keypoints_2[idx][0] = float(np.floor(index % width))
#                 detected_keypoints_2[idx][1] = float(np.floor(index / width))
#
#             keypoints_1 = np.zeros((len(kps_1D_1), 2))
#             for idx, index in enumerate(kps_1D_1):
#                 keypoints_1[idx][0] = float(np.floor(index % width))
#                 keypoints_1[idx][1] = float(np.floor(index / width))
#
#             matches = np.zeros((len(valid_max_responses), 3))
#             for idx, (query_index, response) in enumerate(
#                     zip(valid_correspondence_indexes, valid_max_responses)):
#                 matches[idx][0] = query_index
#                 matches[idx][1] = idx
#                 matches[idx][2] = response
#             return matches, keypoints_1, detected_keypoints_2
#
#     #     detected_keypoints_2 = []
#     #     for index in valid_detected_1D_locations_2:
#     #         detected_keypoints_2.append(
#     #             cv2.KeyPoint(x=float(np.floor(index % width)), y=float(np.floor(index / width)), _size=1.0))
#     #
#     #     matches = []
#     #     for i, (query_index, response) in enumerate(
#     #             zip(valid_correspondence_indexes, valid_max_responses)):
#     #         matches.append(cv2.DMatch(_queryIdx=query_index, _trainIdx=i, _distance=response))
#     #     matches = sorted(matches, key=lambda x: x.distance, reverse=True)
#     #
#     #     color_1 = np.moveaxis(color_1, source=[0, 1, 2], destination=[2, 0, 1])
#     #     color_2 = np.moveaxis(color_2, source=[0, 1, 2], destination=[2, 0, 1])
#     #
#     #     # Extract corner points
#     #     color_1 = np.uint8(255 * (color_1 * 0.5 + 0.5))
#     #     color_2 = np.uint8(255 * (color_2 * 0.5 + 0.5))
#     #
#     #     display_matches_ai = cv2.drawMatches(color_1, kps_1, color_2, detected_keypoints_2, matches,
#     #                                          flags=2,
#     #                                          outImg=None)
#     #     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     #     orb_matches = bf.knnMatch(des_1, des_2, k=1)
#     #     good = []
#     #     for m in orb_matches:
#     #         if len(m) != 0:
#     #             good.append(m[0])
#     #
#     #     good = sorted(good, key=lambda x: x.distance)
#     #     display_matches_orb = cv2.drawMatches(color_1, kps_1, color_2, kps_2, good, flags=2,
#     #                                           outImg=None)
#     # return display_matches_ai, display_matches_orb
#

def feature_matching_no_display(rough_feature_map_1, rough_feature_map_2, fine_feature_map_1, fine_feature_map_2,
                                boundary, kps_1D_1, scale, threshold, cross_check_distance, gpu_id):
    # Color image 3 x H x W
    # Feature map C x H x W
    rough_feature_length, height, width = rough_feature_map_1.shape
    fine_feature_length, height, width = fine_feature_map_1.shape

    # Extend 1D locations to B x C x Sampling_size
    keypoint_number = len(kps_1D_1)
    rough_source_feature_1D_locations = torch.from_numpy(kps_1D_1).long().cuda(gpu_id).view(
        1, 1,
        keypoint_number).expand(
        -1, rough_feature_length, -1)

    # Sampled rough locator feature vectors
    sampled_rough_feature_vectors = torch.gather(
        rough_feature_map_1.view(1, rough_feature_length, height * width), 2,
        rough_source_feature_1D_locations.long())
    sampled_rough_feature_vectors = sampled_rough_feature_vectors.view(1, rough_feature_length,
                                                                       keypoint_number,
                                                                       1,
                                                                       1).permute(0, 2, 1, 3,
                                                                                  4).view(1,
                                                                                          keypoint_number,
                                                                                          rough_feature_length,
                                                                                          1, 1)

    rough_filter_response_map = torch.nn.functional.conv2d(
        input=rough_feature_map_2.view(1, rough_feature_length, height, width),
        weight=sampled_rough_feature_vectors.view(keypoint_number,
                                                  rough_feature_length,
                                                  1, 1), padding=0)

    # 1 x Sampling_size x H x W
    rough_filter_response_map = 0.5 * rough_filter_response_map + 0.5
    rough_filter_response_map = torch.exp(scale * (rough_filter_response_map - threshold))
    rough_filter_response_map = rough_filter_response_map / torch.sum(rough_filter_response_map,
                                                                      dim=(2, 3),
                                                                      keepdim=True)

    # # Sampled texture matcher feature vectors
    fine_source_feature_1D_locations = torch.from_numpy(kps_1D_1).cuda(gpu_id).long().view(1, 1,
                                                                                           keypoint_number).expand(
        -1, fine_feature_length, -1)
    sampled_fine_feature_vectors = torch.gather(
        fine_feature_map_1.view(1, fine_feature_length, height * width), 2,
        fine_source_feature_1D_locations.long())
    sampled_fine_feature_vectors = sampled_fine_feature_vectors.view(1, fine_feature_length,
                                                                     keypoint_number, 1,
                                                                     1).permute(0, 2, 1, 3, 4).view(
        1, keypoint_number,
        fine_feature_length,
        1, 1)
    fine_filter_response_map = torch.nn.functional.conv2d(
        input=fine_feature_map_2.view(1, fine_feature_length, height, width),
        weight=sampled_fine_feature_vectors.view(keypoint_number,
                                                 fine_feature_length,
                                                 1, 1), padding=0)
    # 1 x Sampling_size x H x W
    fine_filter_response_map = 0.5 * fine_filter_response_map + 0.5
    fine_filter_response_map = torch.exp(
        scale * (fine_filter_response_map - threshold)) * boundary.view(1, 1, height, width).expand(
        -1, keypoint_number, -1, -1)
    fine_filter_response_map = fine_filter_response_map / torch.sum(fine_filter_response_map,
                                                                    dim=(2, 3), keepdim=True)

    merged_response_map = rough_filter_response_map * fine_filter_response_map
    max_reponses, max_indexes = torch.max(merged_response_map.view(keypoint_number, -1), dim=1,
                                          keepdim=False)
    # query is 1 and train is 2 here
    selected_detected_1D_locations_2 = max_indexes.view(-1)
    selected_max_responses = max_reponses.view(-1)
    # Do cross check
    rough_feature_1D_locations_2 = selected_detected_1D_locations_2.long().view(
        1, 1, -1).expand(-1, rough_feature_length, -1)
    keypoint_number_2 = keypoint_number

    # Sampled rough locator feature vectors
    sampled_rough_feature_vectors_2 = torch.gather(
        rough_feature_map_2.view(1, rough_feature_length, height * width), 2,
        rough_feature_1D_locations_2.long())
    sampled_rough_feature_vectors_2 = sampled_rough_feature_vectors_2.view(1, rough_feature_length,
                                                                           keypoint_number_2,
                                                                           1,
                                                                           1).permute(0, 2, 1, 3,
                                                                                      4).view(1,
                                                                                              keypoint_number_2,
                                                                                              rough_feature_length,
                                                                                              1, 1)

    rough_filter_response_map_2 = torch.nn.functional.conv2d(
        input=rough_feature_map_1.view(1, rough_feature_length, height, width),
        weight=sampled_rough_feature_vectors_2.view(keypoint_number_2,
                                                    rough_feature_length,
                                                    1, 1), padding=0)

    # 1 x Sampling_size x H x W
    rough_filter_response_map_2 = 0.5 * rough_filter_response_map_2 + 0.5
    rough_filter_response_map_2 = torch.exp(scale * (rough_filter_response_map_2 - threshold))
    rough_filter_response_map_2 = rough_filter_response_map_2 / torch.sum(
        rough_filter_response_map_2,
        dim=(2, 3),
        keepdim=True)
    # Sampled texture matcher feature vectors
    fine_source_feature_1D_locations_2 = selected_detected_1D_locations_2.long().cuda(gpu_id).view(
        1, 1, -1).expand(-1, fine_feature_length, -1)
    sampled_fine_feature_vectors_2 = torch.gather(
        fine_feature_map_2.view(1, fine_feature_length, height * width), 2,
        fine_source_feature_1D_locations_2.long())
    sampled_fine_feature_vectors_2 = sampled_fine_feature_vectors_2.view(1, fine_feature_length,
                                                                         keypoint_number_2, 1,
                                                                         1).permute(0, 2, 1, 3,
                                                                                    4).view(
        1, keypoint_number_2,
        fine_feature_length,
        1, 1)
    fine_filter_response_map_2 = torch.nn.functional.conv2d(
        input=fine_feature_map_1.view(1, fine_feature_length, height, width),
        weight=sampled_fine_feature_vectors_2.view(keypoint_number_2,
                                                   fine_feature_length,
                                                   1, 1), padding=0)
    # 1 x Sampling_size x H x W
    fine_filter_response_map_2 = 0.5 * fine_filter_response_map_2 + 0.5
    fine_filter_response_map_2 = torch.exp(
        scale * (fine_filter_response_map_2 - threshold)) * boundary.view(1, 1, height,
                                                                          width).expand(
        -1, keypoint_number_2, -1, -1)
    fine_filter_response_map_2 = fine_filter_response_map_2 / torch.sum(fine_filter_response_map_2,
                                                                        dim=(2, 3), keepdim=True)

    merged_response_map_2 = rough_filter_response_map_2 * fine_filter_response_map_2
    max_reponses_2, max_indexes_2 = torch.max(merged_response_map_2.view(keypoint_number_2, -1),
                                              dim=1,
                                              keepdim=False)

    keypoint_1D_locations_1 = torch.from_numpy(np.asarray(kps_1D_1)).float().cuda(gpu_id).view(
        keypoint_number, 1)
    keypoint_2D_locations_1 = torch.cat(
        [torch.fmod(keypoint_1D_locations_1, width),
         torch.floor(keypoint_1D_locations_1 / width)],
        dim=1).view(keypoint_number, 2).float()

    detected_keypoint_1D_locations_1 = max_indexes_2.float().view(keypoint_number, 1)
    detected_keypoint_2D_locations_1 = torch.cat(
        [torch.fmod(detected_keypoint_1D_locations_1, width),
         torch.floor(detected_keypoint_1D_locations_1 / width)],
        dim=1).view(keypoint_number, 2).float()

    # We will accept the feature matches if the max indexes here is not far away from the original key point location from ORB
    cross_check_correspondence_distances = torch.norm(
        keypoint_2D_locations_1 - detected_keypoint_2D_locations_1, dim=1, p=2).view(
        keypoint_number)
    valid_query_indexes = torch.nonzero(cross_check_correspondence_distances < cross_check_distance).view(
        -1)

    if valid_query_indexes.shape[0] == 0:
        return None

    valid_detected_train_1D_locations = torch.gather(selected_detected_1D_locations_2.long().view(-1),
                                                     0, valid_query_indexes.long())
    valid_max_responses = torch.gather(selected_max_responses.view(-1), 0,
                                       valid_query_indexes.long())

    return valid_query_indexes, valid_detected_train_1D_locations, valid_max_responses
    # valid_detected_1D_locations_2_cpu = valid_detected_1D_locations_2.data.cpu().numpy()
    # valid_max_responses_cpu = valid_max_responses.data.cpu().numpy()
    # valid_query_indexes_cpu = valid_query_indexes.data.cpu().numpy()

    # detected_keypoints_2 = np.zeros((len(valid_max_responses_cpu), 2))
    # for i, index in enumerate(valid_detected_1D_locations_2_cpu):
    #     detected_keypoints_2[i][0] = float(np.floor(index % width))
    #     detected_keypoints_2[i][1] = float(np.floor(index / width))
    #
    # keypoints_1 = np.zeros((len(kps_1D_1), 2))
    # for i, index in enumerate(kps_1D_1):
    #     keypoints_1[i][0] = float(np.floor(index % width))
    #     keypoints_1[i][1] = float(np.floor(index / width))
    #
    # matches = np.zeros((len(valid_max_responses_cpu), 3))
    # for i, (query_index, response) in enumerate(
    #         zip(valid_query_indexes_cpu, valid_max_responses_cpu)):
    #     matches[i][0] = query_index
    #     matches[i][1] = i
    #     matches[i][2] = response
    # return matches, keypoints_1, detected_keypoints_2


def feature_matching_cpu(color_1, color_2, rough_feature_map_1, rough_feature_map_2, fine_feature_map_1,
                         fine_feature_map_2,
                         boundary, kps_1, kps_2, kps_1D_1, des_1, des_2,
                         scale, threshold, display_number, cross_check_distance, display_matches=False):
    # Color image 3 x H x W
    # Feature map C x H x W
    rough_feature_length, height, width = rough_feature_map_1.shape
    fine_feature_length, height, width = fine_feature_map_1.shape

    # kernel = np.ones((10, 10), np.uint8)
    # boundary = boundary.view(height, width).data.cpu().numpy()
    # boundary = cv2.erode(boundary, kernel, iterations=2)
    # boundary = torch.from_numpy(boundary).float().cuda()
    # indexes_1 = []
    # indexes_2 = []

    # Extend 1D locations to B x C x Sampling_size
    keypoint_number = len(kps_1D_1)
    rough_source_feature_1D_locations = torch.from_numpy(kps_1D_1).long().view(
        1, 1,
        keypoint_number).expand(
        -1, rough_feature_length, -1)

    # Sampled rough locator feature vectors
    sampled_rough_feature_vectors = torch.gather(
        rough_feature_map_1.view(1, rough_feature_length, height * width), 2,
        rough_source_feature_1D_locations.long())
    sampled_rough_feature_vectors = sampled_rough_feature_vectors.view(1, rough_feature_length,
                                                                       keypoint_number,
                                                                       1,
                                                                       1).permute(0, 2, 1, 3,
                                                                                  4).view(1,
                                                                                          keypoint_number,
                                                                                          rough_feature_length,
                                                                                          1, 1)

    rough_filter_response_map = torch.nn.functional.conv2d(
        input=rough_feature_map_2.view(1, rough_feature_length, height, width),
        weight=sampled_rough_feature_vectors.view(keypoint_number,
                                                  rough_feature_length,
                                                  1, 1), padding=0)

    # 1 x Sampling_size x H x W
    rough_filter_response_map = 0.5 * rough_filter_response_map + 0.5
    rough_filter_response_map = torch.exp(scale * (rough_filter_response_map - threshold))
    rough_filter_response_map = rough_filter_response_map / torch.sum(rough_filter_response_map,
                                                                      dim=(2, 3),
                                                                      keepdim=True)

    # # Sampled texture matcher feature vectors
    fine_source_feature_1D_locations = torch.from_numpy(kps_1D_1).long().view(1, 1, keypoint_number).expand(
        -1, fine_feature_length, -1)
    sampled_fine_feature_vectors = torch.gather(
        fine_feature_map_1.view(1, fine_feature_length, height * width), 2,
        fine_source_feature_1D_locations.long())
    sampled_fine_feature_vectors = sampled_fine_feature_vectors.view(1, fine_feature_length,
                                                                     keypoint_number, 1,
                                                                     1).permute(0, 2, 1, 3, 4).view(
        1, keypoint_number,
        fine_feature_length,
        1, 1)
    fine_filter_response_map = torch.nn.functional.conv2d(
        input=fine_feature_map_2.view(1, fine_feature_length, height, width),
        weight=sampled_fine_feature_vectors.view(keypoint_number,
                                                 fine_feature_length,
                                                 1, 1), padding=0)
    # 1 x Sampling_size x H x W
    fine_filter_response_map = 0.5 * fine_filter_response_map + 0.5
    fine_filter_response_map = torch.exp(
        scale * (fine_filter_response_map - threshold)) * boundary.view(1, 1, height, width).expand(
        -1, keypoint_number, -1, -1)
    fine_filter_response_map = fine_filter_response_map / torch.sum(fine_filter_response_map,
                                                                    dim=(2, 3), keepdim=True)

    merged_response_map = rough_filter_response_map * fine_filter_response_map
    max_reponses, max_indexes = torch.max(merged_response_map.view(keypoint_number, -1), dim=1,
                                          keepdim=False)
    # query is 1 and train is 2 here
    selected_detected_1D_locations_2 = max_indexes.view(-1)
    selected_max_responses = max_reponses.view(-1)
    # Do cross check
    rough_feature_1D_locations_2 = selected_detected_1D_locations_2.long().view(
        1, 1, -1).expand(-1, rough_feature_length, -1)
    keypoint_number_2 = keypoint_number

    # Sampled rough locator feature vectors
    sampled_rough_feature_vectors_2 = torch.gather(
        rough_feature_map_2.view(1, rough_feature_length, height * width), 2,
        rough_feature_1D_locations_2.long())
    sampled_rough_feature_vectors_2 = sampled_rough_feature_vectors_2.view(1, rough_feature_length,
                                                                           keypoint_number_2,
                                                                           1,
                                                                           1).permute(0, 2, 1, 3,
                                                                                      4).view(1,
                                                                                              keypoint_number_2,
                                                                                              rough_feature_length,
                                                                                              1, 1)

    rough_filter_response_map_2 = torch.nn.functional.conv2d(
        input=rough_feature_map_1.view(1, rough_feature_length, height, width),
        weight=sampled_rough_feature_vectors_2.view(keypoint_number_2,
                                                    rough_feature_length,
                                                    1, 1), padding=0)

    # 1 x Sampling_size x H x W
    rough_filter_response_map_2 = 0.5 * rough_filter_response_map_2 + 0.5
    rough_filter_response_map_2 = torch.exp(scale * (rough_filter_response_map_2 - threshold))
    rough_filter_response_map_2 = rough_filter_response_map_2 / torch.sum(
        rough_filter_response_map_2,
        dim=(2, 3),
        keepdim=True)
    # Sampled texture matcher feature vectors
    fine_source_feature_1D_locations_2 = selected_detected_1D_locations_2.long().view(
        1, 1, -1).expand(-1, fine_feature_length, -1)
    sampled_fine_feature_vectors_2 = torch.gather(
        fine_feature_map_2.view(1, fine_feature_length, height * width), 2,
        fine_source_feature_1D_locations_2.long())
    sampled_fine_feature_vectors_2 = sampled_fine_feature_vectors_2.view(1, fine_feature_length,
                                                                         keypoint_number_2, 1,
                                                                         1).permute(0, 2, 1, 3,
                                                                                    4).view(
        1, keypoint_number_2,
        fine_feature_length,
        1, 1)
    fine_filter_response_map_2 = torch.nn.functional.conv2d(
        input=fine_feature_map_1.view(1, fine_feature_length, height, width),
        weight=sampled_fine_feature_vectors_2.view(keypoint_number_2,
                                                   fine_feature_length,
                                                   1, 1), padding=0)
    # 1 x Sampling_size x H x W
    fine_filter_response_map_2 = 0.5 * fine_filter_response_map_2 + 0.5
    fine_filter_response_map_2 = torch.exp(
        scale * (fine_filter_response_map_2 - threshold)) * boundary.view(1, 1, height,
                                                                          width).expand(
        -1, keypoint_number_2, -1, -1)
    fine_filter_response_map_2 = fine_filter_response_map_2 / torch.sum(fine_filter_response_map_2,
                                                                        dim=(2, 3), keepdim=True)

    merged_response_map_2 = rough_filter_response_map_2 * fine_filter_response_map_2
    max_reponses_2, max_indexes_2 = torch.max(merged_response_map_2.view(keypoint_number_2, -1),
                                              dim=1,
                                              keepdim=False)

    keypoint_1D_locations_1 = torch.from_numpy(np.asarray(kps_1D_1)).float().view(
        keypoint_number, 1)
    keypoint_2D_locations_1 = torch.cat(
        [torch.fmod(keypoint_1D_locations_1, width),
         torch.floor(keypoint_1D_locations_1 / width)],
        dim=1).view(keypoint_number, 2).float()

    detected_keypoint_1D_locations_1 = max_indexes_2.float().view(keypoint_number, 1)
    detected_keypoint_2D_locations_1 = torch.cat(
        [torch.fmod(detected_keypoint_1D_locations_1, width),
         torch.floor(detected_keypoint_1D_locations_1 / width)],
        dim=1).view(keypoint_number, 2).float()

    # TODO: We will accept the feature matches if the max indexes here is not far away from the original key point location from ORB
    cross_check_correspondence_distances = torch.norm(
        keypoint_2D_locations_1 - detected_keypoint_2D_locations_1, dim=1, p=2).view(
        keypoint_number)
    valid_correspondence_indexes = torch.nonzero(cross_check_correspondence_distances < cross_check_distance).view(
        -1)

    if valid_correspondence_indexes.shape[0] == 0:
        return None

    valid_detected_1D_locations_2 = torch.gather(selected_detected_1D_locations_2.long().view(-1),
                                                 0, valid_correspondence_indexes.long())
    valid_max_responses = torch.gather(selected_max_responses.view(-1), 0,
                                       valid_correspondence_indexes.long())

    valid_detected_1D_locations_2_cpu = valid_detected_1D_locations_2.data.cpu().numpy()
    valid_max_responses_cpu = valid_max_responses.data.cpu().numpy()
    valid_correspondence_indexes_cpu = valid_correspondence_indexes.data.cpu().numpy()

    detected_keypoints_2 = []
    for index in valid_detected_1D_locations_2_cpu:
        detected_keypoints_2.append(
            cv2.KeyPoint(x=float(np.floor(index % width)), y=float(np.floor(index / width)), _size=1.0))

    # matches = []
    # orb_keypoints_2, descriptions_2 = descriptor.compute(color_2, detected_keypoints_2)
    # # If all points have valid descriptions
    # distances = []
    #
    # if len(orb_keypoints_2) == len(detected_keypoints_2):
    #     for i, (query_index, response) in enumerate(
    #             zip(valid_correspondence_indexes_cpu, valid_max_responses_cpu)):
    #         distances.append(
    #             hamming_distance(np.asarray(des_1[query_index], dtype=np.int32), np.asarray(descriptions_2[i], dtype=np.int32)))
    #     mean_distance = np.mean(distances)
    #     for i, (query_index, response) in enumerate(
    #             zip(valid_correspondence_indexes_cpu, valid_max_responses_cpu)):
    #         distance = hamming_distance(np.asarray(des_1[query_index], dtype=np.int32), np.asarray(descriptions_2[i], dtype=np.int32))
    #         if distance < mean_distance:
    #             matches.append(cv2.DMatch(_trainIdx=i, _queryIdx=query_index, _distance=response))
    # else:
    matches = []
    for i, (query_index, response) in enumerate(
            zip(valid_correspondence_indexes_cpu, valid_max_responses_cpu)):
        matches.append(cv2.DMatch(_trainIdx=i, _queryIdx=query_index, _distance=response))
    matches = sorted(matches, key=lambda x: x.distance, reverse=True)

    if display_matches:
        color_1 = np.moveaxis(color_1, source=[0, 1, 2], destination=[2, 0, 1])
        color_2 = np.moveaxis(color_2, source=[0, 1, 2], destination=[2, 0, 1])

        # Extract corner points
        color_1 = cv2.cvtColor(np.uint8(255 * (color_1 * 0.5 + 0.5)), cv2.COLOR_HSV2BGR_FULL)
        color_2 = cv2.cvtColor(np.uint8(255 * (color_2 * 0.5 + 0.5)), cv2.COLOR_HSV2BGR_FULL)

        display_matches_ai = cv2.drawMatches(color_1, kps_1, color_2, detected_keypoints_2, matches[:display_number],
                                             flags=2,
                                             outImg=None)
        # for point in kps_2:
        #     indexes_2.append(np.round(point.pt[0]) + np.round(point.pt[1]) * width)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        orb_matches = bf.knnMatch(des_1, des_2, k=1)
        # Apply ratio test
        good = []
        for m in orb_matches:
            if len(m) != 0:
                good.append(m[0])

        good = sorted(good, key=lambda x: x.distance)
        display_matches_orb = cv2.drawMatches(color_1, kps_1, color_2, kps_2, good[:display_number], flags=2,
                                              outImg=None)
        return display_matches_ai, display_matches_orb
    else:
        return matches, kps_1, detected_keypoints_2


# sub_rough_feature_maps_list, sub_fine_feature_maps_list,
#                                  sub_keypoints_list_1D, boundary,
#                                  pairing_range, scale, threshold, cross_check_distance, gpu_id, end_sequence

def one_process_feature_matching(input_queue, output_queue, boundary, scale, threshold,
                                 cross_check_distance, done, id):
    # TODO: Output frame indexes
    # accum_count = 0
    # rough_feature_map_1_list = []
    # fine_feature_map_1_list = []
    # rough_feature_map_2_list = []
    # fine_feature_map_2_list = []
    # keypoints_1D_1_list = []
    # boundary_list = []
    # for i in range(4):
    #     boundary_list.append(torch.from_numpy(boundary).cuda(i))
    boundary = torch.from_numpy(boundary)
    while True:
        input_list = input_queue.get()
        if input_list is not None:
            print("Process {} get a tensor from the queue".format(id))
            rough_feature_map_1, fine_feature_map_1, keypoints_1D_1, rough_feature_map_2, \
            fine_feature_map_2, gpu_id = input_list
            valid_query_indexes, valid_detected_train_1D_locations, valid_max_responses = \
                feature_matching_no_display(rough_feature_map_1,
                                            rough_feature_map_2,
                                            fine_feature_map_1,
                                            fine_feature_map_2,
                                            boundary.cuda(rough_feature_map_1.device), keypoints_1D_1, scale, threshold,
                                            cross_check_distance, id)

            output_queue.put(valid_query_indexes)
            del input_list
            # accum_count += 1
            # rough_feature_map_1_list.append(torch.from_numpy(rough_feature_map_1).unsqueeze(dim=0))
            # rough_feature_map_2_list.append(torch.from_numpy(rough_feature_map_2).unsqueeze(dim=0))
            # fine_feature_map_1_list.append(torch.from_numpy(fine_feature_map_1).unsqueeze(dim=0))
            # fine_feature_map_2_list.append(torch.from_numpy(fine_feature_map_2).unsqueeze(dim=0))
            # keypoints_1D_1_list.append(keypoints_1D_1)

            # if accum_count >= 5:
            #     accum_count = 0
            # rough_feature_maps_1 = torch.cat(rough_feature_map_1_list, dim=0).cuda(id)
            # rough_feature_maps_2 = torch.cat(rough_feature_map_2_list, dim=0).cuda(id)
            # fine_feature_maps_1 = torch.cat(fine_feature_map_1_list, dim=0).cuda(id)
            # fine_feature_maps_2 = torch.cat(fine_feature_map_2_list, dim=0).cuda(id)
            # for i in range(rough_feature_maps_1.shape[0]):
            #     valid_query_indexes, valid_detected_train_1D_locations, valid_max_responses = \
            #         feature_matching_no_display(rough_feature_maps_1[i],
            #                                     rough_feature_maps_2[i],
            #                                     fine_feature_maps_1[i],
            #                                     fine_feature_maps_2[i],
            #                                     boundary, keypoints_1D_1_list[i], scale, threshold,
            #                                     cross_check_distance, id)
            #     output_queue.put(valid_query_indexes)
            # output_queue.put(rough_feature_map_1)
            # rough_feature_map_1_list.clear()
            # rough_feature_map_2_list.clear()
            # fine_feature_map_1_list.clear()
            # fine_feature_map_2_list.clear()
            # keypoints_1D_1_list.clear()

        else:
            print("Process {} get a None from the queue".format(id))
            output_queue.put(None)
            break

    print("Process {} waiting to complete...".format(id))
    done.wait()
    print("Process {} complete".format(id))
    return
    # print("GPU {}".format(gpu_id))
    # sys.stdout.flush()
    # if not end_sequence:
    #     query_frame_range = [0, len(sub_rough_feature_maps_list) - pairing_range]
    # else:
    #     query_frame_range = [0, len(sub_rough_feature_maps_list) - 1]
    # print(query_frame_range)
    # total_frame_count = len(sub_rough_feature_maps_list)
    # train_frame_increment_range = [1, pairing_range + 1]
    # rough_feature_length, _, _ = sub_rough_feature_maps_list.shape
    # fine_feature_length, height, width = sub_fine_feature_maps_list.shape
    # # Convert all lists to cuda tensors
    # sub_rough_feature_maps_list = torch.from_numpy(np.asarray(sub_rough_feature_maps_list)).cuda(gpu_id).view(-1,
    #                                                                                                           rough_feature_length,
    #                                                                                                           height,
    #                                                                                                           width)
    # sub_fine_feature_maps_list = torch.from_numpy(np.asarray(sub_fine_feature_maps_list)).cuda(gpu_id).view(-1,
    #                                                                                                         fine_feature_length,
    #                                                                                                         height,
    #                                                                                                         width)
    # boundary = torch.from_numpy(boundary).cuda(gpu_id)
    # if not end_sequence:
    #     for i in range(query_frame_range[0], query_frame_range[1]):
    #         print("GPU {}: processing frame {}".format(gpu_id, i))
    #         for j in range(train_frame_increment_range[0], train_frame_increment_range[1]):
    #             valid_query_indexes, valid_detected_train_1D_locations, valid_max_responses = \
    #                 feature_matching_no_display(rough_feature_map_1=sub_rough_feature_maps_list[i],
    #                                         rough_feature_map_2=sub_rough_feature_maps_list[i + j],
    #                                         fine_feature_map_1=sub_fine_feature_maps_list[i],
    #                                         fine_feature_map_2=sub_fine_feature_maps_list[i + j],
    #                                         boundary=boundary, kps_1D_1=np.array(sub_keypoints_list_1D[i]), scale=scale,
    #                                         threshold=threshold, cross_check_distance=cross_check_distance,
    #                                         gpu_id=gpu_id)
    # else:
    #     for i in range(query_frame_range[0], query_frame_range[1]):
    #         print("GPU {}: processing frame {}".format(gpu_id, i))
    #         for j in range(train_frame_increment_range[0], min(train_frame_increment_range[1], total_frame_count - i)):
    #             valid_query_indexes, valid_detected_train_1D_locations, valid_max_responses = \
    #                 feature_matching_no_display(rough_feature_map_1=sub_rough_feature_maps_list[i],
    #                                         rough_feature_map_2=sub_rough_feature_maps_list[i + j],
    #                                         fine_feature_map_1=sub_fine_feature_maps_list[i],
    #                                         fine_feature_map_2=sub_fine_feature_maps_list[i + j],
    #                                         boundary=boundary, kps_1D_1=np.array(sub_keypoints_list_1D[i]), scale=scale,
    #                                         threshold=threshold, cross_check_distance=cross_check_distance,
    #                                         gpu_id=gpu_id)
    # valid_query_indexes.data#, valid_detected_train_1D_locations, valid_max_responses


# TODO: Encounter the problem of efficiency, now this function is deprecated
def multi_processing_feature_matching(rough_feature_maps_list, fine_feature_maps_list,
                                      keypoints_list_1D, boundary,
                                      pairing_interval, scale, threshold, cross_check_distance,
                                      gpu_ids):
    torch.cuda.empty_cache()
    with torch.no_grad():
        ctx = mp.get_context("spawn")
        gpu_count = len(gpu_ids)
        worker_number = len(gpu_ids)
        feature_map_queue = ctx.Queue()
        result_queue = ctx.Queue()
        done = ctx.Event()

        process_pool = []
        for i in range(worker_number):
            print("Adding worker {} for processing...".format(i))
            process_pool.append(ctx.Process(target=one_process_feature_matching,
                                            args=(feature_map_queue, result_queue, boundary, scale, threshold,
                                                  cross_check_distance, done, i % gpu_count)))
        for i in range(worker_number):
            print("Starting worker {} for processing...".format(i))
            process_pool[i].start()

        query_size_per_gpu = int(np.floor(len(rough_feature_maps_list) / gpu_count) + 1)
        train_size_per_gpu = query_size_per_gpu + pairing_interval

        # Allocate GPU memory to data list
        rough_feature_maps_list_per_gpu = []
        fine_feature_maps_list_per_gpu = []
        keypoints_1D_list_per_gpu = []
        frame_indexes_list_per_gpu = []

        for gpu_idx in range(gpu_count):
            rough_feature_maps_list_per_gpu.append([])
            fine_feature_maps_list_per_gpu.append([])
            keypoints_1D_list_per_gpu.append([])
            frame_indexes_list_per_gpu.append([])

        for gpu_idx in range(gpu_count):
            start_idx = gpu_idx * query_size_per_gpu
            end_idx = np.minimum((gpu_idx + 1) * train_size_per_gpu, len(rough_feature_maps_list))
            for sample_idx in range(start_idx, end_idx):
                tensor = torch.from_numpy(rough_feature_maps_list[sample_idx]).cuda(gpu_idx)
                rough_feature_maps_list_per_gpu[gpu_idx].append(tensor)
                tensor = torch.from_numpy(fine_feature_maps_list[sample_idx]).cuda(gpu_idx)
                fine_feature_maps_list_per_gpu[gpu_idx].append(tensor)
                keypoints_1D_list_per_gpu[gpu_idx].append(np.asarray(keypoints_list_1D[sample_idx]))
                frame_indexes_list_per_gpu[gpu_idx].append(sample_idx)

                # tensor1_gpu = tensor1.cuda(gpu_idx).detach()

                # rough_feature_maps_list_per_gpu[gpu_idx].append(
                #     torch.from_numpy(rough_feature_maps_list[sample_idx]).cuda(gpu_idx).detach())
                # fine_feature_maps_list_per_gpu[gpu_idx].append(
                #     torch.from_numpy(fine_feature_maps_list[sample_idx]).cuda(gpu_idx).detach())

        scan_status_per_gpu = np.zeros((gpu_count, 2), dtype=np.int32)
        last_chunk_size = len(rough_feature_maps_list_per_gpu[gpu_count - 1])
        prev_chunk_size = len(rough_feature_maps_list_per_gpu[0])
        for gpu_idx in range(scan_status_per_gpu.shape[0]):
            scan_status_per_gpu[gpu_idx][0] = 0
            scan_status_per_gpu[gpu_idx][1] = 1

        job_count = 0
        complete_count = 0
        is_completed = np.zeros((gpu_count,), dtype=np.int32)

        completed_gpu_indexes = []
        while True:
            for gpu_idx in range(gpu_count):
                if is_completed[gpu_idx] == 1:
                    if gpu_idx not in completed_gpu_indexes:
                        completed_gpu_indexes.append(gpu_idx)
                        complete_count += 1
                        if complete_count >= gpu_count:
                            break
                        else:
                            continue
                    else:
                        continue

                rough_feature_map_1 = rough_feature_maps_list_per_gpu[gpu_idx][scan_status_per_gpu[gpu_idx][0]]
                rough_feature_map_2 = rough_feature_maps_list_per_gpu[gpu_idx][scan_status_per_gpu[gpu_idx][0] +
                                                                               scan_status_per_gpu[gpu_idx][1]]
                fine_feature_map_1 = fine_feature_maps_list_per_gpu[gpu_idx][scan_status_per_gpu[gpu_idx][0]]
                fine_feature_map_2 = fine_feature_maps_list_per_gpu[gpu_idx][scan_status_per_gpu[gpu_idx][0] +
                                                                             scan_status_per_gpu[gpu_idx][1]]
                keypoints_1D_1 = keypoints_1D_list_per_gpu[gpu_idx][scan_status_per_gpu[gpu_idx][0]]
                frame_index_1 = frame_indexes_list_per_gpu[gpu_idx][scan_status_per_gpu[gpu_idx][0]]
                frame_index_2 = frame_indexes_list_per_gpu[gpu_idx][scan_status_per_gpu[gpu_idx][0] +
                                                                    scan_status_per_gpu[gpu_idx][1]]

                feature_map_queue.put([rough_feature_map_1,
                                       fine_feature_map_1,
                                       keypoints_1D_1,
                                       rough_feature_map_2,
                                       fine_feature_map_2, gpu_idx])
                job_count += 1
                if gpu_idx < gpu_count - 1:
                    if scan_status_per_gpu[gpu_idx][1] < pairing_interval:
                        scan_status_per_gpu[gpu_idx][1] += 1
                    else:
                        if scan_status_per_gpu[gpu_idx][0] + scan_status_per_gpu[gpu_idx][1] < prev_chunk_size - 1:
                            scan_status_per_gpu[gpu_idx][0] += 1
                            scan_status_per_gpu[gpu_idx][1] = 1
                        else:
                            assert (scan_status_per_gpu[gpu_idx][0] == query_size_per_gpu - 1)
                            is_completed[gpu_idx] = 1
                            continue
                else:
                    # The second one has reached the last sample in the list
                    if scan_status_per_gpu[gpu_idx][0] + scan_status_per_gpu[gpu_idx][1] < last_chunk_size - 1:
                        scan_status_per_gpu[gpu_idx][1] += 1
                    # There are still space for other pairs
                    elif scan_status_per_gpu[gpu_idx][1] > 1:
                        scan_status_per_gpu[gpu_idx][0] += 1
                        scan_status_per_gpu[gpu_idx][1] = 1
                    # The last chunk has reached the last pair, skip for the further iterations
                    else:
                        assert (scan_status_per_gpu[gpu_idx][0] == last_chunk_size - 2)
                        is_completed[gpu_idx] = 1
                        continue

            if complete_count >= gpu_count:
                break

        # Put signs to stop processing for each worker
        for i in range(worker_number):
            feature_map_queue.put(None)

        print("All job pushing finished")
        output_result_count = 0
        count = 0
        while True:
            a = result_queue.get()
            if a is not None:
                print("Main process get {} / {} item from {}".format(output_result_count, job_count, a.device))
                output_result_count += 1
                del a
            else:
                count += 1
                if count >= worker_number:
                    break

        print("Main process setting event object...")
        done.set()

        for i in range(worker_number):
            print("Process {} is joining...".format(i))
            process_pool[i].join()

        return


def one_to_all_feature_matching(query_frame_index, colors_list, rough_feature_maps_list, fine_feature_maps_list,
                                boundaries_list, orb_keypoints_list_1D,
                                orb_descriptions_list, pairing_interval, scale, threshold, cross_check_distance,
                                gpu_id):
    print("Processing {}...".format(query_frame_index))
    sys.stdout.flush()
    frame_number = len(colors_list)
    color_1 = colors_list[query_frame_index]
    rough_feature_map_1 = rough_feature_maps_list[query_frame_index]
    fine_feature_map_1 = fine_feature_maps_list[query_frame_index]
    boundary = boundaries_list[query_frame_index]
    orb_keypoint_1D_1 = orb_keypoints_list_1D[query_frame_index]
    orb_description_1 = orb_descriptions_list[query_frame_index]

    count = np.minimum(pairing_interval + 1, frame_number - query_frame_index)
    for increment in range(1, count):
        print("{} / {}".format(increment, count))
        train_frame_index = query_frame_index + increment
        color_2 = colors_list[train_frame_index]
        rough_feature_map_2 = rough_feature_maps_list[train_frame_index]
        fine_feature_map_2 = fine_feature_maps_list[train_frame_index]
        orb_description_2 = orb_descriptions_list[train_frame_index]

        matches, query_keypoints, train_keypoints = feature_matching(
            color_1=color_1,
            color_2=color_2,
            rough_feature_map_1=rough_feature_map_1,
            rough_feature_map_2=rough_feature_map_2,
            fine_feature_map_1=fine_feature_map_1,
            fine_feature_map_2=fine_feature_map_2,
            boundary=boundary,
            kps_1D_1=np.array(orb_keypoint_1D_1),
            des_1=orb_description_1,
            des_2=orb_description_2,
            scale=scale, threshold=threshold,
            display_number=0,
            cross_check_distance=cross_check_distance,
            display_matches=False,
            gpu_id=gpu_id)
    return matches, query_keypoints, train_keypoints, query_frame_index, train_frame_index


# def feature_matching_(descriptor, color_1, color_2, rough_feature_map_1, rough_feature_map_2, fine_feature_map_1,
#                      fine_feature_map_2,
#                      boundary, scale, threshold, display_number):
#
#     # Color image 3 x H x W
#     # Feature map C x H x W
#     rough_feature_length, height, width = rough_feature_map_1.shape
#     fine_feature_length, height, width = fine_feature_map_1.shape
#
#     color_1 = np.moveaxis(color_1.data.cpu().numpy(), source=[0, 1, 2], destination=[2, 0, 1])
#     color_2 = np.moveaxis(color_2.data.cpu().numpy(), source=[0, 1, 2], destination=[2, 0, 1])
#
#     # Extract corner points
#     color_1 = cv2.cvtColor(np.uint8(255 * (color_1 * 0.5 + 0.5)), cv2.COLOR_HSV2BGR_FULL)
#     color_2 = cv2.cvtColor(np.uint8(255 * (color_2 * 0.5 + 0.5)), cv2.COLOR_HSV2BGR_FULL)
#
#     cv2.imshow("1", color_1)
#     cv2.imshow("2", color_2)
#     cv2.waitKey()
#
#     kernel = np.ones((10, 10), np.uint8)
#     boundary = boundary.view(height, width).data.cpu().numpy()
#     boundary = cv2.erode(boundary, kernel, iterations=2)
#     kps_1, des_1 = descriptor.detectAndCompute(color_1, boundary)
#     kps_2, des_2 = descriptor.detectAndCompute(color_2, boundary)
#     boundary = torch.from_numpy(boundary).float().cuda()
#
#     # TODO: ORB traditional sparse matching
#     indexes_1 = []
#     indexes_2 = []
#     for point in kps_1:
#         indexes_1.append(np.round(point.pt[0]) + np.round(point.pt[1]) * width)
#     for point in kps_2:
#         indexes_2.append(np.round(point.pt[0]) + np.round(point.pt[1]) * width)
#
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     matches = bf.knnMatch(des_1, des_2, k=1)
#     # Apply ratio test
#     good = []
#     for m in matches:
#         if len(m) != 0:
#             good.append(m[0])
#
#     good = sorted(good, key=lambda x: x.distance)
#     display_matches_orb = cv2.drawMatches(color_1, kps_1, color_2, kps_2, good[:display_number], flags=2,
#                                           outImg=None)
#
#     # Extend 1D locations to B x C x Sampling_size
#     keypoint_number = len(indexes_1)
#     rough_source_feature_1D_locations = torch.from_numpy(np.asarray(indexes_1)).long().cuda().view(
#         1, 1,
#         keypoint_number).expand(
#         -1, rough_feature_length, -1)
#
#     # Sampled rough locator feature vectors
#     sampled_rough_feature_vectors = torch.gather(
#         rough_feature_map_1.view(1, rough_feature_length, height * width), 2,
#         rough_source_feature_1D_locations.long())
#     sampled_rough_feature_vectors = sampled_rough_feature_vectors.view(1, rough_feature_length,
#                                                                        keypoint_number,
#                                                                        1,
#                                                                        1).permute(0, 2, 1, 3,
#                                                                                   4).view(1,
#                                                                                           keypoint_number,
#                                                                                           rough_feature_length,
#                                                                                           1, 1)
#
#     rough_filter_response_map = torch.nn.functional.conv2d(
#         input=rough_feature_map_2.view(1, rough_feature_length, height, width),
#         weight=sampled_rough_feature_vectors.view(keypoint_number,
#                                                   rough_feature_length,
#                                                   1, 1), padding=0)
#
#     # 1 x Sampling_size x H x W
#     rough_filter_response_map = 0.5 * rough_filter_response_map + 0.5
#     rough_filter_response_map = torch.exp(scale * (rough_filter_response_map - threshold))
#     rough_filter_response_map = rough_filter_response_map / torch.sum(rough_filter_response_map,
#                                                                       dim=(2, 3),
#                                                                       keepdim=True)
#
#     # # Sampled texture matcher feature vectors
#     fine_source_feature_1D_locations = torch.from_numpy(np.asarray(indexes_1)).cuda().long().view(1, 1,
#                                                                                                   keypoint_number).expand(
#         -1, fine_feature_length, -1)
#     sampled_fine_feature_vectors = torch.gather(
#         fine_feature_map_1.view(1, fine_feature_length, height * width), 2,
#         fine_source_feature_1D_locations.long())
#     sampled_fine_feature_vectors = sampled_fine_feature_vectors.view(1, fine_feature_length,
#                                                                      keypoint_number, 1,
#                                                                      1).permute(0, 2, 1, 3, 4).view(
#         1, keypoint_number,
#         fine_feature_length,
#         1, 1)
#     fine_filter_response_map = torch.nn.functional.conv2d(
#         input=fine_feature_map_2.view(1, fine_feature_length, height, width),
#         weight=sampled_fine_feature_vectors.view(keypoint_number,
#                                                  fine_feature_length,
#                                                  1, 1), padding=0)
#     # 1 x Sampling_size x H x W
#     fine_filter_response_map = 0.5 * fine_filter_response_map + 0.5
#     fine_filter_response_map = torch.exp(
#         scale * (fine_filter_response_map - threshold)) * boundary.view(1, 1, height, width).expand(
#         -1, keypoint_number, -1, -1)
#     fine_filter_response_map = fine_filter_response_map / torch.sum(fine_filter_response_map,
#                                                                     dim=(2, 3), keepdim=True)
#
#     merged_response_map = rough_filter_response_map * fine_filter_response_map
#     max_reponses, max_indexes = torch.max(merged_response_map.view(keypoint_number, -1), dim=1,
#                                           keepdim=False)
#     # query is 1 and train is 2 here
#     # selected_keypoint_indexes_1 = torch.arange(max_reponses.shape[0]).cuda().view(-1)
#     # torch.nonzero(-torch.log(max_reponses) < 100.0).view(-1)
#     selected_detected_1D_locations_2 = max_indexes.view(-1)  # torch.gather(max_indexes.view(-1), 0,
#     # selected_keypoint_indexes_1).view(-1)
#     selected_max_responses = max_reponses.view(-1)  # torch.gather(max_reponses.view(-1), 0,
#     #             selected_keypoint_indexes_1).view(-1)
#
#     # Do cross check
#     rough_feature_1D_locations_2 = selected_detected_1D_locations_2.long().view(
#         1, 1, -1).expand(-1, rough_feature_length, -1)
#     keypoint_number_2 = keypoint_number
#     # _, _, keypoint_number_2 = rough_feature_1D_locations_2.shape
#
#     # Sampled rough locator feature vectors
#     sampled_rough_feature_vectors_2 = torch.gather(
#         rough_feature_map_2.view(1, rough_feature_length, height * width), 2,
#         rough_feature_1D_locations_2.long())
#     sampled_rough_feature_vectors_2 = sampled_rough_feature_vectors_2.view(1, rough_feature_length,
#                                                                            keypoint_number_2,
#                                                                            1,
#                                                                            1).permute(0, 2, 1, 3,
#                                                                                       4).view(1,
#                                                                                               keypoint_number_2,
#                                                                                               rough_feature_length,
#                                                                                               1, 1)
#
#     rough_filter_response_map_2 = torch.nn.functional.conv2d(
#         input=rough_feature_map_1.view(1, rough_feature_length, height, width),
#         weight=sampled_rough_feature_vectors_2.view(keypoint_number_2,
#                                                     rough_feature_length,
#                                                     1, 1), padding=0)
#
#     # 1 x Sampling_size x H x W
#     rough_filter_response_map_2 = 0.5 * rough_filter_response_map_2 + 0.5
#     rough_filter_response_map_2 = torch.exp(scale * (rough_filter_response_map_2 - threshold))
#     rough_filter_response_map_2 = rough_filter_response_map_2 / torch.sum(
#         rough_filter_response_map_2,
#         dim=(2, 3),
#         keepdim=True)
#     # # Sampled texture matcher feature vectors
#     fine_source_feature_1D_locations_2 = selected_detected_1D_locations_2.long().cuda().view(
#         1, 1, -1).expand(-1, fine_feature_length, -1)
#     sampled_fine_feature_vectors_2 = torch.gather(
#         fine_feature_map_2.view(1, fine_feature_length, height * width), 2,
#         fine_source_feature_1D_locations_2.long())
#     sampled_fine_feature_vectors_2 = sampled_fine_feature_vectors_2.view(1, fine_feature_length,
#                                                                          keypoint_number_2, 1,
#                                                                          1).permute(0, 2, 1, 3,
#                                                                                     4).view(
#         1, keypoint_number_2,
#         fine_feature_length,
#         1, 1)
#     fine_filter_response_map_2 = torch.nn.functional.conv2d(
#         input=fine_feature_map_1.view(1, fine_feature_length, height, width),
#         weight=sampled_fine_feature_vectors_2.view(keypoint_number_2,
#                                                    fine_feature_length,
#                                                    1, 1), padding=0)
#     # 1 x Sampling_size x H x W
#     fine_filter_response_map_2 = 0.5 * fine_filter_response_map_2 + 0.5
#     fine_filter_response_map_2 = torch.exp(
#         scale * (fine_filter_response_map_2 - threshold)) * boundary.view(1, 1, height,
#                                                                           width).expand(
#         -1, keypoint_number_2, -1, -1)
#     fine_filter_response_map_2 = fine_filter_response_map_2 / torch.sum(fine_filter_response_map_2,
#                                                                         dim=(2, 3), keepdim=True)
#
#     merged_response_map_2 = rough_filter_response_map_2 * fine_filter_response_map_2
#     max_reponses_2, max_indexes_2 = torch.max(merged_response_map_2.view(keypoint_number_2, -1),
#                                               dim=1,
#                                               keepdim=False)
#
#     keypoint_1D_locations_1 = torch.from_numpy(np.asarray(indexes_1)).float().cuda().view(
#         keypoint_number, 1)
#     keypoint_2D_locations_1 = torch.cat(
#         [torch.fmod(keypoint_1D_locations_1, width),
#          torch.floor(keypoint_1D_locations_1 / width)],
#         dim=1).view(keypoint_number, 2).float()
#
#     detected_keypoint_1D_locations_1 = max_indexes_2.float().view(keypoint_number, 1)
#     detected_keypoint_2D_locations_1 = torch.cat(
#         [torch.fmod(detected_keypoint_1D_locations_1, width),
#          torch.floor(detected_keypoint_1D_locations_1 / width)],
#         dim=1).view(keypoint_number, 2).float()
#
#     # TODO: We will accept the feature matches if the max indexes here is not far away from the original key point location from ORB
#     cross_check_correspondence_distances = torch.norm(
#         keypoint_2D_locations_1 - detected_keypoint_2D_locations_1, dim=1, p=2).view(
#         keypoint_number)
#     valid_correspondence_indexes = torch.nonzero(cross_check_correspondence_distances < 10.0).view(
#         -1)
#
#     valid_detected_1D_locations_2 = torch.gather(selected_detected_1D_locations_2.long().view(-1),
#                                                  0, valid_correspondence_indexes.long())
#     valid_max_responses = torch.gather(selected_max_responses.view(-1), 0,
#                                        valid_correspondence_indexes.long())
#
#     valid_detected_1D_locations_2_cpu = valid_detected_1D_locations_2.data.cpu().numpy()
#     valid_max_responses_cpu = valid_max_responses.data.cpu().numpy()
#     valid_correspondence_indexes_cpu = valid_correspondence_indexes.data.cpu().numpy()
#
#     keypoints_2 = []
#     for index in valid_detected_1D_locations_2_cpu:
#         keypoints_2.append(
#             cv2.KeyPoint(x=float(np.floor(index % width)), y=float(np.floor(index / width)), _size=1.0))
#
#     matches = []
#     for i, (query_index, response) in enumerate(
#             zip(valid_correspondence_indexes_cpu, valid_max_responses_cpu)):
#         # distance = hamming_distance(np.asarray(des_1[query_index], dtype=np.int32), np.asarray(descriptions_2[i], dtype=np.int32))
#         matches.append(cv2.DMatch(_trainIdx=i, _queryIdx=query_index, _distance=response))
#     matches = sorted(matches, key=lambda x: x.distance, reverse=True)
#     display_matches_ai = cv2.drawMatches(color_1, kps_1, color_2, keypoints_2, matches[:display_number], flags=2,
#                                          outImg=None)
#     return display_matches_ai, display_matches_orb


# cv2.namedWindow('matches', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('matches', 2000, 1500)
# cv2.imshow("matches", img3)

# def feature_matching(descriptor, color_1, color_2, feature_map_1, feature_map_2,
#                      boundary, scale, threshold, display_number):
#     # Color image 3 x H x W
#     # Feature map C x H x W
#     feature_length, height, width = feature_map_1.shape
#
#     color_1 = np.moveaxis(color_1.data.cpu().numpy(), source=[0, 1, 2], destination=[2, 0, 1])
#     color_2 = np.moveaxis(color_2.data.cpu().numpy(), source=[0, 1, 2], destination=[2, 0, 1])
#
#     # Extract corner points
#     color_1 = cv2.cvtColor(np.uint8(255 * (color_1 * 0.5 + 0.5)), cv2.COLOR_HSV2BGR_FULL)
#     color_2 = cv2.cvtColor(np.uint8(255 * (color_2 * 0.5 + 0.5)), cv2.COLOR_HSV2BGR_FULL)
#
#     kernel = np.ones((10, 10), np.uint8)
#     boundary = boundary.view(height, width).data.cpu().numpy()
#     boundary = cv2.erode(boundary, kernel, iterations=2)
#     kps_1, des_1 = descriptor.detectAndCompute(color_1, boundary)
#     kps_2, des_2 = descriptor.detectAndCompute(color_2, boundary)
#
#     source_keypoint_1D_locations_list = []
#     target_keypoint_1D_locations_list = []
#     for point in kps_1:
#         source_keypoint_1D_locations_list.append(np.round(point.pt[0]) + np.round(point.pt[1]) * width)
#
#     for point in kps_2:
#         target_keypoint_1D_locations_list.append(np.round(point.pt[0]) + np.round(point.pt[1]) * width)
#
#     # Extend 1D locations to B x C x Sampling_size
#     keypoint_number = len(source_keypoint_1D_locations_list)
#
#     with torch.no_grad():
#         y_grid, x_grid = torch.meshgrid(
#             [torch.arange(start=0, end=height, dtype=torch.float32).cuda(),
#              torch.arange(start=0, end=width, dtype=torch.float32).cuda()])
#         y_grid = y_grid.view(1, 1, height, width).expand(1, keypoint_number, -1, -1).float() / height
#         x_grid = x_grid.view(1, 1, height, width).expand(1, keypoint_number, -1, -1).float() / width
#
#     source_feature_1D_locations = torch.from_numpy(np.array(source_keypoint_1D_locations_list)).float().cuda()
#     source_feature_2D_locations = torch.cat(
#         [torch.fmod(source_feature_1D_locations.view(-1, 1), width),
#          torch.floor(source_feature_1D_locations.view(-1, 1) / width)],
#         dim=1).view(keypoint_number, 2).float()
#     # Sampled rough locator feature vectors
#     sampled_source_feature_vectors = torch.gather(
#         feature_map_1.view(1, feature_length, height * width), 2,
#         source_feature_1D_locations.long().view(1, 1, keypoint_number).expand(
#             -1, feature_length, -1))
#     sampled_source_feature_vectors = sampled_source_feature_vectors.view(1, feature_length,
#                                                                          keypoint_number,
#                                                                          1,
#                                                                          1).permute(0, 2, 1, 3,
#                                                                                     4).view(1,
#                                                                                             keypoint_number,
#                                                                                             feature_length,
#                                                                                             1, 1)
#     filter_target_response_map = torch.nn.functional.conv2d(
#         input=feature_map_2.view(1, feature_length, height, width),
#         weight=sampled_source_feature_vectors.view(keypoint_number,
#                                                    feature_length,
#                                                    1, 1), padding=0)
#     # 1 x Sampling_size x H x W
#     filter_target_response_map = 0.5 * filter_target_response_map + 0.5
#     filter_target_response_map = torch.exp(scale * (filter_target_response_map - threshold))
#     print(torch.min(filter_target_response_map), torch.max(filter_target_response_map))
#     filter_target_response_map = filter_target_response_map / (1.0e-8 + torch.sum(filter_target_response_map,
#                                                                         dim=(2, 3),
#                                                                         keepdim=True))
#
#
#     detected_target_feature_2D_locations = \
#         torch.cat(
#             [torch.sum(filter_target_response_map * x_grid, dim=(2, 3), keepdim=True).view(1,
#                                                                                            keypoint_number,
#                                                                                            1),
#              torch.sum(filter_target_response_map * y_grid, dim=(2, 3), keepdim=True).view(1,
#                                                                                            keypoint_number,
#                                                                                            1)], dim=2).view(
#             keypoint_number, 2)
#     detected_target_feature_1D_locations = torch.round(detected_target_feature_2D_locations[:, 0]) + \
#                                            torch.round(detected_target_feature_2D_locations[:, 1]) * width
#
#     print(torch.min(detected_target_feature_1D_locations), torch.max(detected_target_feature_1D_locations))
#
#     # query is source and train is target here
#     # selected_detected_target_feature_1D_locations = detected_target_feature_1D_locations.view(-1)
#     # Do cross check
#     detected_target_feature_1D_locations = detected_target_feature_1D_locations.view(
#         1, 1, keypoint_number).expand(-1, feature_length, -1)
#     print(source_feature_2D_locations)
#     # Sampled rough locator feature vectors
#     sampled_target_feature_vectors = torch.gather(
#         feature_map_2.view(1, feature_length, height * width), 2,
#         detected_target_feature_1D_locations.long())
#     sampled_target_feature_vectors = sampled_target_feature_vectors.view(1, feature_length,
#                                                                          keypoint_number,
#                                                                          1,
#                                                                          1).permute(0, 2, 1, 3,
#                                                                                     4).view(1,
#                                                                                             keypoint_number,
#                                                                                             feature_length,
#                                                                                             1, 1)
#     print(source_feature_2D_locations)
#     source_filter_response_map = torch.nn.functional.conv2d(
#         input=feature_map_1.view(1, feature_length, height, width),
#         weight=sampled_target_feature_vectors.view(keypoint_number,
#                                                    feature_length,
#                                                    1, 1), padding=0)
#
#     # 1 x Sampling_size x H x W
#     source_filter_response_map = 0.5 * source_filter_response_map + 0.5
#     source_filter_response_map = torch.exp(scale * (source_filter_response_map - threshold))
#     source_filter_response_map = source_filter_response_map / torch.sum(
#         source_filter_response_map,
#         dim=(2, 3),
#         keepdim=True)
#     print(source_feature_2D_locations)
#     detected_source_feature_2D_locations = \
#         torch.cat(
#             [torch.sum(source_filter_response_map * x_grid, dim=(2, 3), keepdim=True).view(1,
#                                                                                            keypoint_number,
#                                                                                            1),
#              torch.sum(source_filter_response_map * y_grid, dim=(2, 3), keepdim=True).view(1,
#                                                                                            keypoint_number,
#                                                                                            1)], dim=2).view(
#             keypoint_number, 2)
#     # source_feature_1D_locations_2 = torch.floor(torch.from_numpy(np.array(source_keypoint_1D_locations_list, dtype=np.float32))).cuda()
#     # print(source_feature_1D_locations)
#     print(source_feature_2D_locations)
#     detected_source_feature_2D_locations = detected_source_feature_2D_locations.float().view(keypoint_number, 2)
#
#     # TODO: We will accept the feature matches if the max indexes here is not far away from the original key point location from ORB
#     cross_check_correspondence_distances = torch.norm(
#         source_feature_2D_locations - detected_source_feature_2D_locations, dim=1, p=2).view(
#         -1).float()
#     valid_correspondence_indexes = torch.nonzero(cross_check_correspondence_distances < 100.0).view(
#         -1)
#
#     valid_detected_target_feature_1D_locations = torch.gather(detected_target_feature_1D_locations.view(-1),
#                                                               0, valid_correspondence_indexes.long())
#     valid_cross_check_correspondence_distances = torch.gather(cross_check_correspondence_distances.float().view(-1),
#                                                               0, valid_correspondence_indexes.long())
#
#     valid_detected_target_feature_1D_locations_cpu = valid_detected_target_feature_1D_locations.data.cpu().numpy()
#     # valid_max_responses_cpu = valid_max_responses.data.cpu().numpy()
#     valid_correspondence_indexes_cpu = valid_correspondence_indexes.data.cpu().numpy()
#     valid_cross_check_correspondence_distances_cpu = valid_cross_check_correspondence_distances.data.cpu().numpy()
#
#     target_keypoints = []
#     for target_location_1D in valid_detected_target_feature_1D_locations_cpu:
#         target_keypoints.append(
#             cv2.KeyPoint(x=float(target_location_1D % width), y=float(np.floor(target_location_1D / width)),
#                          _size=1.0))
#
#     matches = []
#     for i, (query_index, distance) in enumerate(
#             zip(valid_correspondence_indexes_cpu, valid_cross_check_correspondence_distances_cpu)):
#         # distance = hamming_distance(np.asarray(des_1[query_index], dtype=np.int32), np.asarray(descriptions_2[i], dtype=np.int32))
#         matches.append(cv2.DMatch(_trainIdx=i, _queryIdx=query_index, _distance=distance))
#     matches = sorted(matches, key=lambda x: x.distance, reverse=False)
#     display_matches_ai = cv2.drawMatches(color_1, kps_1, color_2, target_keypoints, matches[:display_number], flags=2,
#                                          outImg=None)
#
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     matches = bf.knnMatch(des_1, des_2, k=1)
#     # Apply ratio test
#     good = []
#     for m in matches:
#         if len(m) != 0:
#             good.append(m[0])
#
#     good = sorted(good, key=lambda x: x.distance)
#     display_matches_orb = cv2.drawMatches(color_1, kps_1, color_2, kps_2, good[:display_number], flags=2, outImg=None)
#
#     return display_matches_ai, display_matches_orb


def orb_feature_matches_generation(descriptor, color_1, color_2, boundary, sampling_size):
    height, width, channel = color_1.shape
    # color_1 = cv2.cvtColor(np.uint8(255 * (color_1 * 0.5 + 0.5)), cv2.COLOR_HSV2BGR_FULL)
    # color_2 = cv2.cvtColor(np.uint8(255 * (color_2 * 0.5 + 0.5)), cv2.COLOR_HSV2BGR_FULL)

    kernel = np.ones((10, 10), np.uint8)
    boundary = cv2.erode(boundary, kernel, iterations=1)
    kps_1, des_1 = descriptor.detectAndCompute(color_1, boundary)
    kps_2, des_2 = descriptor.detectAndCompute(color_2, boundary)

    # # TODO: ORB traditional sparse matching
    # indexes_1 = []
    # indexes_2 = []
    # for point in kps_1:
    #     indexes_1.append(np.round(point.pt[0]) + np.round(point.pt[1]) * width)
    #
    # for point in kps_2:
    #     indexes_2.append(np.round(point.pt[0]) + np.round(point.pt[1]) * width)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.knnMatch(des_1, des_2, k=1)

    feature_locations_2D_1 = np.zeros((sampling_size, 2), dtype=np.float32)
    feature_locations_2D_2 = np.zeros((sampling_size, 2), dtype=np.float32)

    feature_locations_1D_1 = np.zeros((sampling_size, 1), dtype=np.float32)
    feature_locations_1D_2 = np.zeros((sampling_size, 1), dtype=np.float32)

    good = [m[0] for m in matches if len(m) != 0]
    good = sorted(good, key=lambda x: x.distance)

    match_number = len(good)
    for i in range(sampling_size):
        feature_locations_2D_1[i, 0] = np.round(kps_1[good[i % match_number].queryIdx].pt[0])
        feature_locations_2D_1[i, 1] = np.round(kps_1[good[i % match_number].queryIdx].pt[1])

        feature_locations_2D_2[i, 0] = np.round(kps_2[good[i % match_number].trainIdx].pt[0])
        feature_locations_2D_2[i, 1] = np.round(kps_2[good[i % match_number].trainIdx].pt[1])

        feature_locations_1D_2[i] = feature_locations_2D_2[i, 0] + feature_locations_2D_2[i, 1] * width
        feature_locations_1D_1[i] = feature_locations_2D_1[i, 0] + feature_locations_2D_1[i, 1] * width

    return feature_locations_2D_1, feature_locations_2D_2, feature_locations_1D_1, feature_locations_1D_2

    # temp = cosine_distance_map.view(keypoint_number, height, width)
    # temp_cpu = temp[keypoint_number - 1].view(height, width).data.cpu().numpy()
    # heatmap_display = cv2.applyColorMap(np.uint8(255 * temp_cpu), cv2.COLORMAP_HOT)
    # cv2.imshow("heatmap 1", heatmap_display)
    # print(temp[keypoint_number - 1])
    # # Sampling_size x 1
    # max_reponses, max_indexes = torch.max(cosine_distance_map_ori.view(keypoint_number, -1), dim=1, keepdim=False)
    #
    # # 1 is query, 2 is train for the above case
    # # selected_query_indexes = torch.nonzero(torch.where(-torch.log(max_reponses) < 3.0, max_indexes, torch.tensor(0).long().cuda())).view(-1)
    # selected_query_indexes = torch.nonzero(-torch.log(max_reponses) < 6.0).view(-1)
    # selected_train_indexes = torch.gather(max_indexes.view(-1), 0, selected_query_indexes).view(-1)
    # selected_max_responses = torch.gather(max_reponses.view(-1), 0, selected_query_indexes).view(-1)
    #
    # selected_query_indexes_cpu = selected_query_indexes.data.cpu().numpy()
    # selected_train_indexes_cpu = selected_train_indexes.data.cpu().numpy()
    # selected_max_responses_cpu = selected_max_responses.data.cpu().numpy()
    #
    # # define keypoints for the train image
    # train_keypoints = []
    # for index in selected_train_indexes_cpu:
    #     train_keypoints.append(cv2.KeyPoint(x=float(index % width), y=float(index / width), _size=1.0))
    #
    # matches = []
    # for i, (query_index, train_index, response) in enumerate(zip(selected_query_indexes_cpu, selected_train_indexes_cpu, selected_max_responses_cpu)):
    #     matches.append(cv2.DMatch(_trainIdx=i, _queryIdx=query_index, _distance=response))
    # matches = sorted(matches, key=lambda x: x.distance, reverse=True)
    #
    # img3 = cv2.drawMatches(color_1, kps_1, color_2, train_keypoints, matches[:20], flags=2, outImg=None)
    # cv2.namedWindow('matches', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('matches', 2200, 1100)
    # cv2.imshow("matches", img3)
    #
    # # display_colors("1_goal", step, writer,
    # #                color_1_display.view(1, 3, height, width) - 1.0 * heatmap_1[:, 0].view(1, 1, height, width))
    # # display_colors("2_goal", step, writer,
    # #                color_2_display.view(1, 3, height, width) - 1.0 * heatmap_2.view(1, 1, height, width))
    # # display_colors("1_detected", step, writer, colors_1_display - 1.0 * display_1)
    # cosine_distance_map = cosine_distance_map[0, 0].view(1, 1, height, width)
    # max_cosine_distance_map, _ = torch.max(cosine_distance_map.view(-1, height * width), dim=1)
    # cosine_distance_map = cosine_distance_map / max_cosine_distance_map.view(-1, 1, 1, 1)
    # display_colors("2_detected_debug", step, writer, color_2_display - 1.0 * cosine_distance_map)
    # display_feature_matching_map(2, step, "predicted_heatmap_debug", writer, cosine_distance_map)
    #
    # cv2.waitKey()

    # return cosine_distance_map,
    # color_1 = np.moveaxis(color_1.data.cpu().numpy(), source=[0, 1, 2], destination=[2, 0, 1])
    # kps_2, des_2 = descriptor.detectAndCompute(color_2, None)
    # indexes_2 = []
    # for point in kps_2:
    #     indexes_2.append(point.pt[0] + point.pt[1] * width)
    #
    # # Extend 1D locations to B x C x Sampling_size
    # keypoint_number = len(indexes_2)
    # source_feature_1D_locations = torch.from_numpy(np.asarray(indexes_2)).long().cuda().view(1, 1,
    #                                                                                          keypoint_number).expand(
    #     -1, feature_length, -1)
    #
    # sampled_feature_vectors = torch.gather(feature_map_1.view(1, feature_length, height * width), 2,
    #                                        source_feature_1D_locations.long())
    # sampled_feature_vectors = sampled_feature_vectors.view(1, feature_length, keypoint_number, 1,
    #                                                        1).permute(0, 2, 1, 3, 4).view(1, keypoint_number,
    #                                                                                       feature_length,
    #                                                                                       1, 1)
    #
    # cosine_distance_map = torch.nn.functional.conv2d(input=feature_map_1.view(1, feature_length, height, width),
    #                                                  weight=sampled_feature_vectors.view(keypoint_number,
    #                                                                                      feature_length,
    #                                                                                      1, 1), padding=0)
    # # 1 x Sampling_size x H x W
    # cosine_distance_map = 0.5 * cosine_distance_map + 0.5
    # cosine_distance_map = torch.exp(50.0 * (cosine_distance_map - 0.7))
    # cosine_distance_map = cosine_distance_map / torch.sum(cosine_distance_map, dim=(2, 3), keepdim=True)
    #
    # # Sampling_size x 1
    # max_reponses, max_indexes = torch.max(cosine_distance_map.view(keypoint_number, -1), dim=1, keepdim=True)
    # print("responses 2:", -torch.log(max_reponses))
    # print("indexes 2:", max_indexes)
    # # cosine_distance_map_cpu = cosine_distance_map.data.cpu().numpy()
    #
    # # We only recognize it as a valid correspondence only if the max log value is larger than 2

    # # print(color_1.shape)
    # # H x W x 3
    # color_1 = np.moveaxis(color_1.data.cpu().numpy(), source=[0, 1, 2], destination=[2, 0, 1])
    # color_2 = np.moveaxis(color_2.data.cpu().numpy(), source=[0, 1, 2], destination=[2, 0, 1])
    # # H x W x C
    # feature_map_1 = np.moveaxis(feature_map_1.data.cpu().numpy(), source=[0, 1, 2], destination=[2, 0, 1])
    # feature_map_2 = np.moveaxis(feature_map_2.data.cpu().numpy(), source=[0, 1, 2], destination=[2, 0, 1])
    #
    # # Extract corner points
    # color_1 = cv2.cvtColor(np.uint8(255 * (color_1 * 0.5 + 0.5)), cv2.COLOR_HSV2BGR_FULL)
    # color_2 = cv2.cvtColor(np.uint8(255 * (color_2 * 0.5 + 0.5)), cv2.COLOR_HSV2BGR_FULL)
    #
    # kps_1, des_1 = descriptor.detectAndCompute(color_1, None)
    # kps_2, des_2 = descriptor.detectAndCompute(color_2, None)
    #
    # # Extract keypoints and calculate the distance against the entire other dense feature map to find the best correspondence
    # feature_map_1 = feature_map_1.reshape((-1, feature_length))
    # feature_map_2 = feature_map_2.reshape((-1, feature_length))
    #
    # indexes_1 = []
    # indexes_2 = []
    # for point in kps_1:
    #     indexes_1.append(point.pt[0] + point.pt[1] * width)
    #
    # for point in kps_2:
    #     indexes_2.append(point.pt[0] + point.pt[1] * width)
    #
    # # N x C
    # selected_features_1 = feature_map_1[np.array(indexes_1, dtype=np.int32), :].reshape(
    #     (len(indexes_1), feature_length))
    # selected_features_2 = feature_map_2[np.array(indexes_2, dtype=np.int32), :].reshape(
    #     (len(indexes_2), feature_length))
    #

    # img = cv2.drawKeypoints(color_1, kps_1, color=(255, 0, 0), outImage=None)
    # cv2.imshow("kp1", img)
    #
    # img = cv2.drawKeypoints(color_2, kps_2, color=(255, 0, 0), outImage=None)
    # cv2.imshow("kp2", img)
    #
    # # cv2.imshow("poi", poi_map_1)
    # # cv2.waitKey()
    #
    # feature_map_1 = feature_map_1.reshape((-1, feature_length))
    # feature_map_2 = feature_map_2.reshape((-1, feature_length))
    #
    # indexes_1 = []
    # indexes_2 = []
    # for point in kps_1:
    #     # print(point.pt)
    #     indexes_1.append(point.pt[0] + point.pt[1] * width)
    #
    # for point in kps_2:
    #     indexes_2.append(point.pt[0] + point.pt[1] * width)
    #
    # # print(indexes_1)
    # selected_features_1 = feature_map_1[np.array(indexes_1, dtype=np.int32), :].reshape(
    #     (len(indexes_1), feature_length))
    # selected_features_2 = feature_map_2[np.array(indexes_2, dtype=np.int32), :].reshape(
    #     (len(indexes_2), feature_length))
    #
    # # selected_features_1 = (selected_features_1 / np.linalg.norm(selected_features_1, axis=1, keepdims=True, ord=2)).astype(np.float32)
    # # selected_features_2 = (selected_features_2 / np.linalg.norm(selected_features_2, axis=1, keepdims=True, ord=2)).astype(np.float32)
    #
    # # print(selected_features_1.shape)
    # # print(selected_features_2.shape)
    # # 1 is query, 2 is train
    # bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)
    # matches = bf.knnMatch(selected_features_1, selected_features_2, k=2)
    #
    # # print(matches)
    # # Apply ratio test
    # good = []
    # for m, n in matches:
    #     #     print("n, m", n.distance, m.distance)
    #     #     # print(m.distance)
    #     #     # print(m, n)
    #     if m.distance < 0.1:
    #         # dx = kps_2[n.trainIdx].pt[0] - kps_2[m.trainIdx].pt[0]
    #         # dy = kps_2[n.trainIdx].pt[1] - kps_2[m.trainIdx].pt[1]
    #         # if dx*dx + dy*dy <= 50:
    #         #     good.append(m)
    #         # else:
    #         if m.distance < 0.8 * n.distance:
    #             good.append(m)
    #     # if len(m) != 0:
    #     #     good.append(m[0])
    # # for m in matches:
    # #     if len(m) != 0:
    # #         if m[0].distance < 0.1:
    # #             good.append(m[0])
    # #     # print(m.imgIdx, m.queryIdx, m.trainIdx)
    # #
    # # good = sorted(good, key=lambda x: x.distance)
    #
    # # matches = sorted(matches, key=lambda x: x.distance)
    #
    # # # K_1 x K_2
    # # cosine_correlation_matrix = np.matmul(selected_features_1, selected_features_2)
    # # indexes_max_correlation_2 = np.argmax(cosine_correlation_matrix, axis=1)
    # # max_correlations_2 = np.amax(cosine_correlation_matrix, axis=1, keepdims=True)
    # # # print(indexes_max_correlation_2, max_correlations_2)
    # # matches = []
    # # for i, (pair_index_2, correlation) in enumerate(zip(indexes_max_correlation_2, max_correlations_2)):
    # #     if correlation > 0.7:
    # #         matches.append(cv2.DMatch(_trainIdx=pair_index_2, _queryIdx=i, _distance=correlation))
    # #
    # img3 = cv2.drawMatches(color_1, kps_1, color_2, kps_2, good[:30], flags=2, outImg=None)
    # cv2.namedWindow('matches', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('matches', 2200, 1100)
    # cv2.imshow("matches", img3)
    #
    # matches = bf.knnMatch(des_1, des_2, k=1)
    # # Apply ratio test
    # good = []
    # for m in matches:
    #     # print(m, n)
    #     # if m.distance < 0.4 and m.distance < 0.8 * n.distance:
    #     if len(m) != 0:
    #         good.append(m[0])
    #         # print(m[0].imgIdx, m[0].queryIdx, m[0].trainIdx)
    #
    # good = sorted(good, key=lambda x: x.distance)
    # img3 = cv2.drawMatches(color_1, kps_1, color_2, kps_2, good[:30], flags=2, outImg=None)
    # cv2.namedWindow('matches_2', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('matches_2', 2200, 1100)
    # cv2.imshow("matches_2", img3)
    # cv2.waitKey()
    # return


def gather_feature_matching_data_py3(feature_matching_model_path, sub_folder, sfm_root, downsampling, teacher_depth,
                                     inlier_percentage, load_intermediate_data, precompute_root, batch_size):
    # Feature matching
    feature_matching_model = models.FCDenseNet57_Feature()
    # Multi-GPU running
    feature_matching_model = torch.nn.DataParallel(feature_matching_model)
    feature_matching_model.eval()

    if feature_matching_model_path.exists():
        print("Loading {:s} ...".format(str(feature_matching_model_path)))
        state = torch.load(str(feature_matching_model_path))
        feature_matching_model.load_state_dict(state)
    else:
        print("No previous student model detected")
        raise OSError

    feature_matching_model = feature_matching_model.cuda()

    video_frame_filenames = get_all_color_image_names_in_sequence(sub_folder)
    print("Start gathering fusion data for {}".format(sub_folder))
    folder_list = get_all_subfolder_names(sfm_root, bag_range=(1, 9))
    # TODO: Check why are there no bag 8 in the precompute data
    fusion_dataset = dataset.SfMDataset(image_file_names=video_frame_filenames,
                                        folder_list=folder_list,
                                        to_augment=False,
                                        transform=None,
                                        downsampling=downsampling,
                                        net_depth=teacher_depth, inlier_percentage=inlier_percentage,
                                        use_store_data=load_intermediate_data,
                                        store_data_root=precompute_root,
                                        use_view_indexes_per_point=True,
                                        visualize=False,
                                        phase="load_color_and_boundary", is_hsv=True,
                                        load_optimized_pose=False)
    fusion_loader = torch.utils.data.DataLoader(dataset=fusion_dataset, batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=batch_size)

    colors_list = []
    rough_feature_maps_list = []
    fine_feature_maps_list = []
    # Update progress bar
    tq = tqdm.tqdm(total=len(fusion_loader) * batch_size)
    for batch, (colors_1, boundaries, image_names,
                folders, starts_h, starts_w) in enumerate(fusion_loader):
        tq.update(batch_size)
        colors_1, boundaries = colors_1.cuda(), boundaries.cuda()
        colors_1 = boundaries * colors_1
        rough_feature_maps_1, fine_feature_maps_1 = feature_matching_model(colors_1)
        rough_feature_maps_1 = rough_feature_maps_1 / torch.norm(rough_feature_maps_1,
                                                                 dim=1, keepdim=True)
        fine_feature_maps_1 = fine_feature_maps_1 / torch.norm(fine_feature_maps_1,
                                                               dim=1, keepdim=True)
        start_h = starts_h[0]
        start_w = starts_w[0]
        for i in range(colors_1.shape[0]):
            colors_list.append(colors_1[i].data.cpu().numpy())
            rough_feature_maps_list.append(rough_feature_maps_1[i].data.cpu().numpy())
            fine_feature_maps_list.append(fine_feature_maps_1[i].data.cpu().numpy())
    tq.close()
    torch.cuda.empty_cache()
    return colors_list, boundaries[0].data.cpu().numpy(), \
           rough_feature_maps_list, fine_feature_maps_list, start_h, start_w


def gather_color_boundary_data(sub_folder, sfm_root, downsampling, network_downsampling,
                               inlier_percentage, load_intermediate_data, precompute_root, batch_size, bag_range):
    video_frame_filenames = get_all_color_image_names_in_sequence(sub_folder)
    print("Start gathering fusion data for {}".format(sub_folder))
    folder_list = get_all_subfolder_names(sfm_root, bag_range=(bag_range[0], bag_range[1]))
    fusion_dataset = dataset.SfMDataset(image_file_names=video_frame_filenames,
                                        folder_list=folder_list,
                                        to_augment=False,
                                        transform=None,
                                        downsampling=downsampling,
                                        network_downsampling=network_downsampling, inlier_percentage=inlier_percentage,
                                        use_store_data=load_intermediate_data,
                                        store_data_root=precompute_root,
                                        phase="load_color_and_boundary", is_hsv=False,
                                        load_optimized_pose=False, rgb_mode='rgb')
    fusion_loader = torch.utils.data.DataLoader(dataset=fusion_dataset, batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=batch_size)

    colors_list = []
    # Update progress bar
    tq = tqdm.tqdm(total=len(fusion_loader) * batch_size)
    for batch, (colors_1, boundaries, image_names,
                folders, starts_h, starts_w) in enumerate(fusion_loader):
        tq.update(batch_size)
        if batch == 0:
            boundary = boundaries[0].data.numpy()
            start_h = starts_h[0]
            start_w = starts_w[0]
        for idx in range(colors_1.shape[0]):
            colors_list.append(colors_1[idx].data.numpy())
        # break
    tq.close()
    del fusion_dataset, fusion_loader
    return colors_list, boundary, start_h, start_w


def gather_descriptor_map_data(feature_descriptor_model_path, sub_folder, sfm_root, downsampling,
                               network_downsampling,
                               inlier_percentage, load_intermediate_data, precompute_root, batch_size,
                               bag_range,
                               filter_growth_rate, feature_length, is_hsv, rgb_mode,
                               gpu_id, final_convs_filter_base=None):
    # # Feature matching
    # feature_descriptor_model = models.FeatureFCDenseNetSingle(
    #     in_channels=3, down_blocks=(3, 3, 3, 3, 3),
    #     up_blocks=(3, 3, 3, 3, 3), bottleneck_layers=4,
    #     growth_rate=filter_growth_rate, out_chans_first_conv=16, feature_length=feature_length,
    #     final_convs_filter_base=final_convs_filter_base)
    feature_descriptor_model = models.FCDenseNetNoPyramid(
        in_channels=3, down_blocks=(3, 3, 3, 3, 3),
        up_blocks=(3, 3, 3, 3, 3), bottleneck_layers=4,
        growth_rate=filter_growth_rate, out_chans_first_conv=16, feature_length=feature_length)

    # feature_descriptor_model = feature_descriptor_model.cuda(gpu_id)
    # Multi-GPU running
    feature_descriptor_model = torch.nn.DataParallel(feature_descriptor_model)
    feature_descriptor_model.eval()

    if feature_descriptor_model_path.exists():
        print("Loading {:s} ...".format(str(feature_descriptor_model_path)))
        state = torch.load(str(feature_descriptor_model_path))
        if "model" in state:
            feature_descriptor_model.load_state_dict(state["model"])
        elif "state_dict" in state:
            feature_descriptor_model.load_state_dict(state["state_dict"])
        else:
            raise OSError
    else:
        print("No trained model detected")
        raise OSError
    feature_descriptor_model = feature_descriptor_model.module
    feature_descriptor_model = feature_descriptor_model.cuda(gpu_id)

    video_frame_filenames = get_all_color_image_names_in_sequence(sub_folder)
    print("Start gathering feature matching data for {}".format(sub_folder))
    folder_list = get_all_subfolder_names(sfm_root, bag_range=(bag_range[0], bag_range[1]))
    video_dataset = dataset.SfMDataset(image_file_names=video_frame_filenames,
                                       folder_list=folder_list,
                                       to_augment=False,
                                       transform=None,
                                       downsampling=downsampling,
                                       network_downsampling=network_downsampling, inlier_percentage=inlier_percentage,
                                       use_store_data=load_intermediate_data,
                                       store_data_root=precompute_root,
                                       phase="load_color_boundary_and_intrinsic", is_hsv=is_hsv,
                                       rgb_mode=rgb_mode,
                                       load_optimized_pose=False)
    video_loader = torch.utils.data.DataLoader(dataset=video_dataset, batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=batch_size)

    colors_list = []
    feature_maps_list = []

    # Update progress bar
    tq = tqdm.tqdm(total=len(video_loader) * batch_size)
    for batch, (colors_1, boundaries, intrinsic_matrices, image_names,
                folders) in enumerate(video_loader):
        tq.update(batch_size)
        colors_1 = colors_1.cuda(gpu_id)
        if batch == 0:
            boundary = boundaries[0].data.numpy()
            intrinsic_matrix = intrinsic_matrices[0].data.numpy()

        feature_maps_1 = feature_descriptor_model(colors_1)
        for idx in range(colors_1.shape[0]):
            colors_list.append(colors_1[idx].data.cpu().numpy())
            feature_maps_list.append(feature_maps_1[idx].data.cpu().numpy())
    tq.close()
    torch.cuda.empty_cache()
    return colors_list, boundary, feature_maps_list, intrinsic_matrix


def get_color_file_names_by_bag(root, training_patient_id, validation_patient_id, testing_patient_id):
    training_image_list = []
    validation_image_list = []
    testing_image_list = []

    if not isinstance(training_patient_id, list):
        training_patient_id = [training_patient_id]
    if not isinstance(validation_patient_id, list):
        validation_patient_id = [validation_patient_id]
    if not isinstance(testing_patient_id, list):
        testing_patient_id = [testing_patient_id]

    for id in training_patient_id:
        training_image_list += list(root.glob('*' + str(id) + '/_start*/0*.jpg'))
    for id in testing_patient_id:
        testing_image_list += list(root.glob('*' + str(id) + '/_start*/0*.jpg'))
    for id in validation_patient_id:
        validation_image_list += list(root.glob('*' + str(id) + '/_start*/0*.jpg'))

    training_image_list.sort()
    testing_image_list.sort()
    validation_image_list.sort()
    return training_image_list, validation_image_list, testing_image_list


def gather_single_feature_matching_data(feature_descriptor_model_path, sub_folder, sfm_root, downsampling,
                                        network_downsampling,
                                        inlier_percentage, load_intermediate_data, precompute_root, batch_size,
                                        bag_range,
                                        filter_growth_rate, feature_length, is_hsv, rgb_mode,
                                        gpu_id, final_convs_filter_base=None):
    # # Feature matching
    # feature_descriptor_model = models.FeatureFCDenseNetSingle(
    #     in_channels=3, down_blocks=(3, 3, 3, 3, 3),
    #     up_blocks=(3, 3, 3, 3, 3), bottleneck_layers=4,
    #     growth_rate=filter_growth_rate, out_chans_first_conv=16, feature_length=feature_length,
    #     final_convs_filter_base=final_convs_filter_base)
    feature_descriptor_model = models.FCDenseNetNoPyramid(
        in_channels=3, down_blocks=(3, 3, 3, 3, 3),
        up_blocks=(3, 3, 3, 3, 3), bottleneck_layers=4,
        growth_rate=filter_growth_rate, out_chans_first_conv=16, feature_length=feature_length)

    # Multi-GPU running
    feature_descriptor_model = torch.nn.DataParallel(feature_descriptor_model, device_ids=[gpu_id])
    feature_descriptor_model.eval()

    if feature_descriptor_model_path.exists():
        print("Loading {:s} ...".format(str(feature_descriptor_model_path)))
        state = torch.load(str(feature_descriptor_model_path), map_location='cuda:{}'.format(gpu_id))
        if "model" in state:
            feature_descriptor_model.load_state_dict(state["model"])
        elif "state_dict" in state:
            feature_descriptor_model.load_state_dict(state["state_dict"])
        else:
            raise OSError
        # feature_descriptor_model.load_state_dict(torch.load(str(feature_descriptor_model_path))["model"])
    else:
        print("No trained model detected")
        raise OSError
    del state
    # feature_descriptor_model = feature_descriptor_model.module
    # feature_descriptor_model = feature_descriptor_model.cuda(gpu_id)

    video_frame_filenames = get_all_color_image_names_in_sequence(sub_folder)
    print("Start gathering feature matching data for {}".format(sub_folder))
    folder_list = get_all_subfolder_names(sfm_root, bag_range=(bag_range[0], bag_range[1]))
    video_dataset = dataset.SfMDataset(image_file_names=video_frame_filenames,
                                       folder_list=folder_list,
                                       to_augment=False,
                                       transform=None,
                                       downsampling=downsampling,
                                       network_downsampling=network_downsampling, inlier_percentage=inlier_percentage,
                                       use_store_data=load_intermediate_data,
                                       store_data_root=precompute_root,
                                       phase="load_color_and_boundary", is_hsv=is_hsv,
                                       rgb_mode=rgb_mode,
                                       load_optimized_pose=False)
    video_loader = torch.utils.data.DataLoader(dataset=video_dataset, batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=batch_size)

    colors_list = []
    feature_maps_list = []

    with torch.no_grad():
        # Update progress bar
        tq = tqdm.tqdm(total=len(video_loader) * batch_size)
        for batch, (colors_1, boundaries, image_names,
                    folders, starts_h, starts_w) in enumerate(video_loader):
            # if batch > 20:
            #     break
            tq.update(batch_size)
            colors_1 = colors_1.cuda(gpu_id)
            if batch == 0:
                boundary = boundaries[0].data.numpy()
                start_h = starts_h[0].item()
                start_w = starts_w[0].item()

            feature_maps_1 = feature_descriptor_model(colors_1)
            for idx in range(colors_1.shape[0]):
                colors_list.append(colors_1[idx].data.cpu().numpy())
                feature_maps_list.append(feature_maps_1[idx].data.cpu())
    tq.close()
    del feature_descriptor_model
    return colors_list, boundary, feature_maps_list, start_h, start_w


def gather_single_feature_matching_data_full(sub_folder, sfm_root, downsampling,
                                             network_downsampling,
                                             inlier_percentage, load_intermediate_data, precompute_root, batch_size,
                                             bag_range, is_hsv, rgb_mode):
    video_frame_filenames = get_all_color_image_names_in_sequence(sub_folder)
    print("Start gathering feature matching data for {}".format(sub_folder))
    folder_list = get_all_subfolder_names(sfm_root, bag_range=(bag_range[0], bag_range[1]))
    video_dataset = dataset.SfMDataset(image_file_names=video_frame_filenames,
                                       folder_list=folder_list,
                                       to_augment=False,
                                       transform=None,
                                       downsampling=downsampling,
                                       network_downsampling=network_downsampling, inlier_percentage=inlier_percentage,
                                       use_store_data=load_intermediate_data,
                                       store_data_root=precompute_root,
                                       phase="load_downsample_full_resolution_color_and_boundary", is_hsv=is_hsv,
                                       rgb_mode=rgb_mode,
                                       load_optimized_pose=False)
    video_loader = torch.utils.data.DataLoader(dataset=video_dataset, batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=batch_size)

    colors_list = []
    full_colors_list = []
    with torch.no_grad():
        # Update progress bar
        tq = tqdm.tqdm(total=len(video_loader) * batch_size)
        for batch, (colors_1, full_colors_1, boundaries, image_names,
                    folders, starts_h, starts_w) in enumerate(video_loader):
            # if batch >= 10:
            #     break
            tq.update(batch_size)
            if batch == 0:
                boundary = boundaries[0].data.numpy()
                start_h = starts_h[0].item()
                start_w = starts_w[0].item()
            for idx in range(colors_1.shape[0]):
                colors_list.append(colors_1[idx].data.numpy())
                full_colors_list.append(full_colors_1[idx].data.numpy())
    tq.close()
    return colors_list, full_colors_list, boundary, start_h, start_w


def display_color_depth_sparse_flow_dense_flow(idx, step, writer, colors_1, pred_depths_1,
                                               sparse_flows_1, flows_from_depth_1, is_hsv,
                                               phase="Train", is_return_image=False, color_reverse=True
                                               ):
    colors_display = vutils.make_grid(colors_1 * 0.5 + 0.5, normalize=False)
    colors_display = np.moveaxis(colors_display.data.cpu().numpy(),
                                 source=[0, 1, 2], destination=[2, 0, 1])
    if is_hsv:
        colors_display = cv2.cvtColor(colors_display, cv2.COLOR_HSV2RGB_FULL)

    pred_depths_display = vutils.make_grid(pred_depths_1, normalize=True, scale_each=True)
    pred_depths_display = cv2.applyColorMap(np.uint8(255 * np.moveaxis(pred_depths_display.data.cpu().numpy(),
                                                                       source=[0, 1, 2],
                                                                       destination=[2, 0, 1])), cv2.COLORMAP_JET)
    sparse_flows_display, max_v = draw_flow(sparse_flows_1)
    dense_flows_display, _ = draw_flow(flows_from_depth_1, max_v=max_v)
    if color_reverse:
        pred_depths_display = cv2.cvtColor(pred_depths_display, cv2.COLOR_BGR2RGB)
        sparse_flows_display = cv2.cvtColor(sparse_flows_display, cv2.COLOR_BGR2RGB)
        dense_flows_display = cv2.cvtColor(dense_flows_display, cv2.COLOR_BGR2RGB)

    if is_return_image:
        return colors_display, pred_depths_display.astype(np.float32) / 255.0, \
               sparse_flows_display.astype(np.float32) / 255.0, dense_flows_display.astype(np.float32) / 255.0
    else:
        writer.add_image(phase + '/Images/Color_' + str(idx), colors_display, step, dataformats="HWC")
        writer.add_image(phase + '/Images/Pred_Depth_' + str(idx), pred_depths_display, step, dataformats="HWC")
        writer.add_image(phase + '/Images/Sparse_Flow_' + str(idx), sparse_flows_display, step, dataformats="HWC")
        writer.add_image(phase + '/Images/Dense_Flow_' + str(idx), dense_flows_display, step, dataformats="HWC")
        return


def display_warped_color_color_mean_std_depth_sparse_flow_dense_flow(idx, step, writer, warped_colors_1, colors_1,
                                                                     pred_mean_depths_1,
                                                                     pred_std_depths_1, sparse_flows_1,
                                                                     flows_from_depth_1, is_hsv,
                                                                     phase="Train", is_return_image=False,
                                                                     color_reverse=True
                                                                     ):
    colors_display = vutils.make_grid(colors_1 * 0.5 + 0.5, normalize=False)
    colors_display = np.moveaxis(colors_display.data.cpu().numpy(),
                                 source=[0, 1, 2], destination=[2, 0, 1])
    if is_hsv:
        colors_display = cv2.cvtColor(colors_display, cv2.COLOR_HSV2RGB_FULL)

    warped_colors_display = vutils.make_grid(warped_colors_1 * 0.5 + 0.5, normalize=False)
    warped_colors_display = np.moveaxis(warped_colors_display.data.cpu().numpy(),
                                        source=[0, 1, 2], destination=[2, 0, 1])
    if is_hsv:
        warped_colors_display = cv2.cvtColor(warped_colors_display, cv2.COLOR_HSV2RGB_FULL)

    pred_depths_display = vutils.make_grid(pred_mean_depths_1, normalize=True, scale_each=True)
    pred_depths_display = cv2.applyColorMap(np.uint8(255 * np.moveaxis(pred_depths_display.data.cpu().numpy(),
                                                                       source=[0, 1, 2],
                                                                       destination=[2, 0, 1])), cv2.COLORMAP_JET)

    pred_std_depths_display = vutils.make_grid(pred_std_depths_1, normalize=True, scale_each=True)
    pred_std_depths_display = cv2.applyColorMap(np.uint8(255 * np.moveaxis(pred_std_depths_display.data.cpu().numpy(),
                                                                           source=[0, 1, 2],
                                                                           destination=[2, 0, 1])), cv2.COLORMAP_JET)

    sparse_flows_display, max_v = draw_flow(sparse_flows_1)
    dense_flows_display, _ = draw_flow(flows_from_depth_1, max_v=max_v)
    if color_reverse:
        pred_depths_display = cv2.cvtColor(pred_depths_display, cv2.COLOR_BGR2RGB)
        sparse_flows_display = cv2.cvtColor(sparse_flows_display, cv2.COLOR_BGR2RGB)
        dense_flows_display = cv2.cvtColor(dense_flows_display, cv2.COLOR_BGR2RGB)

    if is_return_image:
        return warped_colors_display, colors_display, pred_depths_display.astype(
            np.float32) / 255.0, pred_std_depths_display.astype(
            np.float32) / 255.0, sparse_flows_display.astype(np.float32) / 255.0, dense_flows_display.astype(
            np.float32) / 255.0
    else:
        writer.add_image(phase + '/Images/Warped_Color_' + str(idx), warped_colors_display, step, dataformats="HWC")
        writer.add_image(phase + '/Images/Color_' + str(idx), colors_display, step, dataformats="HWC")
        writer.add_image(phase + '/Images/Pred_Depth_' + str(idx), pred_depths_display, step, dataformats="HWC")
        writer.add_image(phase + '/Images/Std_Pred_Depth_' + str(idx), pred_std_depths_display, step, dataformats="HWC")
        writer.add_image(phase + '/Images/Sparse_Flow_' + str(idx), sparse_flows_display, step, dataformats="HWC")
        writer.add_image(phase + '/Images/Dense_Flow_' + str(idx), dense_flows_display, step, dataformats="HWC")
        return


def gather_feature_matching_data(feature_descriptor_model_path, sub_folder, sfm_root, downsampling,
                                 network_downsampling,
                                 inlier_percentage, load_intermediate_data, precompute_root, batch_size, bag_range,
                                 filter_growth_rate, feature_length, final_convs_filter_base, fine_layer_count, is_hsv,
                                 rgb_mode):
    # Feature matching
    # feature_matching_model = models.FCDenseNet57_Feature()
    feature_descriptor_model = models.FeatureFCDenseNet(
        in_channels=3, down_blocks=(3, 3, 3, 3, 3),
        up_blocks=(3, 3, 3, 3, 3), bottleneck_layers=4,
        growth_rate=filter_growth_rate, out_chans_first_conv=16, feature_length=feature_length,
        final_convs_filter_base=final_convs_filter_base, fine_layer_count=fine_layer_count)
    feature_descriptor_model = feature_descriptor_model.cuda()
    # Multi-GPU running
    feature_descriptor_model = torch.nn.DataParallel(feature_descriptor_model)
    feature_descriptor_model.eval()

    if feature_descriptor_model_path.exists():
        print("Loading {:s} ...".format(str(feature_descriptor_model_path)))
        state = torch.load(str(feature_descriptor_model_path))
        feature_descriptor_model.load_state_dict(state["model"])
    else:
        print("No trained model detected")
        raise OSError

    video_frame_filenames = get_all_color_image_names_in_sequence(sub_folder)
    print("Start gathering fusion data for {}".format(sub_folder))
    folder_list = get_all_subfolder_names(sfm_root, bag_range=(bag_range[0], bag_range[1]))
    print(folder_list)
    fusion_dataset = dataset.SfMDataset(image_file_names=video_frame_filenames,
                                        folder_list=folder_list,
                                        to_augment=False,
                                        transform=None,
                                        downsampling=downsampling,
                                        network_downsampling=network_downsampling, inlier_percentage=inlier_percentage,
                                        use_store_data=load_intermediate_data,
                                        store_data_root=precompute_root,
                                        phase="load_color_and_boundary", is_hsv=is_hsv,
                                        rgb_mode=rgb_mode,
                                        load_optimized_pose=False)
    fusion_loader = torch.utils.data.DataLoader(dataset=fusion_dataset, batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=batch_size)

    colors_list = []
    rough_feature_maps_list = []
    fine_feature_maps_list = []

    # Update progress bar
    tq = tqdm.tqdm(total=len(fusion_loader) * batch_size)
    for batch, (colors_1, boundaries, image_names,
                folders, starts_h, starts_w) in enumerate(fusion_loader):
        tq.update(batch_size)
        colors_1 = colors_1.cuda()
        boundaries = boundaries.cuda()
        colors_1 = boundaries * colors_1

        if batch == 0:
            boundary = boundaries[0].data.cpu().numpy()

        rough_feature_maps_1, fine_feature_maps_1 = feature_descriptor_model(colors_1)
        rough_feature_maps_1 = rough_feature_maps_1 / torch.norm(rough_feature_maps_1,
                                                                 dim=1, keepdim=True)
        fine_feature_maps_1 = fine_feature_maps_1 / torch.norm(fine_feature_maps_1,
                                                               dim=1, keepdim=True)
        start_h = starts_h[0]
        start_w = starts_w[0]
        for idx in range(colors_1.shape[0]):
            colors_list.append(colors_1[idx].data.cpu().numpy())
            rough_feature_maps_list.append(rough_feature_maps_1[idx].data.cpu().numpy())
            fine_feature_maps_list.append(fine_feature_maps_1[idx].data.cpu().numpy())
    tq.close()
    torch.cuda.empty_cache()
    return colors_list, boundary, rough_feature_maps_list, fine_feature_maps_list, start_h, start_w


# def read_colors(image_file_names):
#     for image_file_name in image_file_names:
#         image_path = Path(image_file_name)
#         folder = image_path.parent
#         # img_file_name = str(self.image_file_names[idx])
#         # Retrieve the folder path
#         #     folder = img_file_name[:-12]
#         # start_h, end_h, start_w, end_w = self.crop_positions_per_seq[folder]
#
#         color_img = utils.get_single_color_img(prefix_seq=folder, index=int(img_file_name[-12:-4]),
#                                                start_h=start_h, end_h=end_h,
#                                                start_w=start_w, end_w=end_w, downsampling_factor=self.downsampling,
#                                                is_hsv=self.is_hsv)
#         training_color_img_1 = color_img
#         height, width, _ = training_color_img_1.shape
#
#         training_mask_boundary = utils.type_float_and_reshape(
#             self.mask_boundary_per_seq[folder].astype(np.float32) / 255.0,
#             (height, width, 1))
#         training_mask_boundary[training_mask_boundary > 0.9] = 1.0
#         training_mask_boundary[training_mask_boundary <= 0.9] = 0.0
#
#         if self.to_augment:
#             if self.is_hsv:
#                 training_color_img_1 = cv2.cvtColor(np.uint8(training_color_img_1), cv2.COLOR_HSV2BGR_FULL)
#             augmented_1 = self.transform(image=training_color_img_1)
#             training_color_img_1 = augmented_1['image']
#             if self.is_hsv:
#                 training_color_img_1 = cv2.cvtColor(np.uint8(training_color_img_1),
#                                                     cv2.COLOR_BGR2HSV_FULL).astype('float32')
#             # Normalize
#             normalize = albu.Normalize(std=(0.5, 0.5, 0.5), mean=(0.5, 0.5, 0.5), max_pixel_value=255.0)
#             training_color_img_1 = normalize(image=training_color_img_1)['image']
#         else:
#             # Normalize
#             normalize = albu.Normalize(std=(0.5, 0.5, 0.5), mean=(0.5, 0.5, 0.5), max_pixel_value=255.0)
#             training_color_img_1 = normalize(image=training_color_img_1)['image']
#
#         return [img_to_tensor(training_color_img_1),
#                 img_to_tensor(training_mask_boundary),
#                 img_file_name[-12:-4], img_file_name[:-12], start_h, start_w]


def extract_and_write_keypoints(descriptor, colors_list, boundary, path, display_matches, downsampling,
                                start_h, start_w, height, width):
    if not display_matches:
        f_keypoints = open(path, "w")
    boundary = np.uint8(255 * boundary.reshape((height, width)))
    keypoints_list = []
    descriptions_list = []
    for i in range(len(colors_list)):
        color_1 = colors_list[i]
        color_1 = np.moveaxis(color_1, source=[0, 1, 2], destination=[2, 0, 1])
        color_1 = np.uint8(255 * (color_1 * 0.5 + 0.5))
        kps, des = descriptor.detectAndCompute(color_1, mask=boundary)
        keypoints_list.append(kps)
        descriptions_list.append(des)
        if not display_matches:
            f_keypoints.write("{},\n".format(len(kps)))
            for point in kps:
                f_keypoints.write("{},{},{},{},{}\n".format(downsampling * (start_w + point.pt[0]),
                                                            downsampling * (start_h + point.pt[1]),
                                                            point.size, point.angle, point.octave))
    if not display_matches:
        f_keypoints.close()
    return keypoints_list, descriptions_list


def extract_keypoints_rgb(descriptor, colors_list, boundary, height, width):
    keypoints_list = []
    descriptions_list = []
    keypoints_list_1D = []
    keypoints_list_2D = []

    boundary = np.uint8(255 * boundary.reshape((height, width)))
    for i in range(len(colors_list)):
        color_1 = colors_list[i]
        color_1 = np.moveaxis(color_1, source=[0, 1, 2], destination=[2, 0, 1])
        color_1 = cv2.cvtColor(np.uint8(255 * (color_1 * 0.5 + 0.5)), cv2.COLOR_RGB2BGR)
        kps, des = descriptor.detectAndCompute(color_1, mask=boundary)
        keypoints_list.append(kps)
        descriptions_list.append(des)
        temp = np.zeros((len(kps)))
        temp_2d = np.zeros((len(kps), 2))

        for j, point in enumerate(kps):
            temp[j] = np.round(point.pt[0]) + np.round(point.pt[1]) * width
            temp_2d[j, 0] = np.round(point.pt[0])
            temp_2d[j, 1] = np.round(point.pt[1])

        keypoints_list_1D.append(temp)
        keypoints_list_2D.append(temp_2d)
    return keypoints_list, keypoints_list_1D, keypoints_list_2D, descriptions_list


def extract_keypoints_raw(descriptor, colors_list, boundary, height, width):
    keypoints_list = []
    descriptions_list = []
    keypoints_list_1D = []
    keypoints_list_2D = []
    boundary = np.uint8(255 * boundary.reshape((height, width)))
    for i in range(len(colors_list)):
        color_1 = colors_list[i]
        kps, des = descriptor.detectAndCompute(color_1, mask=boundary)
        # print(len(kps))
        keypoints_list.append(kps)
        descriptions_list.append(des)
        temp = np.zeros((len(kps), 1))
        temp_2d = np.zeros((len(kps), 2))
        for i, point in enumerate(kps):
            temp[i, 0] = np.round(point.pt[0]) + np.round(point.pt[1]) * width
            temp_2d[i, 0] = np.round(point.pt[0])
            temp_2d[i, 1] = np.round(point.pt[1])

        keypoints_list_1D.append(temp)
        keypoints_list_2D.append(temp_2d)
    return keypoints_list, keypoints_list_1D, keypoints_list_2D, descriptions_list


def extract_keypoints(descriptor, colors_list, boundary, height, width):
    keypoints_list = []
    descriptions_list = []
    keypoints_list_1D = []
    boundary = np.uint8(255 * boundary.reshape((height, width)))
    for i in range(len(colors_list)):
        color_1 = colors_list[i]
        color_1 = np.moveaxis(color_1, source=[0, 1, 2], destination=[2, 0, 1])
        color_1 = cv2.cvtColor(np.uint8(255 * (color_1 * 0.5 + 0.5)), cv2.COLOR_HSV2BGR_FULL)
        kps, des = descriptor.detectAndCompute(color_1, mask=boundary)
        keypoints_list.append(kps)
        descriptions_list.append(des)
        temp = []
        for point in kps:
            temp.append(np.round(point.pt[0]) + np.round(point.pt[1]) * width)
        keypoints_list_1D.append(temp)

    return keypoints_list, keypoints_list_1D, descriptions_list


def memory_tracer():
    count = 0
    for obj in gc.get_objects():
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            count += 1

    print("{} tensors allocated".format(count))


def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> numpy.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> numpy.allclose(M, numpy.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> numpy.allclose(M, numpy.diag([1, -1, -1, 1]))
    True

    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < np.finfo(float).eps * 4.0:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
        [0.0, 0.0, 0.0, 1.0]])


def rotation_translation_perturbation(rotations, translations, angle_limit, trans_scale_limit):
    # rotations: B x 3 x 4
    # translations: B x 3
    rotations = rotations.view(-1, 3, 4)
    translations = translations.view(-1, 3)

    # Uniformly sample B x 3 unit vectors
    batch_size = rotations.shape[0]
    rotate_axies = torch.empty(batch_size, 3).normal_(mean=0, std=1.0).cuda()
    rotate_axies = rotate_axies / torch.norm(input=rotate_axies, p=2, dim=1, keepdim=True)
    # B x 1
    rotate_angle = torch.rand(batch_size, 1).cuda() * angle_limit
    # B x 3
    rot_vectors = rotate_axies * rotate_angle

    # B x 3 x 4
    random_rotations = tgm.angle_axis_to_rotation_matrix(rot_vectors)
    perturbed_rotations = torch.bmm(random_rotations[:, :3, :3], rotations)

    # B x 3
    translation_directions = torch.empty(batch_size, 3).normal_(mean=0, std=1.0).cuda()
    translation_directions = translation_directions / torch.norm(input=translation_directions, p=2, dim=1, keepdim=True)
    # B x 1
    translation_length = torch.norm(input=translations, p=2, dim=1, keepdim=True) * torch.rand(batch_size, 1).cuda()
    translation_length = trans_scale_limit * translation_length
    # B x 3
    random_translations = translation_length * translation_directions
    perturbed_translations = torch.bmm(random_rotations[:, :3, :3],
                                       translations.view(batch_size, 3, 1)) + random_translations.view(batch_size, 3, 1)

    return perturbed_rotations, perturbed_translations


if __name__ == "__main__":
    import imageio
    import os, sys


    class TargetFormat(object):
        GIF = ".gif"
        MP4 = ".mp4"
        AVI = ".avi"


    def convertFile(inputpath, targetFormat):
        """Reference: http://imageio.readthedocs.io/en/latest/examples.html#convert-a-movie"""
        outputpath = os.path.splitext(inputpath)[0] + targetFormat
        print("converting\r\n\t{0}\r\nto\r\n\t{1}".format(inputpath, outputpath))

        reader = imageio.get_reader(inputpath)
        fps = reader.get_meta_data()['fps']

        writer = imageio.get_writer(outputpath, fps=fps)
        for i, im in enumerate(reader):
            sys.stdout.write("\rframe {0}".format(i))
            sys.stdout.flush()
            writer.append_data(im)
        print("\r\nFinalizing...")
        writer.close()
        print("Done.")


    convertFile("/home/xliu89/tmp_ramfs/point_cloud_overlay_fm_only_spatial_grouping.avi", TargetFormat.GIF)

    # import h5py

    # fusion_data = h5py.File("/home/xliu89/RemoteData/Sinus Project Data/xingtong/FullLengthEndoscopicVIdeoData/bag_1/"
    #                         "_start_002603_end_002984_stride_1000_segment_00/feature_matches_DL.hdf5", 'r',
    #                         libver='latest')
    # print(fusion_data)
    # print(fusion_data["matches"].shape)

    # Generate heatmap
    # source_heatmap, target_heatmap = \
    #     generate_heatmap_from_locations(np.asarray([100, 100, 200, 200]).reshape((1, 4)), 300, 300, 10.0)
    #
    # target_colormap = cv2.applyColorMap(np.uint8(255 * target_heatmap).reshape((300, 300, 1)), cv2.COLORMAP_HOT)
    # cv2.imwrite("/home/xliu89/tmp_ramfs/heatmap_display.png", target_colormap)
    # import models
    # import torchsummary
    #
    # feature_descriptor_model = models.FCDenseNetNoPyramid(
    #     in_channels=3, down_blocks=(3, 3, 3, 3, 3, 3),
    #     up_blocks=(3, 3, 3, 3, 3, 3), bottleneck_layers=4,
    #     growth_rate=16, out_chans_first_conv=16, feature_length=256)
    # torchsummary.summary(feature_descriptor_model, input_size=(3, 256, 320), device="cpu")

    # cv2.imshow("", target_colormap)
    # cv2.waitKey()

    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # upper case - yl3
    # out = cv2.VideoWriter('output.avi', fourcc, 20, (1024, 1280))

    # boundary_list = []
    # for i in range(100):
    #     print(i)
    #     boundary_list.append(torch.from_numpy(np.zeros((128, 256, 320))).float().cuda())

    # import numpy as np
    # import matplotlib
    # import matplotlib.pyplot as plt
    # from distutils.version import LooseVersion
    # from scipy.stats import norm
    # from sklearn.neighbors import KernelDensity
    #
    # # from sklearn.datasets.species_distributions import construct_grids
    #
    # # Plot the progression of histograms to kernels
    # np.random.seed(1)
    # N = 20
    # X = np.concatenate((np.random.normal(0, 1, int(0.3 * N)),
    #                     np.random.normal(5, 1, int(0.7 * N))))[:, np.newaxis]
    #
    # X2 = np.concatenate((np.random.normal(0, 1, int(0.3 * N)),
    #                      np.random.normal(5, 1, int(0.7 * N))))[:, np.newaxis]
    #
    # X = np.hstack([X, X2])
    #
    # # X_plot = np.linspace(-5, 10, 100)[:, np.newaxis]
    #
    # # fig, ax = plt.subplots()
    # # Gaussian KDE
    # kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(X)
    # dens = kde.score_samples(X)
    # print(np.exp(dens))

    # log_dens = kde.score_samples(X_plot)
    # ax.fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
    # ax.text(-3.5, 0.31, "Gaussian Kernel Density")

    # for axi in ax.ravel():
    #     axi.plot(X[:, 0], np.full(X.shape[0], -0.01), '+k')
    #     axi.set_xlim(-4, 9)
    #     axi.set_ylim(-0.02, 0.34)
    #
    # for axi in ax[:, 0]:
    #     axi.set_ylabel('Normalized Density')
    #
    # for axi in ax[1, :]:
    #     axi.set_xlabel('x')
    # plt.show()
    # plt.savefig("/home/xliu89/figure.png")
    # indexes = np.arange(1000)
    # np.random.shuffle(indexes)
    # print(indexes)
    # img = cv2.imread(
    #     "/home/xliu89/RemoteData/Sinus Project Data/xingtong/EndoscopicVideoData/bag_2/_start_000285_end_000740_stride_25_segment_00/00000285.jpg")
    # img2 = cv2.imread(
    #     "/home/xliu89/RemoteData/Sinus Project Data/xingtong/EndoscopicVideoData/bag_2/_start_000285_end_000740_stride_25_segment_00/00000288.jpg")
    #
    # display_img, kp, des = orb_feature_detection(img)
    # display_img_2, kp_2, des_2 = orb_feature_detection(img2)
    # cv2.imshow("1", display_img)
    # cv2.imshow("2", display_img_2)
    #
    # # # create BFMatcher object
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # # # Match descriptors.
    # # matches = bf.match(des, des_2)
    # # # Sort them in the order of their distance.
    # # matches = sorted(matches, key=lambda x: x.distance)
    # # # Draw first 10 matches.
    # # img3 = cv2.drawMatches(img, kp, img2, kp_2, matches[:20], flags=2, outImg=None)
    # # cv2.namedWindow('matches', cv2.WINDOW_NORMAL)
    # # cv2.resizeWindow('matches', 2200, 1100)
    # # cv2.imshow("matches", img3)
    # cv2.waitKey()

    # rotation = rotation_matrix_from_rpy(torch.from_numpy(np.array([1.0, 2.0, 3.0]).reshape(1, 3)))
    # print(rotation.data.cpu().numpy())
    # rotation = rotation.data.cpu().numpy()
    # rpy = rotation_to_euler_angles(rotation.reshape(3, 3))
    #
    # rotation = rotation_matrix_from_rpy(torch.from_numpy(rpy).reshape(1, 3))
    # print(rotation)
    # # pass

    # print(torch.cos(torch.tensor(3.1415 / 2.0)))
    # array = np.array([1, 2, 3])
    # array2 = np.array([1, 2, 4])
    #
    #
    # # bin_array = np.zeros((len(array) * 8), dtype=int)
    # # print(bin(array))
    # # # [int(x) for x in bin(8)[2:]]
    # # count = 0
    # # for number in array:
    #
    # def hamming_distance_v2(a, b):
    #     r = (1 << np.arange(8))[:, None]
    #     return np.count_nonzero((np.bitwise_xor(a, b) & r) != 0)
    #
    #
    # print(hamming_distance_v2(array, array2))
    #
    # import moviepy.editor as mp
    #
    # clip = mp.VideoFileClip(
    #     "/home/xliu89/RemoteData/Sinus Project Data/xingtong/FullLengthEndoscopicVIdeoData/bag_1/_start_002603_end_002984_stride_1000_segment_00/point_cloud_overlay.gif")
    # clip.write_videofile(
    #     "/home/xliu89/RemoteData/Sinus Project Data/xingtong/FullLengthEndoscopicVIdeoData/bag_1/_start_002603_end_002984_stride_1000_segment_00/point_cloud_overlay.mp4")

    # def f(x):
    #     print(x)
    #     return x * x
    #
    #
    # #
    # # pool = mp.Pool(processes=4)  # start 4 worker processes
    # #
    # # # # print "[0, 1, 4,..., 81]"
    # # # print pool.map(f, range(10))
    # # # for i in pool.imap_unordered(f, range(10)):
    # # #     print i
    # #
    # # multiple_results = [pool.apply_async(f, (i,)) for i in range(4)]
    # # print [res.get(timeout=1) for res in multiple_results]
    #
    # mp.set_start_method('spawn')
    # queue_matches = Queue()
    # process_pool = []
    #
    # for i in range(16):
    #     temp = torch.tensor(i).cuda(i % 4)
    #     process_pool.append(Process(target=f, args=(temp,)))
    #
    # for i in range(16):
    #     process_pool[i].start()
    # image = cv2.imread("/home/xliu89/Downloads/my-visa-photo.jpg")
    # image = cv2.resize(image, dsize=(2180, 2180))
    # height, width, _ = image.shape
    # full_image = np.uint8(np.zeros((height * 2, width * 3, 3), dtype=np.float32))
    # full_image[height / 2 : -height / 2, int(0.25 * width):int(1.25 * width), :] = image
    # full_image[height / 2: -height / 2, int(1.75 * width):int(2.75 * width), :] = image
    # # cv2.imshow("full", full_image)
    # cv2.imwrite("/home/xliu89/Downloads/my-visa-photo-full.jpg", full_image)
    # path_1 = "/home/xliu89/RemoteData/Sinus Project Data/xingtong/Training/confidence_down_4.0_depth_7_base_3_inliner_0.998_hsv_True_bag_2/models/best_feature_matching_model.pt"
    # path_2 = "/home/xliu89/RemoteData/Sinus Project Data/xingtong/Training/confidence_down_4.0_depth_7_base_3_inliner_0.998_hsv_True_bag_2/models/best_feature_matching_model_py3.pt"
    # state = torch.load(path_1)
    # torch.save(state['model'], path_2)
