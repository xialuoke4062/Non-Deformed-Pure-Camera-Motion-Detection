import os

import cv2
import subprocess
import numpy as np
from pathlib import Path
import time
# from albumentations.pytorch.functional import img_to_tensor
# import torch
from skimage import data
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import tqdm
import shutil


def colmap_generator(video_path):
    print(video_path)
    start_time = time.time()

    colmap_path = Path(str(video_path)[:-4] + "COLMAP")
    frame_path = Path(str(video_path)[:-4])
    db_path = Path(str(colmap_path) + "/database.db")
    flag_path = Path(str(colmap_path) + "/flag.txt")
    colmap_alg_path = "/Users/apple/Downloads/COLMAP_2.app/Contents/MacOS/colmap"
    # colmap_alg_path = "/Users/xwang169/Downloads/COLMAP.app/Contents/MacOS/colmap"
    # colmap_alg_path = "/Users/xwang169/Downloads/COLMAP_2.app/Contents/MacOS/colmap"
    stats_path = "/Users/apple/Desktop/Non-Deformed-Pure-Camera-Motion-Detection/COLMAP/time.txt"
    stats_path = "/Users/xwang169/Downloads/Non-Deformed-Pure-Camera-Motion-Detection/COLMAP/time.txt"
    stats_path = "/Users/apple/Desktop/Xingtong_2/www.gastrointestinalatlas.com/videos/A/A/time.txt"

    if flag_path.exists():
        if frame_path.exists():
            shutil.rmtree(frame_path)
        if db_path.exists():
            os.remove(db_path)
            with open(stats_path, 'a') as the_file3:
                the_file3.write("Deleted"+str(db_path)+'\n\n')
        return

    if colmap_path.exists():
        shutil.rmtree(colmap_path)
    colmap_path.mkdir()

    vidcap = cv2.VideoCapture(str(video_path))
    if not frame_path.exists():
        frame_path.mkdir()
        ### Collecting Frames ###
        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(str(frame_path / "frame{}.jpg".format(count)), image)  # save frame as JPEG file
            success, image = vidcap.read()
            count += 1
    # else:
    #     return
    
    ### COLMAP ###
    subprocess.run([colmap_alg_path, "automatic_reconstructor",
                    "--image_path", str(frame_path), "--workspace_path", str(colmap_path),
                    "--data_type", "video", "--single_camera", "1", "--use_gpu", "0", "--gpu_index", "0"])

    ### Convert Bin to Txt ###
    sparse_path = Path(str(colmap_path) + "/sparse/")
    for f in os.scandir(sparse_path):
        if f.is_dir():
            subprocess.run([colmap_alg_path, "model_converter",
                            "--input_path", str(sparse_path / f),
                            "--output_path", str(sparse_path / f),
                            "--output_type", "TXT"])

    ### Stats ###
    images, points, ratios = [], [], []
    sparse_path = Path(str(colmap_path)+"/sparse/")
    for f in os.scandir(sparse_path):
        if f.is_dir():
            f_image = open(str(sparse_path / f)+"/images.txt", "r")
            f_point = open(str(sparse_path / f) + "/points3D.txt", "r")
            for _ in range(3):
                f_image.readline()
            image = int(f_image.readline().split()[4][:-1])
            images.append(image)
            for _ in range(2):
                f_point.readline()
            point = int(f_point.readline().split()[4][:-1])
            points.append(point)
            ratios.append(point/image)
    mean_img, mean_point, mean_ratio = 0, 0, 0
    if len(images) != 0 and len(points) != 0 and len(ratios) != 0:
        mean_img, mean_point, mean_ratio = sum(images)/len(images), sum(points)/len(points), sum(ratios)/len(ratios)
    print(mean_img, mean_point, mean_ratio)

    with open(flag_path, 'a') as the_file2:
        the_file2.write("The sparse reconstruction is completed."+"\n\n")

    with open(stats_path, 'a') as the_file:
        the_file.write("video_path: {}\n"
                       "frames: {}, time: {:.2f}\nimages: {}\npoints: {}\nratios: {}\nmean_img: {}\nmean_point: {}\n"
                       "mean_ratio: {}\n\n".
                       format(video_path, vidcap.get(cv2.CAP_PROP_FRAME_COUNT), time.time() - start_time, images, points, ratios,
                              mean_img, mean_point, mean_ratio))

    ### Delete Frame Images and COLMAP Database ###
    if frame_path.exists():
        shutil.rmtree(frame_path)
    if db_path.exists():
        os.remove(db_path)
    with open(stats_path, 'a') as the_file4:
        the_file4.write("Deleted"+str(db_path)+"\n\n")

if __name__ == "__main__":
    video_root = Path("/Users/apple/Desktop/Non-Deformed-Pure-Camera-Motion-Detection/COLMAP/testing_videos")
    video_root = Path("/Users/xwang169/Downloads/Non-Deformed-Pure-Camera-Motion-Detection/COLMAP/A")
    video_root = Path("/Users/apple/Desktop/Xingtong_2/www.gastrointestinalatlas.com/videos/A/A")
    # with open(stats_path, 'w') as the_file:
    #     pass
    whole_video = False
    if whole_video:
        video_path_list = list(video_root.glob("*.mp4"))
    else:
        video_path_list = []
        for f in os.scandir(video_root):
            if f.is_dir():
                temp = list(Path(str(video_root/f)).glob("*.mp4"))
                video_path_list.extend(temp)
    c = cpu_count()
    pool = Pool(c)
    pool.map(colmap_generator, [video_path for video_path in video_path_list])
    pool.close()
