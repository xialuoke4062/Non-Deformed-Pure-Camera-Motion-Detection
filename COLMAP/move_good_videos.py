import multiprocessing
import os
import sys
import shutil
import matplotlib.pyplot as plt
import numpy as np

# def copytree(src, dst, symlinks=False, ignore=None):
#     for item in os.listdir(src):
#         s = os.path.join(src, item)
#         d = os.path.join(dst, item)
#         if os.path.isdir(s):
#             shutil.copytree(s, d, symlinks, ignore)
#         else:
#             shutil.copy2(s, d)

if __name__ == "__main__":
    stats_path = "/Users/apple/Desktop/Non-Deformed-Pure-Camera-Motion-Detection/COLMAP/time.txt"
    stats_path = "/Users/xwang169/Downloads/Non-Deformed-Pure-Camera-Motion-Detection/COLMAP/time.txt"
    stats_path = "/Users/apple/Desktop/Non-Deformed-Pure-Camera-Motion-Detection/COLMAP/testing_videos/" \
                 "THRESHOLD_70/time.txt"
    threshold, all_ratios = 50, []
    if len(sys.argv) > 1:
        sys.argv[1]
    dest_root = "/Users/xwang169/Desktop/Good_videos/"
    # if not os.path.exists(dest_root):
    #     dest_root.mkdir()
    with open(stats_path) as f_stats:
        video_path, ratios, video_name, video_num = "", "", "", ""
        good, total = 0, 0
        for count, line in enumerate(f_stats):
            if count % 9 == 0:
                video_path = line.strip()[12:]
                video_name = video_path.split("/")[-2]
                video_num = video_path.split("/")[-1][:-4]
                total += 1
            if count % 9 == 4:
                ratios = line[9:-2].split(', ')
                for ratio in ratios:
                    all_ratios.append((float(ratio)))
                    if float(ratio) > threshold:
                        # sparse_path = video_path[:-4]+"COLMAP/sparse/"
                        # copytree_des = dest_root+video_name+"/"+video_num+"COLMAP"
                        # print(copytree_des)
                        # shutil.copytree(sparse_path, copytree_des)
                        # shutil.copy(video_path, dest_root+video_name)
                        good += 1
                        # break
        stats = "If threshold = {}, then good: {}, bad: {}, total: {}".format(threshold, good, total-good, total)
        # with open(dest_root+"/stats.txt", 'w') as the_file:
        #     the_file.write(stats)
    plt.figure()
    all_ratios = sorted(all_ratios)
    print(all_ratios)
    left = np.arange(len(all_ratios))
    # plt.plot(np.arange(len(all_ratios)), all_ratios)
    plt.hist(all_ratios, bins=20, range=(0, 100), color='green',
             histtype='bar', rwidth=0.8)
    plt.savefig(str(stats_path[:-8] + "/histogram.png"))
    plt.show()
    plt.close()