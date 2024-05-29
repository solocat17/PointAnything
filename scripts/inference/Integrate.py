import argparse
import cv2
import numpy as np
import os

def depth_map_inference(depth_map, x, y, pred_depth, threshold):
    n = 1
    if min(depth_map.shape[0], depth_map.shape[1]) > 500:
        n = 2
    if min(depth_map.shape[0], depth_map.shape[1]) > 2500:
        n = 3
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            if x + i < 0 or x + i >= depth_map.shape[0] or y + j < 0 or y + j >= depth_map.shape[1]:
                continue
            if abs(int(depth_map[x + i, y + j]) - pred_depth) < threshold:
                return True
    return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-name', type=str, default='spiderman.jpg', help='name of the image in /data/input/')
    parser.add_argument('--threshold', type=int, default=25, help='threshold for depth change')
    args = parser.parse_args()

    img_name_without_ext = os.path.splitext(args.img_name)[0]
    img = cv2.imread(f'../../data/input/{args.img_name}')
    depth = cv2.imread(f'../../data/intermediate/depth_map/{img_name_without_ext}_depth.png', cv2.IMREAD_GRAYSCALE)

    arm_vector1 = np.load(f'../../data/intermediate/arm_vectors/{img_name_without_ext}/arm1.npy')
    arm_vector2 = np.load(f'../../data/intermediate/arm_vectors/{img_name_without_ext}/arm2.npy')
    
    arm_vector1 = arm_vector1.astype(int)
    arm_vector1 = [(arm_vector1[0][1], arm_vector1[0][0]), (arm_vector1[1][1], arm_vector1[1][0])]
    # print(arm_vector1)
    arm_dir_1 = [arm_vector1[1][0] - arm_vector1[0][0], arm_vector1[1][1] - arm_vector1[0][1]]

    arm_vector2 = arm_vector2.astype(int)
    arm_vector2 = [(arm_vector2[0][1], arm_vector2[0][0]), (arm_vector2[1][1], arm_vector2[1][0])]
    # print(arm_vector2)
    arm_dir_2 = [arm_vector2[1][0] - arm_vector2[0][0], arm_vector2[1][1] - arm_vector2[0][1]]

    depth_elbow_1 = int(depth[arm_vector1[0]])
    depth_wrist_1 = int(depth[arm_vector1[1]])
    # print(f'Arm 1: Elbow depth: {depth_elbow_1}, Wrist depth: {depth_wrist_1}')
    depth_dir_1 = depth_wrist_1 - depth_elbow_1
    # print(f'Arm 1: Depth direction: {depth_dir_1}')

    depth_elbow_2 = int(depth[arm_vector2[0]])
    depth_wrist_2 = int(depth[arm_vector2[1]])
    # print(f'Arm 2: Elbow depth: {depth_elbow_2}, Wrist depth: {depth_wrist_2}')
    depth_dir_2 = depth_wrist_2 - depth_elbow_2
    # print(f'Arm 2: Depth direction: {depth_dir_2}')
    
    # normalize the arm direction vectors
    depth_dir_1 = depth_dir_1 / np.linalg.norm(arm_dir_1)
    normalized_arm_dir_1 = arm_dir_1 / np.linalg.norm(arm_dir_1)
    # print(normalized_arm_dir_1)
    depth_dir_2 = depth_dir_2 / np.linalg.norm(arm_dir_2)
    normalized_arm_dir_2 = arm_dir_2 / np.linalg.norm(arm_dir_2)
    # print(normalized_arm_dir_2)
    
    # for below, only dealing with arm 1
    t = 1
    threshold = args.threshold
    while True:
        x = arm_vector1[1][0] + int(t * normalized_arm_dir_1[0])
        y = arm_vector1[1][1] + int(t * normalized_arm_dir_1[1])
        t = t + 1
        if x < 0 or x >= depth.shape[0] or y < 0 or y >= depth.shape[1]:
            break
        if x == arm_vector1[1][0] and y == arm_vector1[1][1]:
            continue
        d = int(depth[x, y])
        pred_d = depth_wrist_1 + int(t * depth_dir_1)
        if abs(d - pred_d) > threshold:
            break
    t *= 2
    threshold *= 3
    img_arm1 = img.copy()
    while True:
        x = arm_vector1[1][0] + int(t * normalized_arm_dir_1[0])
        y = arm_vector1[1][1] + int(t * normalized_arm_dir_1[1])
        t = t + 1
        if x == arm_vector1[1][0] and y == arm_vector1[1][1]:
            continue
        if x < 0 or x >= depth.shape[0] or y < 0 or y >= depth.shape[1]:
            break
        d = int(depth[x, y])
        pred_d = depth_wrist_1 + int(t * depth_dir_1)
        if depth_map_inference(depth, x, y, pred_d, threshold):
            print(f'Arm 1: The first point where the depth changes is at ({x}, {y})')
            img_arm1 = cv2.circle(img_arm1, (y, x), 5, (0, 255, 0), -1)
            np.save(f'../../data/intermediate/Integrator/{img_name_without_ext}/arm1.npy', np.array([x, y]))
            break
        else:
            img_arm1 = cv2.circle(img_arm1, (y, x), 2, (0, 127, 0), -1)


    # for below, only dealing with arm 2
    t = 10
    threshold = args.threshold
    while True:
        x = arm_vector2[1][0] + int(t * normalized_arm_dir_2[0])
        y = arm_vector2[1][1] + int(t * normalized_arm_dir_2[1])
        t = t + 1
        if x < 0 or x >= depth.shape[0] or y < 0 or y >= depth.shape[1]:
            break
        if x == arm_vector2[1][0] and y == arm_vector2[1][1]:
            continue
        d = int(depth[x, y])
        pred_d = depth_wrist_2 + int(t * depth_dir_2)
        if abs(d - pred_d) > threshold:
            break
    t *= 2
    threshold *= 3
    img_arm2 = img.copy()
    while True:
        x = arm_vector2[1][0] + int(t * normalized_arm_dir_2[0])
        y = arm_vector2[1][1] + int(t * normalized_arm_dir_2[1])
        t = t + 1
        if x == arm_vector2[1][0] and y == arm_vector2[1][1]:
            continue
        if x < 0 or x >= depth.shape[0] or y < 0 or y >= depth.shape[1]:
            break
        d = int(depth[x, y])
        pred_d = depth_wrist_2 + int(t * depth_dir_2)
        if depth_map_inference(depth, x, y, pred_d, threshold):
            print(f'Arm 2: The first point where the depth changes is at ({x}, {y})')
            img_arm2 = cv2.circle(img_arm2, (y, x), 5, (0, 255, 0), -1)
            np.save(f'../../data/intermediate/Integrator/{img_name_without_ext}/arm2.npy', np.array([x, y]))
            break
        else:
            img_arm2 = cv2.circle(img_arm2, (y, x), 2, (0, 127, 0), -1)
    
    os.makedirs(f'../../data/intermediate/Integrator/{img_name_without_ext}', exist_ok=True)
    os.makedirs(f'../../data/intermediate/Integrator/{img_name_without_ext}/images', exist_ok=True)
    cv2.imwrite(f'../../data/intermediate/Integrator/{img_name_without_ext}/images/arm1.png', img_arm1)
    cv2.imwrite(f'../../data/intermediate/Integrator/{img_name_without_ext}/images/arm2.png', img_arm2)

    