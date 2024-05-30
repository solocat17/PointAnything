import argparse
import cv2
import numpy as np
import os
from ultralytics import YOLOv10
import supervision as sv
import itertools

def depth_map_inference(depth_map, x, y, pred_depth, threshold):
    n = 10
    isin = 0
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            if x + i < 0 or x + i >= depth_map.shape[0] or y + j < 0 or y + j >= depth_map.shape[1]:
                continue
            if abs(int(depth_map[x + i, y + j]) - pred_depth) < threshold:
                isin = 1
    posx, posy = x, y
    min_diff = 100000
    if isin:
        for i in range(-n, n+1):
            for j in range(-n, n+1):
                if x + i < 0 or x + i >= depth_map.shape[0] or y + j < 0 or y + j >= depth_map.shape[1]:
                    continue
                diff = abs(int(depth_map[x + i, y + j]) - pred_depth)
                if diff < min_diff:
                    min_diff = diff
                    posx, posy = x + i, y + j
    return isin, posx, posy

def get_arm_depth_vec(depth, arm_vector):
    arm_vector = [(arm_vector[0][1], arm_vector[0][0]), (arm_vector[1][1], arm_vector[1][0])]
    arm_dir = [arm_vector[1][0] - arm_vector[0][0], arm_vector[1][1] - arm_vector[0][1]]

    depth_elbow = int(depth[arm_vector[0]])
    depth_wrist = int(depth[arm_vector[1]])
    depth_dir = depth_wrist - depth_elbow

    depth_dir = depth_dir / np.linalg.norm(arm_dir)
    normalized_arm_dir = arm_dir / np.linalg.norm(arm_dir)
    return arm_vector, depth_elbow, depth_wrist, depth_dir, normalized_arm_dir

def get_inference_point(img, img_name_without_ext, ith_arm, depth, arm_vector, thres=25):
    
    arm_vector, depth_elbow, depth_wrist, depth_dir, normalized_arm_dir = get_arm_depth_vec(depth, arm_vector)
    

    os.makedirs(f'../../data/intermediate/Integrator/{img_name_without_ext}', exist_ok=True)
    os.makedirs(f'../../data/intermediate/Integrator/{img_name_without_ext}/images', exist_ok=True)
    img_annotate = img.copy()
    t = 1
    threshold = thres
    while True:
        x = arm_vector[1][0] + int(t * normalized_arm_dir[0])
        y = arm_vector[1][1] + int(t * normalized_arm_dir[1])
        t = t + 1
        if x < 0 or x >= depth.shape[0] or y < 0 or y >= depth.shape[1]:
            break
        if x == arm_vector[1][0] and y == arm_vector[1][1]:
            continue
        d = int(depth[x, y])
        pred_d = depth_wrist + int(t * depth_dir)
        if abs(d - pred_d) > threshold:
            break
    t *= 2
    threshold *= 3
    while True:
        x = arm_vector[1][0] + int(t * normalized_arm_dir[0])
        y = arm_vector[1][1] + int(t * normalized_arm_dir[1])
        t = t + 1
        if x < 0 or x >= depth.shape[0] or y < 0 or y >= depth.shape[1]:
            break
        if x == arm_vector[1][0] and y == arm_vector[1][1]:
            continue
        pred_d = depth_wrist + int(t * depth_dir)
        pred_d = min(max(pred_d, 0), 255)
        isin, posx, posy = depth_map_inference(depth, x, y, pred_d, threshold)
        if isin:
            print(f'Arm {ith_arm}: The first point where the depth changes is at ({posx}, {posy})')
            img_annotate = cv2.circle(img_annotate, (posy, posx), 5, (0, 255, 0), -1)
            np.save(f'../../data/intermediate/Integrator/{img_name_without_ext}/arm{ith_arm}.npy', np.array([posx, posy]))
            break
        else:
            img_annotate = cv2.circle(img_annotate, (y, x), 2, (0, 127, 0), -1)
    cv2.imwrite(f'../../data/intermediate/Integrator/{img_name_without_ext}/images/arm{ith_arm}.png', img_annotate)
    return [posy, posx]

def if_in_box(x, y, x1, y1, x2, y2):
    return x >= x1 and x <= x2 and y >= y1 and y <= y2

def find_important_point(depth, x1, y1, x2, y2, thres=5):
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
    
    # find the most appearing depth value in the bounding box
    # more robust but slower if taking threshold into account
    # further experiments needed
    '''
    thres //= 2
    points = list(itertools.product(range(x1, x2+1), range(y1, y2+1), range(-thres, thres+1)))
    freq = {}
    for i, j, k in points:
        if depth[j, i] + k > 255 or depth[j, i] + k < 0:
            continue
        if depth[j, i] + k not in freq:
            freq[depth[j, i] + k] = 0
        freq[depth[j, i] + k] += 1
    '''

    # '''
    points = list(itertools.product(range(x1, x2+1), range(y1, y2+1)))
    freq = {}
    for i, j in points:
        if depth[j, i] not in freq:
            freq[depth[j, i]] = 0
        freq[depth[j, i]] += 1
    # '''

    max_freq = max(freq.values())
    freq_depth = [k for k, v in freq.items() if v == max_freq][0]

    points = list(itertools.product(range(x1, x2+1), range(y1, y2+1)))
    points = [(i, j) for i, j in points if depth[j, i] == freq_depth]
    xavg = sum([i for i, j in points]) // len(points)
    yavg = sum([j for i, j in points]) // len(points)
    return int(xavg), int(yavg)

def distance_point_to_line(p, l1, l2):
    # distance from point p to line l1l2
    return np.linalg.norm(np.cross(np.array([l2[0] - l1[0], l2[1] - l1[1]]), np.array([p[0] - l1[0], p[1] - l1[1]])) / np.linalg.norm(np.array([l2[0] - l1[0], l2[1] - l1[1]])))

def get_bounding_boxes(img, depth, thres=5):
    model = YOLOv10('../../models/yolov10s.pt')
    results = model(img, conf=0.25)[0]
    possible_bounding_boxes = []
    for result in results:
        x1, y1, x2, y2 = result.boxes.data[0].cpu().numpy()[:4]
        targetx, targety = find_important_point(depth, x1, y1, x2, y2, thres)
        possible_bounding_boxes.append([x1, y1, x2, y2, targetx, targety, result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()])
    return possible_bounding_boxes

def detect_object_with_point(img, inference_point, img_name_without_ext, possible_bounding_boxes, num_arm, wrist_pos, classes):
    os.makedirs(f'../../data/output/{img_name_without_ext}', exist_ok=True)
    # L is the line passing through the wrist position and the target point
    # sort the bounding boxes by the distance(inference, L), the smaller the distance, the more likely the bounding box contains the object
    possible_bounding_boxes = sorted(possible_bounding_boxes, key=lambda x: distance_point_to_line(inference_point, wrist_pos, [x[4], x[5]]))
    for i, (x1, y1, x2, y2, targetx, targety, conf, cls) in enumerate(possible_bounding_boxes):
        if if_in_box(inference_point[0], inference_point[1], x1, y1, x2, y2):
            x1, y1, x2, y2, targetx, targety, conf, cls = possible_bounding_boxes[i]
            break
    
    item_class = classes[int(cls)]
    item_class = item_class + f': {conf[0]:.2f}'

    img_cp = img.copy()
    img_cp = cv2.circle(img_cp, (int(inference_point[0]), int(inference_point[1])), 10, (255, 0, 0), -1)
    img_cp = cv2.rectangle(img_cp, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 10)
    img_cp = cv2.rectangle(img_cp, (int(max(x1 - 5, 0)), int(min(y1 + 5, img.shape[1]))), (int(min(x1 + 250, img.shape[0])), int(max(y1 - 25, 0))), (0, 255, 0), -1)
    img_cp = cv2.putText(img_cp, item_class, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imwrite(f'../../data/output/{img_name_without_ext}/detected_arm{num_arm}.png', img_cp)
    return item_class

def draw_bounding_box(img, possible_bounding_boxes, classes):
    img_cp = img.copy()
    for x1, y1, x2, y2, targetx, targety, conf, cls in possible_bounding_boxes:
        img_cp = cv2.rectangle(img_cp, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 10)
        img_cp = cv2.rectangle(img_cp, (int(max(x1 - 5, 0)), int(min(y1 + 5, img.shape[1]))), (int(min(x1 + 250, img.shape[0])), int(max(y1 - 25, 0))), (0, 255, 0), -1)
        img_cp = cv2.putText(img_cp, classes[int(cls)], (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imwrite(f'../../data/intermediate/Integrator/{img_name_without_ext}/images/bounding_boxes.png', img_cp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-name', type=str, default='spiderman.jpg', help='name of the image in /data/input/')
    parser.add_argument('--threshold', type=int, default=25, help='threshold for depth change')
    args = parser.parse_args()

    img_name_without_ext = os.path.splitext(args.img_name)[0]
    img = cv2.imread(f'../../data/input/{args.img_name}')
    depth = cv2.imread(f'../../data/intermediate/depth_map/{img_name_without_ext}_depth.png', cv2.IMREAD_GRAYSCALE)

    arm_vector1 = np.load(f'../../data/intermediate/arm_vectors/{img_name_without_ext}/arm1.npy').astype(int)
    arm_vector2 = np.load(f'../../data/intermediate/arm_vectors/{img_name_without_ext}/arm2.npy').astype(int)
    
    inference_point_1 = get_inference_point(img, img_name_without_ext, 1, depth, arm_vector1, args.threshold)
    inference_point_2 = get_inference_point(img, img_name_without_ext, 2, depth, arm_vector2, args.threshold)

    with open(f'../../yolov10/ultralytics/cfg/datasets/coco.yaml', 'r+') as f:
        while True:
            line = f.readline()
            if line.startswith('names'):
                break
        classes = {i: line.strip() for i, line in enumerate(f.readlines()) if not line.startswith('#')}
    # remove classes after 79
    classes = {k: v for k, v in classes.items() if k <= 79}
    classes = {k: v.split(' ')[1] if ' ' in v else v for k, v in classes.items()}
    print('Classes loaded.')

    possible_bounding_boxes = get_bounding_boxes(img, depth, thres=5)
    print("possible bounding boxes and their target points are computed.")
    if len(possible_bounding_boxes) == 0:
        print('No object detected')
        exit(0)
    
    draw_bounding_box(img, possible_bounding_boxes, classes)

    result1 = detect_object_with_point(img, inference_point_1, img_name_without_ext, possible_bounding_boxes, 1, arm_vector1[1], classes)
    result2 = detect_object_with_point(img, inference_point_2, img_name_without_ext, possible_bounding_boxes, 2, arm_vector2[1], classes)
    print(f'Arm 1: {result1}')
    print(f'Arm 2: {result2}')