# import argparse
# import cv2
# import numpy as np
# import os
# from ultralytics import YOLOv10
# import supervision as sv
# import itertools
# import cv2
# import torch
# import numpy as np
# import detectron2
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2 import model_zoo
# from detectron2.utils.visualizer import Visualizer, ColorMode
# from detectron2.data import MetadataCatalog
# from PIL import Image
# # yan lee
# def depth_map_inference(depth_map, x, y, pred_depth, threshold):
#     n = 10
#     isin = False
#     for i in range(-n, n+1):
#         for j in range(-n, n+1):
#             if x + i < 0 or x + i >= depth_map.shape[0] or y + j < 0 or y + j >= depth_map.shape[1]:
#                 continue
#             if abs(int(depth_map[x + i, y + j]) - pred_depth) < threshold:
#                 isin = 1
#     posx, posy = x, y
#     min_diff = 100000
#     if isin:
#         for i in range(-n, n+1):
#             for j in range(-n, n+1):
#                 if x + i < 0 or x + i >= depth_map.shape[0] or y + j < 0 or y + j >= depth_map.shape[1]:
#                     continue
#                 diff = abs(int(depth_map[x + i, y + j]) - pred_depth)
#                 if diff < min_diff:
#                     min_diff = diff
#                     posx, posy = x + i, y + j
#     return isin, posx, posy

# def get_arm_depth_vec(depth, arm_vector):
#     arm_vector = [(arm_vector[0][1], arm_vector[0][0]), (arm_vector[1][1], arm_vector[1][0])]
#     arm_dir = [arm_vector[1][0] - arm_vector[0][0], arm_vector[1][1] - arm_vector[0][1]]

#     depth_elbow = int(depth[arm_vector[0]])
#     depth_wrist = int(depth[arm_vector[1]])
#     depth_dir = depth_wrist - depth_elbow

#     depth_dir = depth_dir / np.linalg.norm(arm_dir)
#     normalized_arm_dir = arm_dir / np.linalg.norm(arm_dir)
#     return arm_vector, depth_elbow, depth_wrist, depth_dir, normalized_arm_dir

# def get_inference_point(img, img_name_without_ext, ith_arm, depth, arm_vector, thres=25):
    
#     arm_vector, depth_elbow, depth_wrist, depth_dir, normalized_arm_dir = get_arm_depth_vec(depth, arm_vector)
    

#     os.makedirs(f'../../data/intermediate/Integrator/{img_name_without_ext}', exist_ok=True)
#     os.makedirs(f'../../data/intermediate/Integrator/{img_name_without_ext}/images', exist_ok=True)
#     img_annotate = img.copy()
#     # t = 1
#     # threshold = thres
#     # while True:
#     #     x = arm_vector[1][0] + int(t * normalized_arm_dir[0])
#     #     y = arm_vector[1][1] + int(t * normalized_arm_dir[1])
#     #     t = t + 1
#     #     if x < 0 or x >= depth.shape[0] or y < 0 or y >= depth.shape[1]:
#     #         print(f"Arm {ith_arm}: The arm is out of the image in first while loop.")
#     #         break
#     #     if x == arm_vector[1][0] and y == arm_vector[1][1]:
#     #         continue
#     #     d = int(depth[x, y])
#     #     pred_d = depth_wrist + int(t * depth_dir)
#     #     if abs(d - pred_d) > threshold:
#     #         break
#     # t *= 2
#     # threshold *= 3
#     # while True:
#     #     x = arm_vector[1][0] + int(t * normalized_arm_dir[0])
#     #     y = arm_vector[1][1] + int(t * normalized_arm_dir[1])
#     #     t = t + 1
#     #     if x < 0 or x >= depth.shape[0] or y < 0 or y >= depth.shape[1]:
#     #         print(f"Arm {ith_arm}: The arm is out of the image in while loop.")
#     #         break
#     #     if x == arm_vector[1][0] and y == arm_vector[1][1]:
#     #         continue
#     #     pred_d = depth_wrist + int(t * depth_dir)
#     #     pred_d = min(max(pred_d, 0), 255)
#     #     isin, posx, posy = depth_map_inference(depth, x, y, pred_d, threshold)
#     #     if isin:
#     #         print(f'Arm {ith_arm}: The first point where the depth changes is at ({posx}, {posy})')
#     #         img_annotate = cv2.circle(img_annotate, (posy, posx), 20, (0, 255, 0), -1)
#     #         np.save(f'../../data/intermediate/Integrator/{img_name_without_ext}/arm{ith_arm}.npy', np.array([posx, posy]))
#     #         break
#     #     else:
#     #         img_annotate = cv2.circle(img_annotate, (y, x), 20, (0, 127, 0), -1)

#     t = 0
#     threshold = thres
#     # set starting point as the wrist plus arm vector
#     start_x = arm_vector[1][0] + 1*int(arm_vector[1][0] - arm_vector[0][0])
#     start_y = arm_vector[1][1] + 1*int(arm_vector[1][1] - arm_vector[0][1])
#     start_depth = int(depth[int(round(start_x)), int(round(start_y))])
#     while True:
#         x = start_x+ int(t * normalized_arm_dir[0])
#         y = start_y + int(t * normalized_arm_dir[1])
        
#         if x < 0 or x >= depth.shape[0] or y < 0 or y >= depth.shape[1]:
#             print(f"Arm {ith_arm}: The arm is out of the image in while loop.")
#             break

#         pred_d = start_depth+ int(t * depth_dir)
#         pred_d = min(max(pred_d, 0), 255)
#         isin, posx, posy = depth_map_inference(depth, int(round(x)), int(round(y)), pred_d, threshold)
#         if isin:
#             print(f'Arm {ith_arm}: The first point where the depth changes is at ({posx}, {posy})')
#             img_annotate = cv2.circle(img_annotate, (posy, posx), 20, (0, 255, 0), -1)
#             np.save(f'../../data/intermediate/Integrator/{img_name_without_ext}/arm{ith_arm}.npy', np.array([posx, posy]))
#             break
#         else:
#             img_annotate = cv2.circle(img_annotate, (y, x), 20, (0, 127, 0), -1)
#         t = t + 1

#     # cv2.imwrite(f'../../data/intermediate/Integrator/{img_name_without_ext}/images/arm{ith_arm}.png', img_annotate)
#     # return [posy, posx] 

# def get_inference_point_v2(img, img_name_without_ext, ith_arm, depth, eye_center, finger_pos, thres=5):
#     arm_vector = np.array([eye_center, finger_pos])
#     arm_dir = np.array([arm_vector[1][0] - arm_vector[0][0], arm_vector[1][1] - arm_vector[0][1]])

#     depth_eye = int(depth[round(eye_center[0]), round(eye_center[1])])
#     depth_finger = int(depth[round(finger_pos[0]), round(finger_pos[1])])
#     depth_dir = depth_finger - depth_eye
    
#     print("hello v2")

#     # 射线方向的单位向量
#     depth_dir = depth_dir / np.linalg.norm(arm_dir)
#     normalized_arm_dir = arm_dir / np.linalg.norm(arm_dir)

#     os.makedirs(f'../../data/intermediate/Integrator/{img_name_without_ext}', exist_ok=True)
#     os.makedirs(f'../../data/intermediate/Integrator/{img_name_without_ext}/images', exist_ok=True)
#     img_annotate = img.copy()
#     # t = 1
#     # threshold = thres
#     # img_annotate = img.copy()

#     # while True:
#     #     x = arm_vector[1][0] + int(t * normalized_arm_dir[0])
#     #     y = arm_vector[1][1] + int(t * normalized_arm_dir[1])
#     #     t = t + 1
#     #     if x < 0 or x >= depth.shape[0] or y < 0 or y >= depth.shape[1]:
#     #         print(f"Arm{ith_arm}: The arm is out of the image in first while loop.")
#     #         break
#     #     if x == arm_vector[1][0] and y == arm_vector[1][1]:
#     #         continue
#     #     d = int(depth[x, y])
#     #     pred_d = depth_finger + int(t * depth_dir)
#     #     if abs(d - pred_d) > threshold:
#     #         break

#     # while True:
#     #     x = arm_vector[1][0] + int(t * normalized_arm_dir[0])
#     #     y = arm_vector[1][1] + int(t * normalized_arm_dir[1])
#     #     t = t + 1
#     #     if x < 0 or x >= depth.shape[0] or y < 0 or y >= depth.shape[1]:
#     #         print(f"Arm{ith_arm}: The arm is out of the image in second while loop.")
#     #         break
#     #     if x == arm_vector[1][0] and y == arm_vector[1][1]:
#     #         continue
#     #     pred_d = depth_finger + int(t * depth_dir)
#     #     pred_d = min(max(pred_d, 0), 255)
#     #     isin, posx, posy = depth_map_inference(depth, x, y, pred_d, threshold)
#     #     if isin:
#     #         print(f'Arm {ith_arm}: The first point where the depth changes is at ({posx}, {posy})')
#     #         img_annotate = cv2.circle(img_annotate, (posy, posx), 20, (0, 255, 0), -1)
#     #         np.save(f'../../data/intermediate/Integrator/{img_name_without_ext}/arm{ith_arm}_2.npy', np.array([posx, posy]))
#     #         break
#     #     else:
#     #         img_annotate = cv2.circle(img_annotate, (y, x), 20, (0, 127, 0), -1)
#     # logic same as v1
#     t = 0
#     threshold = thres
#     # set starting point as the finger plus arm vector
#     start_x = arm_vector[1][0] +0.2*(arm_vector[1][0] - arm_vector[0][0])
#     start_y = arm_vector[1][1] +0.2*(arm_vector[1][1] - arm_vector[0][1])
#     start_depth = int(depth[int(round(start_x)), int(round(start_y))])

#     while True:
#         x = start_x+ int(t * normalized_arm_dir[0])
#         y = start_y + int(t * normalized_arm_dir[1])
#         t = t + 1
#         if x < 0 or x >= depth.shape[0] or y < 0 or y >= depth.shape[1]:
#             print(f"Arm {ith_arm}: The arm is out of the image in while loop.")
#             break
#         if x == arm_vector[1][0] and y == arm_vector[1][1]:
#             continue
#         pred_d = start_depth+ int(t * depth_dir)
#         pred_d = min(max(pred_d, 0), 255)
#         isin, posx, posy = depth_map_inference(depth, int(round(x)), int(round(y)), pred_d, threshold)
#         if isin:
#             print(f'Arm {ith_arm}: The first point where the depth changes is at ({posx}, {posy})')
#             img_annotate = cv2.circle(img_annotate, (posy, posx), 20, (0, 255, 0), -1)
#             np.save(f'../../data/intermediate/Integrator/{img_name_without_ext}/arm{ith_arm}.npy', np.array([posx, posy]))
#             break
#         else:
#             img_annotate = cv2.circle(img_annotate, (y, x), 20, (0, 127, 0), -1)
#         t = t + 1
#     cv2.imwrite(f'../../data/intermediate/Integrator/{img_name_without_ext}/images/arm{ith_arm}_v2.png', img_annotate)
#     return [posy, posx] 

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--img-name', type=str, default='spiderman.jpg', help='name of the image in /data/input/')
#     parser.add_argument('--threshold', type=int, default=1, help='threshold for depth change')
#     args = parser.parse_args()

#     img_name_without_ext = os.path.splitext(args.img_name)[0]
#     img = cv2.imread(f'../../data/input/{args.img_name}')
#     depth = cv2.imread(f'../../data/intermediate/depth_map/{img_name_without_ext}_depth.png', cv2.IMREAD_GRAYSCALE)
#     depth_original = np.load(f'../../data/intermediate/depth_map/{img_name_without_ext}_depth.npy')

#     arm_vector1 = np.load(f'../../data/intermediate/arm_vectors/{img_name_without_ext}/arm1.npy').astype(int)
#     arm_vector2 = np.load(f'../../data/intermediate/arm_vectors/{img_name_without_ext}/arm2.npy').astype(int)
    
#     eye_center = np.load(f'../../data/intermediate/arm_vectors/{img_name_without_ext}/eye_center.npy').astype(int)
#     wrist_left = np.load(f'../../data/intermediate/arm_vectors/{img_name_without_ext}/wrist_left.npy').astype(int)
#     wrist_right = np.load(f'../../data/intermediate/arm_vectors/{img_name_without_ext}/wrist_right.npy').astype(int)
#     finger_left = np.load(f'../../data/intermediate/arm_vectors/{img_name_without_ext}/finger_left.npy').astype(int)
#     finger_right = np.load(f'../../data/intermediate/arm_vectors/{img_name_without_ext}/finger_right.npy').astype(int)


#     # 檢查 Pillow 版本
#     print(f"Pillow version: {Image.__version__}")

#     # 設定配置
#     cfg = get_cfg()
#     # cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
#     # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
#     # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 設定預測閾值
#     cfg.merge_from_file(model_zoo.get_config_file("Misc/panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml"))
#     cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detectron2/Misc/panoptic_fpn_R_101_dconv_cascade_gn_3x/139797668/model_final_be35db.pkl"

#     # 建立預測器
#     predictor = DefaultPredictor(cfg)

#     # 載入圖像
#     image = cv2.imread(f'../../data/input/{args.img_name}')

#     # 執行預測
#     outputs = predictor(image)

    

#     # 獲取 panoptic_seg 和 segments_info
#     panoptic_seg, segments_info = outputs["panoptic_seg"]
#     panoptic_seg = panoptic_seg.cpu()  # 將 CUDA 張量移動到 CPU
#     img_name_without_ext = os.path.splitext(args.img_name)[0]
#     os.makedirs(f'../../data/output/{img_name_without_ext}', exist_ok=True)
#     os.makedirs(f'../../data/intermediate/detectron_masks', exist_ok=True)
#     # 視覺化整體結果
#     v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
#     v = v.draw_panoptic_seg_predictions(panoptic_seg, segments_info)
#     result = v.get_image()[:, :, ::-1]

#     # 保存整體結果圖像
#     output_path = f"../../data/intermediate/detectron_masks/{img_name_without_ext}.png"
#     cv2.imwrite(output_path, result)
#     print(f"The masks saved to {output_path}")
#     msg1 = 'Point 1 not in any category'
#     msg2 = 'Point 2 not in any category'
#     msg3 = "Point 1 not in any category"
#     msg4 = "Point 2 not in any category"
#     msg5= "Point 1 not in any category"
#     msg6 = "Point 2 not in any category"
#     msg7 = "Point 1 not in any category"
#     msg8 = "Point 2 not in any category"
#     try:
#         inference_point_1 = get_inference_point(img, img_name_without_ext, 1, depth, arm_vector1, args.threshold)
#     # 檢查每個點是否在某個mask內，並輸出物品種類和概率
#         x, y = inference_point_1
#         segment_id = panoptic_seg[y, x].item()
#         segment = next((s for s in segments_info if s["id"] == segment_id), None)
#         if segment:
#             category_id = segment["category_id"]

#             score = segment.get("score", 1.0)  # 默認為1.0（對於stuff類別沒有score）
#             if score == 1.0:
#                 category_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).stuff_classes[category_id]
#             else:
#                 category_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[category_id]

#             msg1 = f"Point 1 ({inference_point_1}) is in category '{category_name}' with score {score}"

#             # 創建只顯示指定點mask和文字標籤的圖像
#             mask = panoptic_seg == segment_id
#             mask = mask.numpy()  # 將張量轉換為 NumPy 數組

#             # 創建空白圖像來顯示mask
#             masked_image = np.zeros_like(image)
#             masked_image[mask] = image[mask]

#             # 顯示點的位置，使用星號表示
#             cv2.drawMarker(masked_image, (x, y), (0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=40, thickness=2)

#             # 添加文字標籤
#             label = f"{category_name} {int(score * 100)}%"
#             font_scale = 1.5
#             font_thickness = 3
#             (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
#             cv2.rectangle(masked_image, (x, y - 55), (x + w, y), (0, 0, 0), -1)
#             cv2.putText(masked_image, label, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)



#             # 保存結果圖像
#             output_path = f"../../data/output/{img_name_without_ext}/v1_arm1.png"
#             cv2.imwrite(output_path, masked_image)
#             print(f"Result saved to {output_path}")
#     except:
#         pass  
      
#     try:
#         inference_point_2 = get_inference_point(img, img_name_without_ext, 2, depth, arm_vector2, args.threshold)
#     # 檢查每個點是否在某個mask內，並輸出物品種類和概率
#         x, y = inference_point_2
#         segment_id = panoptic_seg[y, x].item()
#         segment = next((s for s in segments_info if s["id"] == segment_id), None)
#         if segment:
#             category_id = segment["category_id"]

#             score = segment.get("score", 1.0)  # 默認為1.0（對於stuff類別沒有score）
#             if score == 1.0:
#                 category_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).stuff_classes[category_id]
#             else:
#                 category_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[category_id]
#             msg2 = f"Point 2 ({inference_point_2}) is in category '{category_name}' with score {score}"

#             # 創建只顯示指定點mask和文字標籤的圖像
#             mask = panoptic_seg == segment_id
#             mask = mask.numpy()  # 將張量轉換為 NumPy 數組

#             # 創建空白圖像來顯示mask
#             masked_image = np.zeros_like(image)
#             masked_image[mask] = image[mask]

#             # 顯示點的位置，使用星號表示
#             cv2.drawMarker(masked_image, (x, y), (0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=40, thickness=2)

#             # 添加文字標籤
#             label = f"{category_name} {int(score * 100)}%"
#             font_scale = 1.5
#             font_thickness = 3
#             (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
#             cv2.rectangle(masked_image, (x, y - 55), (x + w, y), (0, 0, 0), -1)
#             cv2.putText(masked_image, label, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)



#             # 保存結果圖像
#             output_path = f"../../data/output/{img_name_without_ext}/v1_arm2.png"
#             cv2.imwrite(output_path, masked_image)
#             print(f"Result saved to {output_path}")
#     except:
#         pass
#     try:
#         inference_point_3 = get_inference_point_v2(img, img_name_without_ext, 1, depth, eye_center, finger_left, args.threshold)
#     # 檢查每個點是否在某個mask內，並輸出物品種類和概率
#         x, y = inference_point_3
#         segment_id = panoptic_seg[y, x].item()
#         segment = next((s for s in segments_info if s["id"] == segment_id), None)
#         if segment:
#             category_id = segment["category_id"]

#             score = segment.get("score", 1.0)  # 默認為1.0（對於stuff類別沒有score）
#             if score == 1.0:
#                 category_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).stuff_classes[category_id]
#             else:
#                 category_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[category_id]
#             msg3 = f"Point 1 ({inference_point_3}) is in category '{category_name}' with score {score}"

#             # 創建只顯示指定點mask和文字標籤的圖像
#             mask = panoptic_seg == segment_id
#             mask = mask.numpy()  # 將張量轉換為 NumPy 數組

#             # 創建空白圖像來顯示mask
#             masked_image = np.zeros_like(image)
#             masked_image[mask] = image[mask]

#             # 顯示點的位置，使用星號表示
#             cv2.drawMarker(masked_image, (x, y), (0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=40, thickness=2)

#             # 添加文字標籤
#             label = f"{category_name} {int(score * 100)}%"
#             font_scale = 1.5
#             font_thickness = 3
#             (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
#             cv2.rectangle(masked_image, (x, y - 55), (x + w, y), (0, 0, 0), -1)
#             cv2.putText(masked_image, label, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)



#             # 保存結果圖像
#             output_path = f"../../data/output/{img_name_without_ext}/v2_arm1.png"
#             cv2.imwrite(output_path, masked_image)
#             print(f"Result saved to {output_path}")
#     except:
#         pass
#     try:
#         inference_point_4 = get_inference_point_v2(img, img_name_without_ext, 2, depth, eye_center, finger_right, args.threshold)
        
#     # 檢查每個點是否在某個mask內，並輸出物品種類和概率
#         x, y = inference_point_4
#         segment_id = panoptic_seg[y, x].item()
#         segment = next((s for s in segments_info if s["id"] == segment_id), None)
#         if segment:
#             category_id = segment["category_id"]

#             score = segment.get("score", 1.0)  # 默認為1.0（對於stuff類別沒有score）
#             if score == 1.0:
#                 category_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).stuff_classes[category_id]
#             else:
#                 category_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[category_id]
#             msg4 = f"Point 2 ({inference_point_4}) is in category '{category_name}' with score {score}"

#             # 創建只顯示指定點mask和文字標籤的圖像
#             mask = panoptic_seg == segment_id
#             mask = mask.numpy()  # 將張量轉換為 NumPy 數組

#             # 創建空白圖像來顯示mask
#             masked_image = np.zeros_like(image)
#             masked_image[mask] = image[mask]

#             # 顯示點的位置，使用星號表示
#             cv2.drawMarker(masked_image, (x, y), (0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=40, thickness=2)

#             # 添加文字標籤
#             label = f"{category_name} {int(score * 100)}%"
#             font_scale = 1.5
#             font_thickness = 3
#             (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
#             cv2.rectangle(masked_image, (x, y - 55), (x + w, y), (0, 0, 0), -1)
#             cv2.putText(masked_image, label, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)



#             # 保存結果圖像
#             output_path = f"../../data/output/{img_name_without_ext}/v2_arm2.png"
#             cv2.imwrite(output_path, masked_image)
#             print(f"Result saved to {output_path}")
#     except:
#         pass
#     try:
#         inference_point_5 = get_inference_point(img, img_name_without_ext, 1, depth_original, arm_vector1, args.threshold)
#     # 檢查每個點是否在某個mask內，並輸出物品種類和概率
#         x, y = inference_point_5
#         segment_id = panoptic_seg[y, x].item()
#         segment = next((s for s in segments_info if s["id"] == segment_id), None)
#         print("hello")
#         if segment:
#             category_id = segment["category_id"]

#             score = segment.get("score", 1.0)  # 默認為1.0（對於stuff類別沒有score）
#             if score == 1.0:
#                 category_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).stuff_classes[category_id]
#             else:
#                 category_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[category_id]
#             msg5 = f"Point 1 ({inference_point_5}) is in category '{category_name}' with score {score}"

#             # 創建只顯示指定點mask和文字標籤的圖像
#             mask = panoptic_seg == segment_id
#             mask = mask.numpy()  # 將張量轉換為 NumPy 數組

#             # 創建空白圖像來顯示mask
#             masked_image = np.zeros_like(image)
#             masked_image[mask] = image[mask]

#             # 顯示點的位置，使用星號表示
#             cv2.drawMarker(masked_image, (x, y), (0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=40, thickness=2)

#             # 添加文字標籤
#             label = f"{category_name} {int(score * 100)}%"
#             font_scale = 1.5
#             font_thickness = 3
#             (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
#             cv2.rectangle(masked_image, (x, y - 55), (x + w, y), (0, 0, 0), -1)
#             cv2.putText(masked_image, label, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)



#             # 保存結果圖像
#             output_path = f"../../data/output/{img_name_without_ext}/original_v1_arm1.png"
#             cv2.imwrite(output_path, masked_image)
#             print(f"Result saved to {output_path}")
#     except:
#         pass
#     try:
#         inference_point_6 = get_inference_point(img, img_name_without_ext, 2, depth_original, args.threshold)
#     # 檢查每個點是否在某個mask內，並輸出物品種類和概率
#         x, y = inference_point_6
#         segment_id = panoptic_seg[y, x].item()
#         segment = next((s for s in segments_info if s["id"] == segment_id), None)
#         if segment:
#             category_id = segment["category_id"]

#             score = segment.get("score", 1.0)  # 默認為1.0（對於stuff類別沒有score）
#             if score == 1.0:
#                 category_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).stuff_classes[category_id]
#             else:
#                 category_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[category_id]
#             msg6 = f"Point 2 ({inference_point_6}) is in category '{category_name}' with score {score}"

#             # 創建只顯示指定點mask和文字標籤的圖像
#             mask = panoptic_seg == segment_id
#             mask = mask.numpy()  # 將張量轉換為 NumPy 數組

#             # 創建空白圖像來顯示mask
#             masked_image = np.zeros_like(image)
#             masked_image[mask] = image[mask]

#             # 顯示點的位置，使用星號表示
#             cv2.drawMarker(masked_image, (x, y), (0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=40, thickness=2)

#             # 添加文字標籤
#             label = f"{category_name} {int(score * 100)}%"
#             font_scale = 1.5
#             font_thickness = 3
#             (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
#             cv2.rectangle(masked_image, (x, y - 55), (x + w, y), (0, 0, 0), -1)
#             cv2.putText(masked_image, label, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)



#             # 保存結果圖像
#             output_path = f"../../data/output/{img_name_without_ext}/original_v1_arm2.png"
#             cv2.imwrite(output_path, masked_image)
#             print(f"Result saved to {output_path}")
#     except:
#         pass
#     try:
#         inference_point_7 = get_inference_point_v2(img, img_name_without_ext, 1, depth_original, eye_center, finger_left, args.threshold)
#     # 檢查每個點是否在某個mask內，並輸出物品種類和概率
#         x, y = inference_point_7
#         segment_id = panoptic_seg[y, x].item()
#         segment = next((s for s in segments_info if s["id"] == segment_id), None)
#         if segment:
#             category_id = segment["category_id"]

#             score = segment.get("score", 1.0)  # 默認為1.0（對於stuff類別沒有score）
#             if score == 1.0:
#                 category_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).stuff_classes[category_id]
#             else:
#                 category_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[category_id]
#             msg7 = f"Point 1 ({inference_point_7}) is in category '{category_name}' with score {score}"

#             # 創建只顯示指定點mask和文字標籤的圖像
#             mask = panoptic_seg == segment_id
#             mask = mask.numpy()  # 將張量轉換為 NumPy 數組

#             # 創建空白圖像來顯示mask
#             masked_image = np.zeros_like(image)
#             masked_image[mask] = image[mask]

#             # 顯示點的位置，使用星號表示
#             cv2.drawMarker(masked_image, (x, y), (0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=40, thickness=2)

#             # 添加文字標籤
#             label = f"{category_name} {int(score * 100)}%"
#             font_scale = 1.5
#             font_thickness = 3
#             (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
#             cv2.rectangle(masked_image, (x, y - 55), (x + w, y), (0, 0, 0), -1)
#             cv2.putText(masked_image, label, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)



#             # 保存結果圖像
#             output_path = f"../../data/output/{img_name_without_ext}/original_v2_arm1.png"
#             cv2.imwrite(output_path, masked_image)
#             print(f"Result saved to {output_path}")
#     except:
#         pass
#     try:
#         inference_point_8 = get_inference_point_v2(img, img_name_without_ext, 2, depth_original, eye_center,finger_right, args.threshold)
#     # 檢查每個點是否在某個mask內，並輸出物品種類和概率
#         x, y = inference_point_8
#         segment_id = panoptic_seg[y, x].item()
#         segment = next((s for s in segments_info if s["id"] == segment_id), None)
#         if segment:
#             category_id = segment["category_id"]

#             score = segment.get("score", 1.0)  # 默認為1.0（對於stuff類別沒有score）
#             if score == 1.0:
#                 category_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).stuff_classes[category_id]
#             else:
#                 category_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[category_id]
#             msg8 = f"Point 2 ({inference_point_8}) is in category '{category_name}' with score {score}"

#             # 創建只顯示指定點mask和文字標籤的圖像
#             mask = panoptic_seg == segment_id
#             mask = mask.numpy()  # 將張量轉換為 NumPy 數組

#             # 創建空白圖像來顯示mask
#             masked_image = np.zeros_like(image)
#             masked_image[mask] = image[mask]

#             # 顯示點的位置，使用星號表示
#             cv2.drawMarker(masked_image, (x, y), (0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=40, thickness=2)

#             # 添加文字標籤
#             label = f"{category_name} {int(score * 100)}%"
#             font_scale = 1.5
#             font_thickness = 3
#             (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
#             cv2.rectangle(masked_image, (x, y - 55), (x + w, y), (0, 0, 0), -1)
#             cv2.putText(masked_image, label, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)



#             # 保存結果圖像
#             output_path = f"../../data/output/{img_name_without_ext}/original_v2_arm2.png"
#             cv2.imwrite(output_path, masked_image)
#             print(f"Result saved to {output_path}")
#     except:
#         pass


#     print("v1 Arm 1:",end=" ")
#     print(msg1)
#     print("v1 Arm 2:",end=" ")
#     print(msg2)
#     print("v1 original Arm 1:",end=" ")
#     print(msg5)
#     print("v1 original Arm 2:",end=" ")
#     print(msg6)
#     print("v2 Arm 1:",end=" ")
#     print(msg3)
#     print("v2 Arm 2:",end=" ")
#     print(msg4)
#     print("v2 original Arm 1:",end=" ")
#     print(msg7)
#     print("v2 original Arm 2:",end=" ")
#     print(msg8)
#     # save to a txt
#     with open(f'../../data/output/{img_name_without_ext}/output.txt', 'w') as f:
#         f.write(f"v1 Arm 1: {msg1}\n")
#         f.write(f"v1 Arm 2: {msg2}\n")
#         f.write(f"v1 original Arm 1: {msg5}\n")
#         f.write(f"v1 original Arm 2: {msg6}\n")
#         f.write(f"v2 Arm 1: {msg3}\n")
#         f.write(f"v2 Arm 2: {msg4}\n")
#         f.write(f"v2 original Arm 1: {msg7}\n")
#         f.write(f"v2 original Arm 2: {msg8}\n")
import argparse
import cv2
import numpy as np
import os
from ultralytics import YOLOv10
import supervision as sv
import itertools
import cv2
import torch
import numpy as np
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from PIL import Image
# yan lee
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
            print(f"Arm {ith_arm}: The arm is out of the image in first while loop.")
            break
        if x == arm_vector[1][0] and y == arm_vector[1][1]:
            continue
        d = int(depth[x, y])
        pred_d = depth_wrist + int(t * depth_dir)
        if abs(d - pred_d) > threshold:
            break
    t *= 2
    # threshold *= 2
    while True:
        x = arm_vector[1][0] + int(t * normalized_arm_dir[0])
        y = arm_vector[1][1] + int(t * normalized_arm_dir[1])
        t = t + 1
        if x < 0 or x >= depth.shape[0] or y < 0 or y >= depth.shape[1]:
            print(f"Arm {ith_arm}: The arm is out of the image in second while loop.")
            break
        if x == arm_vector[1][0] and y == arm_vector[1][1]:
            continue
        pred_d = depth_wrist + int(t * depth_dir)
        pred_d = min(max(pred_d, 0), 255)
        isin, posx, posy = depth_map_inference(depth, x, y, pred_d, threshold)
        if isin:
            print(f'Arm {ith_arm}: The first point where the depth changes is at ({posx}, {posy})')
            img_annotate = cv2.circle(img_annotate, (posy, posx), 20, (0, 255, 0), -1)
            np.save(f'../../data/intermediate/Integrator/{img_name_without_ext}/arm{ith_arm}.npy', np.array([posx, posy]))
            break
        else:
            img_annotate = cv2.circle(img_annotate, (y, x), 20, (0, 127, 0), -1)
            
    cv2.imwrite(f'../../data/intermediate/Integrator/{img_name_without_ext}/images/arm{ith_arm}.png', img_annotate)
    return [posy, posx] 

def get_inference_point_v2(img, img_name_without_ext, ith_arm, depth, eye_center, finger_pos, thres=5):
    arm_vector = np.array([eye_center, finger_pos])
    arm_dir = np.array([arm_vector[1][0] - arm_vector[0][0], arm_vector[1][1] - arm_vector[0][1]])

    depth_eye = int(depth[eye_center[0], eye_center[1]])
    depth_finger = int(depth[finger_pos[0], finger_pos[1]])
    depth_dir = depth_finger - depth_eye

    # 射线方向的单位向量
    depth_dir = depth_dir / np.linalg.norm(arm_dir)
    normalized_arm_dir = arm_dir / np.linalg.norm(arm_dir)

    os.makedirs(f'../../data/intermediate/Integrator/{img_name_without_ext}', exist_ok=True)
    os.makedirs(f'../../data/intermediate/Integrator/{img_name_without_ext}/images', exist_ok=True)
    img_annotate = img.copy()
    t = 1
    threshold = thres
    img_annotate = img.copy()

    while True:
        x = arm_vector[1][0] + int(t * normalized_arm_dir[0])
        y = arm_vector[1][1] + int(t * normalized_arm_dir[1])
        t = t + 1
        if x < 0 or x >= depth.shape[0] or y < 0 or y >= depth.shape[1]:
            print(f"Arm{ith_arm}: The arm is out of the image in first while loop.")
            break
        if x == arm_vector[1][0] and y == arm_vector[1][1]:
            continue
        d = int(depth[x, y])
        pred_d = depth_finger + int(t * depth_dir)
        if abs(d - pred_d) > threshold:
            break
    t *= 2
    threshold *=3
    while True:
        x = arm_vector[1][0] + int(t * normalized_arm_dir[0])
        y = arm_vector[1][1] + int(t * normalized_arm_dir[1])
        t = t + 1
        if x < 0 or x >= depth.shape[0] or y < 0 or y >= depth.shape[1]:
            print(f"Arm{ith_arm}: The arm is out of the image in second while loop.")
            break
        if x == arm_vector[1][0] and y == arm_vector[1][1]:
            continue
        pred_d = depth_finger + int(t * depth_dir)
        pred_d = min(max(pred_d, 0), 255)
        isin, posx, posy = depth_map_inference(depth, x, y, pred_d, threshold)
        if isin:
            print(f'Arm {ith_arm}: The first point where the depth changes is at ({posx}, {posy})')
            img_annotate = cv2.circle(img_annotate, (posy, posx), 20, (0, 255, 0), -1)
            np.save(f'../../data/intermediate/Integrator/{img_name_without_ext}/arm{ith_arm}_2.npy', np.array([posx, posy]))
            break
        else:
            img_annotate = cv2.circle(img_annotate, (y, x), 20, (0, 127, 0), -1)
    cv2.imwrite(f'../../data/intermediate/Integrator/{img_name_without_ext}/images/arm{ith_arm}_v2.png', img_annotate)
    return [posy, posx] 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-name', type=str, default='spiderman.jpg', help='name of the image in /data/input/')
    parser.add_argument('--threshold', type=int, default=7, help='threshold for depth change')
    args = parser.parse_args()

    img_name_without_ext = os.path.splitext(args.img_name)[0]
    img = cv2.imread(f'../../data/input/{args.img_name}')
    depth = cv2.imread(f'../../data/intermediate/depth_map/{img_name_without_ext}_depth.png', cv2.IMREAD_GRAYSCALE)
    depth_original = np.load(f'../../data/intermediate/depth_map/{img_name_without_ext}_depth.npy')

    arm_vector1 = np.load(f'../../data/intermediate/arm_vectors/{img_name_without_ext}/arm1.npy').astype(int)
    arm_vector2 = np.load(f'../../data/intermediate/arm_vectors/{img_name_without_ext}/arm2.npy').astype(int)
    
    eye_center = np.load(f'../../data/intermediate/arm_vectors/{img_name_without_ext}/eye_center.npy').astype(int)
    wrist_left = np.load(f'../../data/intermediate/arm_vectors/{img_name_without_ext}/wrist_left.npy').astype(int)
    wrist_right = np.load(f'../../data/intermediate/arm_vectors/{img_name_without_ext}/wrist_right.npy').astype(int)
    finger_left = np.load(f'../../data/intermediate/arm_vectors/{img_name_without_ext}/finger_left.npy').astype(int)
    finger_right = np.load(f'../../data/intermediate/arm_vectors/{img_name_without_ext}/finger_right.npy').astype(int)


    # 檢查 Pillow 版本
    print(f"Pillow version: {Image.__version__}")

    # 設定配置
    cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 設定預測閾值
    cfg.merge_from_file(model_zoo.get_config_file("Misc/panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml"))
    cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detectron2/Misc/panoptic_fpn_R_101_dconv_cascade_gn_3x/139797668/model_final_be35db.pkl"

    # 建立預測器
    predictor = DefaultPredictor(cfg)

    # 載入圖像
    image = cv2.imread(f'../../data/input/{args.img_name}')

    # 執行預測
    outputs = predictor(image)

    

    # 獲取 panoptic_seg 和 segments_info
    panoptic_seg, segments_info = outputs["panoptic_seg"]
    panoptic_seg = panoptic_seg.cpu()  # 將 CUDA 張量移動到 CPU
    img_name_without_ext = os.path.splitext(args.img_name)[0]
    os.makedirs(f'../../data/output/{img_name_without_ext}', exist_ok=True)
    os.makedirs(f'../../data/intermediate/detectron_masks', exist_ok=True)
    # 視覺化整體結果
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_panoptic_seg_predictions(panoptic_seg, segments_info)
    result = v.get_image()[:, :, ::-1]

    # 保存整體結果圖像
    output_path = f"../../data/intermediate/detectron_masks/{img_name_without_ext}.png"
    cv2.imwrite(output_path, result)
    print(f"The masks saved to {output_path}")
    msg1 = 'Point 1 not in any category'
    msg2 = 'Point 2 not in any category'
    msg3 = "Point 1 not in any category"
    msg4 = "Point 2 not in any category"
    msg5= "Point 1 not in any category"
    msg6 = "Point 2 not in any category"
    msg7 = "Point 1 not in any category"
    msg8 = "Point 2 not in any category"
    try:
        inference_point_1 = get_inference_point(img, img_name_without_ext, 1, depth, arm_vector1, args.threshold)
    # 檢查每個點是否在某個mask內，並輸出物品種類和概率
        x, y = inference_point_1
        segment_id = panoptic_seg[y, x].item()
        segment = next((s for s in segments_info if s["id"] == segment_id), None)
        if segment:
            category_id = segment["category_id"]

            score = segment.get("score", 1.0)  # 默認為1.0（對於stuff類別沒有score）
            if score == 1.0:
                category_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).stuff_classes[category_id]
            else:
                category_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[category_id]

            msg1 = f"Point 1 ({inference_point_1}) is in category '{category_name}' with score {score}"

            # 創建只顯示指定點mask和文字標籤的圖像
            mask = panoptic_seg == segment_id
            mask = mask.numpy()  # 將張量轉換為 NumPy 數組

            # 創建空白圖像來顯示mask
            masked_image = np.zeros_like(image)
            masked_image[mask] = image[mask]

            # 顯示點的位置，使用星號表示
            cv2.drawMarker(masked_image, (x, y), (0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=40, thickness=2)

            # 添加文字標籤
            label = f"{category_name} {int(score * 100)}%"
            font_scale = 1.5
            font_thickness = 3
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            cv2.rectangle(masked_image, (x, y - 55), (x + w, y), (0, 0, 0), -1)
            cv2.putText(masked_image, label, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)



            # 保存結果圖像
            output_path = f"../../data/output/{img_name_without_ext}/v2_arm1.png"
            cv2.imwrite(output_path, masked_image)
            print(f"Result saved to {output_path}")
    except:
        pass  
      
    try:
        inference_point_2 = get_inference_point(img, img_name_without_ext, 2, depth, arm_vector2, args.threshold)
    # 檢查每個點是否在某個mask內，並輸出物品種類和概率
        x, y = inference_point_2
        segment_id = panoptic_seg[y, x].item()
        segment = next((s for s in segments_info if s["id"] == segment_id), None)
        if segment:
            category_id = segment["category_id"]

            score = segment.get("score", 1.0)  # 默認為1.0（對於stuff類別沒有score）
            if score == 1.0:
                category_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).stuff_classes[category_id]
            else:
                category_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[category_id]
            msg2 = f"Point 2 ({inference_point_2}) is in category '{category_name}' with score {score}"

            # 創建只顯示指定點mask和文字標籤的圖像
            mask = panoptic_seg == segment_id
            mask = mask.numpy()  # 將張量轉換為 NumPy 數組

            # 創建空白圖像來顯示mask
            masked_image = np.zeros_like(image)
            masked_image[mask] = image[mask]

            # 顯示點的位置，使用星號表示
            cv2.drawMarker(masked_image, (x, y), (0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=40, thickness=2)

            # 添加文字標籤
            label = f"{category_name} {int(score * 100)}%"
            font_scale = 1.5
            font_thickness = 3
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            cv2.rectangle(masked_image, (x, y - 55), (x + w, y), (0, 0, 0), -1)
            cv2.putText(masked_image, label, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)



            # 保存結果圖像
            output_path = f"../../data/output/{img_name_without_ext}/v2_arm2.png"
            cv2.imwrite(output_path, masked_image)
            print(f"Result saved to {output_path}")
    except:
        pass
    try:
        inference_point_3 = get_inference_point_v2(img, img_name_without_ext, 1, depth, eye_center, finger_left, args.threshold)
    # 檢查每個點是否在某個mask內，並輸出物品種類和概率
        x, y = inference_point_3
        segment_id = panoptic_seg[y, x].item()
        segment = next((s for s in segments_info if s["id"] == segment_id), None)
        if segment:
            category_id = segment["category_id"]

            score = segment.get("score", 1.0)  # 默認為1.0（對於stuff類別沒有score）
            if score == 1.0:
                category_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).stuff_classes[category_id]
            else:
                category_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[category_id]
            msg3 = f"Point 1 ({inference_point_3}) is in category '{category_name}' with score {score}"

            # 創建只顯示指定點mask和文字標籤的圖像
            mask = panoptic_seg == segment_id
            mask = mask.numpy()  # 將張量轉換為 NumPy 數組

            # 創建空白圖像來顯示mask
            masked_image = np.zeros_like(image)
            masked_image[mask] = image[mask]

            # 顯示點的位置，使用星號表示
            cv2.drawMarker(masked_image, (x, y), (0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=40, thickness=2)

            # 添加文字標籤
            label = f"{category_name} {int(score * 100)}%"
            font_scale = 1.5
            font_thickness = 3
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            cv2.rectangle(masked_image, (x, y - 55), (x + w, y), (0, 0, 0), -1)
            cv2.putText(masked_image, label, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)



            # 保存結果圖像
            output_path = f"../../data/output/{img_name_without_ext}/v3_arm1.png"
            cv2.imwrite(output_path, masked_image)
            print(f"Result saved to {output_path}")
    except:
        pass
    try:
        inference_point_4 = get_inference_point_v2(img, img_name_without_ext, 2, depth, eye_center, finger_right, args.threshold)
        
    # 檢查每個點是否在某個mask內，並輸出物品種類和概率
        x, y = inference_point_4
        segment_id = panoptic_seg[y, x].item()
        segment = next((s for s in segments_info if s["id"] == segment_id), None)
        if segment:
            category_id = segment["category_id"]

            score = segment.get("score", 1.0)  # 默認為1.0（對於stuff類別沒有score）
            if score == 1.0:
                category_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).stuff_classes[category_id]
            else:
                category_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[category_id]
            msg4 = f"Point 2 ({inference_point_4}) is in category '{category_name}' with score {score}"

            # 創建只顯示指定點mask和文字標籤的圖像
            mask = panoptic_seg == segment_id
            mask = mask.numpy()  # 將張量轉換為 NumPy 數組

            # 創建空白圖像來顯示mask
            masked_image = np.zeros_like(image)
            masked_image[mask] = image[mask]

            # 顯示點的位置，使用星號表示
            cv2.drawMarker(masked_image, (x, y), (0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=40, thickness=2)

            # 添加文字標籤
            label = f"{category_name} {int(score * 100)}%"
            font_scale = 1.5
            font_thickness = 3
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            cv2.rectangle(masked_image, (x, y - 55), (x + w, y), (0, 0, 0), -1)
            cv2.putText(masked_image, label, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)



            # 保存結果圖像
            output_path = f"../../data/output/{img_name_without_ext}/v3_arm2.png"
            cv2.imwrite(output_path, masked_image)
            print(f"Result saved to {output_path}")
    except:
        pass
    try:
        inference_point_5 = get_inference_point(img, img_name_without_ext, 1, depth_original, arm_vector1, args.threshold)
    # 檢查每個點是否在某個mask內，並輸出物品種類和概率
        x, y = inference_point_5
        segment_id = panoptic_seg[y, x].item()
        segment = next((s for s in segments_info if s["id"] == segment_id), None)
        print("hello")
        if segment:
            category_id = segment["category_id"]

            score = segment.get("score", 1.0)  # 默認為1.0（對於stuff類別沒有score）
            if score == 1.0:
                category_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).stuff_classes[category_id]
            else:
                category_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[category_id]
            msg5 = f"Point 1 ({inference_point_5}) is in category '{category_name}' with score {score}"

            # 創建只顯示指定點mask和文字標籤的圖像
            mask = panoptic_seg == segment_id
            mask = mask.numpy()  # 將張量轉換為 NumPy 數組

            # 創建空白圖像來顯示mask
            masked_image = np.zeros_like(image)
            masked_image[mask] = image[mask]

            # 顯示點的位置，使用星號表示
            cv2.drawMarker(masked_image, (x, y), (0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=40, thickness=2)

            # 添加文字標籤
            label = f"{category_name} {int(score * 100)}%"
            font_scale = 1.5
            font_thickness = 3
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            cv2.rectangle(masked_image, (x, y - 55), (x + w, y), (0, 0, 0), -1)
            cv2.putText(masked_image, label, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)



            # 保存結果圖像
            output_path = f"../../data/output/{img_name_without_ext}/original_v2_arm1.png"
            cv2.imwrite(output_path, masked_image)
            print(f"Result saved to {output_path}")
    except:
        pass
    try:
        inference_point_6 = get_inference_point(img, img_name_without_ext, 2, depth_original, arm_vector2, args.threshold)
    # 檢查每個點是否在某個mask內，並輸出物品種類和概率
        x, y = inference_point_6
        segment_id = panoptic_seg[y, x].item()
        segment = next((s for s in segments_info if s["id"] == segment_id), None)
        if segment:
            category_id = segment["category_id"]

            score = segment.get("score", 1.0)  # 默認為1.0（對於stuff類別沒有score）
            if score == 1.0:
                category_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).stuff_classes[category_id]
            else:
                category_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[category_id]
            msg6 = f"Point 2 ({inference_point_6}) is in category '{category_name}' with score {score}"

            # 創建只顯示指定點mask和文字標籤的圖像
            mask = panoptic_seg == segment_id
            mask = mask.numpy()  # 將張量轉換為 NumPy 數組

            # 創建空白圖像來顯示mask
            masked_image = np.zeros_like(image)
            masked_image[mask] = image[mask]

            # 顯示點的位置，使用星號表示
            cv2.drawMarker(masked_image, (x, y), (0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=40, thickness=2)

            # 添加文字標籤
            label = f"{category_name} {int(score * 100)}%"
            font_scale = 1.5
            font_thickness = 3
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            cv2.rectangle(masked_image, (x, y - 55), (x + w, y), (0, 0, 0), -1)
            cv2.putText(masked_image, label, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)



            # 保存結果圖像
            output_path = f"../../data/output/{img_name_without_ext}/original_v2_arm2.png"
            cv2.imwrite(output_path, masked_image)
            print(f"Result saved to {output_path}")
    except:
        pass
    try:
        inference_point_7 = get_inference_point_v2(img, img_name_without_ext, 1, depth_original, eye_center, finger_left, args.threshold)
    # 檢查每個點是否在某個mask內，並輸出物品種類和概率
        x, y = inference_point_7
        segment_id = panoptic_seg[y, x].item()
        segment = next((s for s in segments_info if s["id"] == segment_id), None)
        if segment:
            category_id = segment["category_id"]

            score = segment.get("score", 1.0)  # 默認為1.0（對於stuff類別沒有score）
            if score == 1.0:
                category_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).stuff_classes[category_id]
            else:
                category_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[category_id]
            msg7 = f"Point 1 ({inference_point_7}) is in category '{category_name}' with score {score}"

            # 創建只顯示指定點mask和文字標籤的圖像
            mask = panoptic_seg == segment_id
            mask = mask.numpy()  # 將張量轉換為 NumPy 數組

            # 創建空白圖像來顯示mask
            masked_image = np.zeros_like(image)
            masked_image[mask] = image[mask]

            # 顯示點的位置，使用星號表示
            cv2.drawMarker(masked_image, (x, y), (0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=40, thickness=2)

            # 添加文字標籤
            label = f"{category_name} {int(score * 100)}%"
            font_scale = 1.5
            font_thickness = 3
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            cv2.rectangle(masked_image, (x, y - 55), (x + w, y), (0, 0, 0), -1)
            cv2.putText(masked_image, label, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)



            # 保存結果圖像
            output_path = f"../../data/output/{img_name_without_ext}/original_v3_arm1.png"
            cv2.imwrite(output_path, masked_image)
            print(f"Result saved to {output_path}")
    except:
        pass
    try:
        inference_point_8 = get_inference_point_v2(img, img_name_without_ext, 2, depth_original, eye_center, finger_right, args.threshold)
    # 檢查每個點是否在某個mask內，並輸出物品種類和概率
        x, y = inference_point_8
        segment_id = panoptic_seg[y, x].item()
        segment = next((s for s in segments_info if s["id"] == segment_id), None)
        if segment:
            category_id = segment["category_id"]

            score = segment.get("score", 1.0)  # 默認為1.0（對於stuff類別沒有score）
            if score == 1.0:
                category_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).stuff_classes[category_id]
            else:
                category_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[category_id]
            msg8 = f"Point 2 ({inference_point_8}) is in category '{category_name}' with score {score}"

            # 創建只顯示指定點mask和文字標籤的圖像
            mask = panoptic_seg == segment_id
            mask = mask.numpy()  # 將張量轉換為 NumPy 數組

            # 創建空白圖像來顯示mask
            masked_image = np.zeros_like(image)
            masked_image[mask] = image[mask]

            # 顯示點的位置，使用星號表示
            cv2.drawMarker(masked_image, (x, y), (0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=40, thickness=2)

            # 添加文字標籤
            label = f"{category_name} {int(score * 100)}%"
            font_scale = 1.5
            font_thickness = 3
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            cv2.rectangle(masked_image, (x, y - 55), (x + w, y), (0, 0, 0), -1)
            cv2.putText(masked_image, label, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)



            # 保存結果圖像
            output_path = f"../../data/output/{img_name_without_ext}/original_v3_arm2.png"
            cv2.imwrite(output_path, masked_image)
            print(f"Result saved to {output_path}")
    except:
        pass


    print("v1 Arm 1:",end=" ")
    print(msg1)
    print("v1 Arm 2:",end=" ")
    print(msg2)
    print("v1 original Arm 1:",end=" ")
    print(msg5)
    print("v1 original Arm 2:",end=" ")
    print(msg6)
    print("v2 Arm 1:",end=" ")
    print(msg3)
    print("v2 Arm 2:",end=" ")
    print(msg4)
    print("v2 original Arm 1:",end=" ")
    print(msg7)
    print("v2 original Arm 2:",end=" ")
    print(msg8)
    # save to a txt
    with open(f'../../data/output/{img_name_without_ext}/output.txt', 'w') as f:
        f.write(f"v1 Arm 1: {msg1}\n")
        f.write(f"v1 Arm 2: {msg2}\n")
        f.write(f"v1 original Arm 1: {msg5}\n")
        f.write(f"v1 original Arm 2: {msg6}\n")
        f.write(f"v2 Arm 1: {msg3}\n")
        f.write(f"v2 Arm 2: {msg4}\n")
        f.write(f"v2 original Arm 1: {msg7}\n")
        f.write(f"v2 original Arm 2: {msg8}\n")