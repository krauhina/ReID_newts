import os
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.ndimage import map_coordinates, gaussian_filter1d
from skimage.graph import route_through_array

seg_model = YOLO("bot/models/best_seg.pt")
keypoint_model = YOLO("bot/models/best_keypoints.pt")


async def extract_smooth_centerline(mask, sigma=3):
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    dist_bin = (dist > dist.max() * 0.9).astype(np.uint8)
    contours, _ = cv2.findContours(dist_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        raise ValueError("Центральная линия не найдена")

    centerline = max(contours, key=len).squeeze()
    dxy = np.ptp(centerline, axis=0)

    if dxy[0] > dxy[1]:
        centerline = centerline[np.argsort(centerline[:, 0])]
    else:
        centerline = centerline[np.argsort(centerline[:, 1])]

    centerline = centerline.astype(np.float32)
    centerline[:, 0] = gaussian_filter1d(centerline[:, 0], sigma=sigma)
    centerline[:, 1] = gaussian_filter1d(centerline[:, 1], sigma=sigma)
    return centerline


async def unwrap_belly_trimmed_ends(image, mask, pt1, pt2, save_path):
    centerline = await extract_smooth_centerline(mask)

    if len(centerline) < 2:
        raise ValueError("Центральная линия слишком короткая")

    dist_map = np.zeros_like(mask, dtype=np.uint8)
    for x, y in centerline.astype(int):
        dist_map[y, x] = 1

    start = (int(pt1[1]), int(pt1[0]))
    end = (int(pt2[1]), int(pt2[0]))

    path, _ = route_through_array(1 - dist_map, start, end, fully_connected=True)
    centerline = np.array([[x, y] for y, x in path], dtype=np.float32)

    x, y = centerline[:, 0], centerline[:, 1]
    x_interp = np.interp(np.linspace(0, len(x) - 1, 250), np.arange(len(x)), x)
    y_interp = np.interp(np.linspace(0, len(y) - 1, 250), np.arange(len(y)), y)

    x_smooth = gaussian_filter1d(x_interp, sigma=5)
    y_smooth = gaussian_filter1d(y_interp, sigma=5)
    centerline_smooth = np.column_stack((x_smooth, y_smooth))

    dx = gaussian_filter1d(np.gradient(centerline_smooth[:, 0]), sigma=3)
    dy = gaussian_filter1d(np.gradient(centerline_smooth[:, 1]), sigma=3)
    lengths = np.hypot(dx, dy)
    normals = np.column_stack((-dy / lengths, dx / lengths))

    h_img, w_img = image.shape[:2]
    lines = []
    max_strip_width = 0

    valid_indices = []
    for i in range(len(centerline_smooth)):
        cx, cy = map(int, centerline_smooth[i])
        if 0 <= cx < w_img and 0 <= cy < h_img and mask[cy, cx] != 0:
            valid_indices.append(i)

    if valid_indices:
        centerline_smooth = centerline_smooth[valid_indices[0]:valid_indices[-1] + 1]
        normals = normals[valid_indices[0]:valid_indices[-1] + 1]

    for i in range(len(centerline_smooth)):
        cx, cy = centerline_smooth[i]
        nx, ny = normals[i]

        length_neg, length_pos = 0, 0
        for step in range(1, max(h_img, w_img)):
            px, py = int(cx - nx * step), int(cy - ny * step)
            if not (0 <= px < w_img and 0 <= py < h_img) or mask[py, px] == 0:
                break
            length_neg += 1

        for step in range(1, max(h_img, w_img)):
            px, py = int(cx + nx * step), int(cy + ny * step)
            if not (0 <= px < w_img and 0 <= py < h_img) or mask[py, px] == 0:
                break
            length_pos += 1

        strip_width = length_neg + length_pos
        max_strip_width = max(max_strip_width, strip_width)

        line = []
        for j in range(-length_neg, length_pos):
            px, py = cx + nx * j, cy + ny * j
            if 0 <= int(py) < h_img and 0 <= int(px) < w_img:
                coords_sample = np.array([[py], [px]])
                pixel = np.stack([
                    map_coordinates(image[:, :, c], coords_sample, order=1, mode='reflect')[0]
                    for c in range(3)
                ], axis=-1)
                line.append(pixel)
        lines.append(np.array(line, dtype=np.uint8))

    unwrapped = np.zeros((len(centerline_smooth), max_strip_width, 3), dtype=np.uint8)
    for i, line in enumerate(lines):
        if line.shape[0] > 1:
            resized_line = cv2.resize(line[None, :, :], (max_strip_width, 1), cv2.INTER_LINEAR)
            unwrapped[i] = resized_line[0]

    final = cv2.resize(unwrapped, (224, 224), cv2.INTER_LINEAR)
    cv2.imwrite(save_path, final, [cv2.IMWRITE_JPEG_QUALITY, 95])


async def get_segmentation_mask(image, img_path):
    h_img, w_img = image.shape[:2]
    seg_results = seg_model(img_path)
    masks = seg_results[0].masks
    if masks is None:
        return None
    mask_tensor = masks.data[0].cpu().numpy()
    mask_resized = cv2.resize(mask_tensor, (w_img, h_img), cv2.INTER_LINEAR)
    return (mask_resized * 255).astype(np.uint8)


async def get_keypoints_from_model(img_path):
    results = keypoint_model(img_path)
    for result in results:
        if result.keypoints is not None and len(result.keypoints) > 0:
            keypoints_raw = result.keypoints.data.cpu().numpy()[0]
            keypoints = [kp for kp in keypoints_raw if kp[2] > 0.3]
            if len(keypoints) < 2:
                keypoints = [kp for kp in keypoints_raw if kp[2] > 0.05]
            if len(keypoints) >= 2:
                return tuple(map(int, keypoints[0][:2])), tuple(map(int, keypoints[1][:2]))
    return None, None


async def process_single_image(img_path, output_dir):
    try:
        if not img_path.lower().endswith((".jpg", ".jpeg", ".png")):
            return False

        image = cv2.imread(img_path)
        if image is None:
            return False

        mask = await get_segmentation_mask(image, img_path)
        if mask is None:
            return False

        pt1, pt2 = await get_keypoints_from_model(img_path)
        if None in (pt1, pt2):
            return False

        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "image_cropped.jpg")

        await unwrap_belly_trimmed_ends(image, mask, pt1, pt2, save_path)
        return True

    except Exception as e:
        print(f"Ошибка при обработке {img_path}: {str(e)}")
        return False