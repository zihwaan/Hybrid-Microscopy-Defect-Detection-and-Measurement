import cv2
import numpy as np
import math
from scipy.signal import find_peaks
from inference_sdk import InferenceHTTPClient
# ⭐️ Colab 전용 이미지 출력 함수
from google.colab.patches import cv2_imshow

# ==========================================
# 1. 설정 및 준비
# ==========================================
IMAGE_FILE = "image.jpg"  # 분석할 이미지 파일명
API_KEY = "toNQ4VGdFNNhvflr3kYT" # Roboflow API Key
MODEL_ID = "coxem-91ags/5"

# ==========================================
# 2. 유틸리티 함수 (텍스트 그리기)
# ==========================================
def draw_text_centered(img, text, x, y, color=(255, 0, 0), font_scale=1.5, thickness=3):
    """
    텍스트를 지정된 (x, y) 좌표 주변에 겹치지 않게 그리기 위한 함수입니다.
    가독성을 위해 흰색 테두리를 먼저 그리고 그 위에 색깔 텍스트를 올립니다.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # 텍스트 중심점 보정 (좌표가 텍스트의 중앙 하단이 되도록)
    draw_x = int(x - text_w / 2)
    draw_y = int(y)

    # 1. 검은색/흰색 테두리 (Outline) - 가독성 확보
    cv2.putText(img, text, (draw_x, draw_y), font, font_scale, (255, 255, 255), thickness + 5)
    # 2. 메인 텍스트
    cv2.putText(img, text, (draw_x, draw_y), font, font_scale, color, thickness)

# ==========================================
# 3. 메인 로직 함수들
# ==========================================

def run_roboflow_inference(image_path):
    """Roboflow API를 통해 추론을 수행합니다."""
    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=API_KEY
    )
    result = client.infer(image_path, model_id=MODEL_ID)
    return result

def draw_inference_results(image_path, predictions):
    """AI가 무언가를 검출했을 때(불량/이물질) 실행"""
    img = cv2.imread(image_path)
    if img is None: return

    print(f"\n[알림] {len(predictions)}개의 객체가 검출되었습니다. (분석 중단 및 결과 출력)")
    
    for prediction in predictions:
        x, y, w, h = prediction['x'], prediction['y'], prediction['width'], prediction['height']
        x_min, y_min = int(x - w / 2), int(y - h / 2)
        x_max, y_max = int(x + w / 2), int(y + h / 2)
        
        label = prediction['class']
        confidence = prediction['confidence']
        text = f"{label} {confidence:.2f}"

        # 박스 (빨간색 강조)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
        
        # 라벨
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img, (x_min, y_min - text_h - 10), (x_min + text_w, y_min), (0, 0, 255), -1)
        cv2.putText(img, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2_imshow(img)

def run_matlab_geometry_logic(image_path):
    """
    검출된 것이 없을 때(성공 시) 실행되는 정밀 계측 로직
    """
    print("\n[알림] AI 검출 결과 없음 (성공). 정밀 계측을 시작합니다...")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None: return
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    vis_img = img_bgr.copy()

    # --- A. Scale Bar Check ---
    scale_crop = img_gray[800:1001, 500:2500]
    scale_profile_X = np.mean(scale_crop, axis=0)
    scale_diff_X = np.diff(scale_profile_X)
    peaks, _ = find_peaks(scale_diff_X, height=10, distance=10)
    
    if len(peaks) < 2:
        X_scale_delta = 1.0
    else:
        peak_diffs = np.diff(peaks)
        X_scale_delta = 50 / np.mean(peak_diffs)
    
    print(f"Calculated Scale: {X_scale_delta:.4f} um/px")

    # --- B. Segmentation ---
    offset_y, offset_x = 1400, 600
    obj_crop = img_gray[offset_y:2100, offset_x:4500]
    _, mask = cv2.threshold(obj_crop, 55, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    
    final_mask = np.zeros_like(mask)
    found = False
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 5000:
            final_mask[labels == i] = 1
            found = True

    if not found:
        print("계측 대상 미발견.")
        cv2_imshow(vis_img)
        return

    # --- C. Corner Detection ---
    y_idx, x_idx = np.where(final_mask == 1)
    h, w = final_mask.shape
    pts = np.stack((x_idx, y_idx), axis=1)
    
    # 4방향 거리 계산
    dist_tl = x_idx**2 + y_idx**2
    dist_bl = x_idx**2 + (y_idx - h)**2
    dist_tr = (x_idx - w)**2 + y_idx**2
    dist_br = (x_idx - w)**2 + (y_idx - h)**2
    
    def get_corner(dist_arr, pts_arr):
        idx = np.argsort(dist_arr)[:10]
        top_pts = pts_arr[idx]
        return np.mean(top_pts[:, 0]), np.mean(top_pts[:, 1])

    lt_l, lb_l = get_corner(dist_tl, pts), get_corner(dist_bl, pts)
    rt_l, rb_l = get_corner(dist_tr, pts), get_corner(dist_br, pts)

    def to_global(pt): return (pt[0] + offset_x, pt[1] + offset_y)
    LT, LB, RT, RB = to_global(lt_l), to_global(lb_l), to_global(rt_l), to_global(rb_l)

    # --- D. Calculation ---
    # 각도 계산용 벡터
    def get_angle(p1, p2, p3): 
        # p2가 꼭짓점. Vector p2->p1, p2->p3
        v1 = np.array([p1[0]-p2[0], p1[1]-p2[1]])
        v2 = np.array([p3[0]-p2[0], p3[1]-p2[1]])
        ang = math.degrees(math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0]))
        return abs(ang) if abs(ang) < 180 else 360 - abs(ang) # 단순화된 내각 계산

    # 기존 MATLAB 로직의 각도 계산 방식 유지 (벡터 방향성 고려)
    # (단, 시각화 겹침 해결이 주 목적이므로 값 계산은 기존 로직을 따름)
    VLT_Top   = [RT[0]-LT[0], RT[1]-LT[1]]
    VLT_Left  = [LB[0]-LT[0], LB[1]-LT[1]]
    VLB_Bot   = -1 * np.array([LB[0]-RB[0], LB[1]-RB[1]])
    VLB_Left  = -1 * np.array([LB[0]-LT[0], LB[1]-LT[1]])
    VRB_Bot   = [LB[0]-RB[0], LB[1]-RB[1]]
    VRB_Right = [RT[0]-RB[0], RT[1]-RB[1]]
    VRT_Top   = -1 * np.array([RT[0]-LT[0], RT[1]-LT[1]])
    VRT_Right = -1 * np.array([RT[0]-RB[0], RT[1]-RB[1]])

    def calc_deg(v1, v2, add_360=False):
        val = math.degrees(math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0]))
        if add_360: return val + 360
        return val

    LT_A = calc_deg(VLT_Top, VLT_Left)
    LB_A = calc_deg(VLB_Left, VLB_Bot)
    RT_A = calc_deg(VRT_Right, VRT_Top)
    RB_A = calc_deg(VRB_Right, VRB_Bot) + 360 # MATLAB Logic preserved

    length_px = math.sqrt((RT[0]-LT[0])**2 + (RT[1]-LT[1])**2)
    length_um = length_px * X_scale_delta

    # --- E. Visualization (겹침 방지 로직 적용) ---
    
    # 1. 점(Corner)과 선(Line) 그리기
    corners = [LT, LB, RT, RB]
    for pt in corners:
        cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 20, (0, 255, 255), -1)
    
    pts_poly = np.array([LT, LB, RB, RT], np.int32)
    cv2.polylines(vis_img, [pts_poly], True, (0, 0, 255), 8)

    # 2. 텍스트 위치 계산 (겹침 방지)
    
    # [길이]는 윗변(Top Edge)의 정중앙에 표시
    mid_top_x = (LT[0] + RT[0]) / 2
    mid_top_y = (LT[1] + RT[1]) / 2
    # 선보다 약간 위로 띄움 (y - 40)
    draw_text_centered(vis_img, f"{length_um:.2f} um", mid_top_x, mid_top_y - 40, color=(0, 0, 255)) # 빨간 글씨

    # [각도]는 각 코너에서 바깥쪽으로 밀어내서 표시
    # LT (좌상) -> 더 위, 더 왼쪽
    draw_text_centered(vis_img, f"{LT_A:.2f} deg", LT[0] - 60, LT[1] - 40, color=(255, 0, 0)) # 파란 글씨
    
    # LB (좌하) -> 더 아래, 더 왼쪽
    draw_text_centered(vis_img, f"{LB_A:.2f} deg", LB[0] - 60, LB[1] + 80, color=(255, 0, 0))
    
    # RT (우상) -> 더 위, 더 오른쪽
    draw_text_centered(vis_img, f"{RT_A:.2f} deg", RT[0] + 60, RT[1] - 40, color=(255, 0, 0))
    
    # RB (우하) -> 더 아래, 더 오른쪽
    draw_text_centered(vis_img, f"{RB_A:.2f} deg", RB[0] + 60, RB[1] + 80, color=(255, 0, 0))

    print(f"Measured Length: {length_um:.2f} um")
    cv2_imshow(vis_img)

# ==========================================
# 4. 실행
# ==========================================
print(">> [1단계] Roboflow AI 검사 시작...")
result = run_roboflow_inference(IMAGE_FILE)

if 'predictions' in result and len(result['predictions']) > 0:
    draw_inference_results(IMAGE_FILE, result['predictions'])
    print(">> [결과] 불량/이물질 검출됨. 계측 중단.")
else:
    print(">> [결과] AI 검출 없음(정상).")
    print(">> [2단계] 정밀 계측 로직 실행...")
    run_matlab_geometry_logic(IMAGE_FILE)