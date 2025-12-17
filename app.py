import sys
import asyncio
import os

# ==========================================
# 1. SYSTEM CONFIG & ERROR HANDLING
# ==========================================
# Thiết lập event loop policy trên Windows để tránh lỗi Proactor/asyncio
if sys.platform == 'win32':
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

# Ngăn Gradio cố tạo share link qua Internet
os.environ.setdefault('GRADIO_SHARE', 'False')

import gradio as gr
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import tempfile

# ==========================================
# 2. CORE ALGORITHMS
# ==========================================

def median_filter(curve, kernel_size=5):
    """Lọc median loại bỏ spike/outlier trong signal 1D."""
    if kernel_size <= 1 or len(curve) == 0:
        return curve
    pad = kernel_size // 2
    curve_pad = np.pad(curve, (pad, pad), 'edge')
    out = np.empty_like(curve)
    for i in range(len(curve)):
        out[i] = np.median(curve_pad[i:i+kernel_size])
    return out

def moving_average(curve, radius):
    """
    Làm mượt dữ liệu (Smoothing Trajectory).
    Sử dụng kỹ thuật tích chập (Convolution) với cửa sổ trượt.
    """
    window_size = 2 * radius + 1
    # Tạo bộ lọc (kernel) trung bình
    f = np.ones(window_size) / window_size
    # Padding biên để giữ nguyên kích thước mảng sau khi lọc
    curve_pad = np.pad(curve, (radius, radius), 'edge')
    # Áp dụng tích chập
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Cắt bỏ phần padding
    return curve_smoothed[radius:-radius]

def compute_metrics_safe(img1, img2):
    """
    Tính PSNR và SSIM trên ảnh thu nhỏ để tối ưu hiệu năng.
    """
    # Resize về chiều rộng 320px để tính toán nhanh
    h, w = img1.shape[:2]
    scale = 320 / w
    new_size = (320, int(h * scale))
    
    s1 = cv2.resize(img1, new_size)
    s2 = cv2.resize(img2, new_size)
    
    # Tính PSNR
    psnr_val = cv2.PSNR(s1, s2)
    
    # Tính SSIM (cần ảnh xám)
    g1 = cv2.cvtColor(s1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(s2, cv2.COLOR_BGR2GRAY)
    ssim_val = ssim(g1, g2, data_range=g2.max() - g2.min())
    
    return psnr_val, ssim_val   

# ==========================================
# 3. MAIN PROCESSING PIPELINE
# ==========================================

def process_video(input_path, tech_detector, smoothing_radius, fast_mode=False, progress=gr.Progress()):
    """
    Hàm xử lý chính: Nhận video -> Ổn định -> Xuất video & Metrics
    """
    if input_path is None:
        return None, None
    
    # --- SETUP INPUT ---
    cap = cv2.VideoCapture(input_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # --- SETUP OUTPUT ---
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_temp = tmpfile.name
    tmpfile.close()
    # Codec mp4v tương thích tốt với OpenCV cơ bản
    # Choose metric frequency to reduce load in fast_mode
    metric_freq = 4 if fast_mode else 2
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_temp, fourcc, fps, (w, h))

    # --- SETUP DETECTOR (adjust for speed_mode) ---
    if tech_detector == "ORB":
        nfeat = 1000 if fast_mode else 5000
        detector = cv2.ORB_create(nfeatures=nfeat)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else: # SIFT
        # SIFT is slower; if fast_mode, use fewer keypoints by limiting detection later
        detector = cv2.SIFT_create()
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Resize factor used only for feature detection/matching when fast_mode
    if fast_mode:
        scale = 0.5 if max(w, h) > 800 else 1.0
    else:
        scale = 1.0

    # ==========================================
    # PASS 1: MOTION ESTIMATION (Ước lượng chuyển động)
    # ==========================================
    transforms = [] # Lưu trữ dx, dy, da cho từng frame
    
    _, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Pre-pad transform đầu tiên là 0
    transforms.append([0, 0, 0]) 

    progress(0, desc="Giai đoạn 1/2: Phân tích chuyển động...")
    
    for i in range(n_frames - 2):
        success, curr_frame = cap.read()
        if not success: break
        
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        # use resized gray for faster feature detection in fast mode
        if scale != 1.0:
            prev_gray_small = cv2.resize(prev_gray, (0,0), fx=scale, fy=scale)
            curr_gray_small = cv2.resize(curr_gray, (0,0), fx=scale, fy=scale)
        else:
            prev_gray_small = prev_gray
            curr_gray_small = curr_gray
        
        # 1. Detect Features (on small images if fast_mode)
        kp1, des1 = detector.detectAndCompute(prev_gray_small, None)
        kp2, des2 = detector.detectAndCompute(curr_gray_small, None)
        
        # Mặc định: không chuyển động
        delta_x, delta_y, delta_angle = 0, 0, 0
        
        if des1 is not None and des2 is not None:
            # 2. Match Features
            matches = matcher.match(des1, des2)
            # Lọc lấy top matches (giảm khi fast_mode)
            max_matches = 200 if fast_mode else 500
            matches = sorted(matches, key=lambda x: x.distance)[:max_matches]
            
            if len(matches) > 10:
                # Trích xuất tọa độ điểm
                src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                # If detection was done on scaled images, map points back to original coords
                if scale != 1.0:
                    src_pts = src_pts / scale
                    dst_pts = dst_pts / scale
                
                # 3. Estimate Transform (với RANSAC)
                # estimateAffinePartial2D tốt hơn estimateAffine2D cho video quay tay
                # vì nó hạn chế biến dạng (chỉ tịnh tiến + xoay + scale)
                m_trans, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                
                if m_trans is not None:
                    # Trích xuất tham số từ ma trận 2x3
                    delta_x = m_trans[0, 2]
                    delta_y = m_trans[1, 2]
                    delta_angle = np.arctan2(m_trans[1, 0], m_trans[0, 0])

        transforms.append([delta_x, delta_y, delta_angle])
        prev_gray = curr_gray

    # ==========================================
    # TRAJECTORY SMOOTHING (Làm mượt quỹ đạo)
    # ==========================================
    transforms = np.array(transforms)
    
    # Tính quỹ đạo tích lũy (Cumulative Trajectory)
    trajectory = np.cumsum(transforms, axis=0)
    
    # Làm mượt quỹ đạo với median filter trước (loại spike)
    smoothed_trajectory = np.copy(trajectory)
    for i in range(3): # 0:x, 1:y, 2:angle
        smoothed_trajectory[:, i] = median_filter(trajectory[:, i], kernel_size=5)
        smoothed_trajectory[:, i] = moving_average(smoothed_trajectory[:, i], smoothing_radius)
        
    # Tính độ lệch cần bù trừ (Correction)
    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference

    # ==========================================
    # PASS 2: RENDERING & METRICS (Xuất video & Tính toán)
    # ==========================================
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    psnr_history = []
    ssim_history = []
    
    progress(0.5, desc="Giai đoạn 2/2: Ổn định hình ảnh & Render...")

    for i in range(len(transforms_smooth) - 1):
        success, frame = cap.read()
        if not success: break
        
        # Lấy tham số transform đã làm mượt
        dx, dy, da = transforms_smooth[i]
        
        # Tạo ma trận xoay quanh tâm khung hình (tốt hơn xoay quanh gốc)
        center = (w / 2.0, h / 2.0)
        angle_deg = (da * 180.0) / np.pi
        m = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        # Thêm phần tịnh tiến
        m[0, 2] += dx
        m[1, 2] += dy
        
        # Warp ảnh (Stabilize) với BORDER_REPLICATE để tránh viền đen
        frame_stabilized = cv2.warpAffine(frame, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        # FIX BORDER: Zoom nhẹ 4% để cắt bỏ viền do warp
        scale_zoom = 1.04
        M_zoom = cv2.getRotationMatrix2D((w/2, h/2), 0, scale_zoom)
        frame_stabilized = cv2.warpAffine(frame_stabilized, M_zoom, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        # Ghi video
        out.write(frame_stabilized)
        
        # Tính Metrics (giảm tần suất khi fast_mode để tăng tốc)
        if i % metric_freq == 0:
            p, s = compute_metrics_safe(frame, frame_stabilized)
            psnr_history.append(p)
            ssim_history.append(s)
        else:
            # Dùng lại giá trị cũ để giữ array liên tục
            if psnr_history:
                psnr_history.append(psnr_history[-1])
                ssim_history.append(ssim_history[-1])
            else:
                psnr_history.append(0)
                ssim_history.append(0)

    cap.release()
    out.release()
    
    # ==========================================
    # VISUALIZATION (Vẽ biểu đồ)
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(psnr_history, color='tab:blue')
    ax1.set_title('PSNR (Độ nhiễu tín hiệu)')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('dB')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(ssim_history, color='tab:orange')
    ax2.set_title('SSIM (Độ tương đồng cấu trúc)')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Index (0-1)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()

    # Ensure the temp file is readable by other processes on Windows
    try:
        os.chmod(output_temp, 0o666)
    except Exception:
        pass

    return output_temp, fig

# ==========================================
# 4. GRADIO INTERFACE
# ==========================================
with gr.Blocks(title="Professional Video Stabilizer") as demo:
    gr.Markdown(
        """
        # 🎥 Video Stabilization System (Detailed Implementation)
        Hệ thống ổn định video sử dụng pipeline: **Feature Matching -> Motion Estimation (RANSAC) -> Trajectory Smoothing**.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            input_video = gr.Video(label="Input Video", sources=["upload"])
            
            with gr.Group():
                gr.Markdown("### ⚙️ Cấu hình thuật toán (Enhanced)")
                rad_detector = gr.Radio(["ORB", "SIFT"], value="ORB", label="Phương pháp trích xuất đặc trưng")
                slider_smooth = gr.Slider(15, 80, value=35, step=5, label="Bán kính làm mượt (Smoothing Radius)")
                fast_mode = gr.Checkbox(value=False, label="Fast Mode (tăng tốc, giảm chất lượng)")
                gr.Info("✨ Cải tiến: Median filter + nfeatures 5000 + Center rotation + BORDER_REPLICATE")
            
            btn_run = gr.Button("🚀 Bắt đầu xử lý", variant="primary")
            
        with gr.Column(scale=1):
            output_video = gr.Video(label="Kết quả Ổn định (Stabilized)")
            plot_result = gr.Plot(label="Phân tích chất lượng (PSNR/SSIM)")

    btn_run.click(
        fn=process_video,
        inputs=[input_video, rad_detector, slider_smooth, fast_mode],
        outputs=[output_video, plot_result]
    )

if __name__ == "__main__":
    # share=False để chạy local nhanh hơn
    demo.launch(share=False)