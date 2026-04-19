import streamlit as st
import cv2
import numpy as np
from PIL import Image, ExifTags

st.set_page_config(page_title="ICD Precision V2", layout="centered")
st.markdown("<h2 style='text-align:center; color:#00f2fe;'>📊 ICD Bottle Detector 2.0</h2>",
            unsafe_allow_html=True)
st.write("---")

# ── helpers ───────────────────────────────────────────────────────────────────

def fix_exif_rotation(img: Image.Image) -> Image.Image:
    try:
        exif = img._getexif()
        if exif is None:
            return img
        orient_key = next(k for k, v in ExifTags.TAGS.items() if v == "Orientation")
        orientation = exif.get(orient_key)
        rotations = {3: 180, 6: 270, 8: 90}
        if orientation in rotations:
            img = img.rotate(rotations[orientation], expand=True)
    except Exception:
        pass
    return img

def rotate_image(img_np: np.ndarray, angle: int) -> np.ndarray:
    k = (angle // 90) % 4
    return np.rot90(img_np, k=-k)

def find_water_surface(img_cv: np.ndarray):
    """
    Multi-method water surface detection.
    Returns (detected_y, method_name, debug_img)
    Tries 4 methods in order of reliability.
    """
    h, w = img_cv.shape[:2]
    debug = img_cv.copy()

    # ── Method 1: Hough lines (original, strict) ─────────────────────────────
    gray  = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blur, 30, 100)
    lines = cv2.HoughLinesP(edged, 1, np.pi / 180, 40,
                            minLineLength=w // 3, maxLineGap=20)
    if lines is not None:
        h_lines = [(l, l[0][1]) for l in lines if abs(l[0][1] - l[0][3]) < 10]
        if h_lines:
            # pick lowest horizontal line (ignore top 20% = air bubble zone)
            valid = [(l, y) for l, y in h_lines if y > h * 0.2]
            if valid:
                best, y = sorted(valid, key=lambda x: x[1], reverse=True)[0]
                x1, y1, x2, y2 = best[0]
                cv2.line(debug, (x1, y1), (x2, y2), (0, 255, 0), 8)
                return y, "Hough Line", debug

    # ── Method 2: Relaxed Hough (shorter lines, looser threshold) ────────────
    lines2 = cv2.HoughLinesP(edged, 1, np.pi / 180, 20,
                             minLineLength=w // 6, maxLineGap=40)
    if lines2 is not None:
        h_lines2 = [(l, l[0][1]) for l in lines2 if abs(l[0][1] - l[0][3]) < 15]
        valid2 = [(l, y) for l, y in h_lines2 if y > h * 0.2]
        if valid2:
            best, y = sorted(valid2, key=lambda x: x[1], reverse=True)[0]
            x1, y1, x2, y2 = best[0]
            cv2.line(debug, (x1, y1), (x2, y2), (0, 200, 255), 8)
            return y, "Relaxed Hough", debug

    # ── Method 3: Horizontal edge gradient scan ───────────────────────────────
    # Sum horizontal Sobel response row-by-row → peak = surface
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
    row_energy = np.sum(np.abs(sobelx), axis=1)
    # Ignore top 20% and bottom 5%
    search_start = int(h * 0.20)
    search_end   = int(h * 0.95)
    roi_energy   = row_energy[search_start:search_end]
    if roi_energy.max() > 0:
        # Find the LOWEST strong edge (water surface is near bottom of liquid)
        threshold = roi_energy.max() * 0.35
        candidates = np.where(roi_energy > threshold)[0]
        if len(candidates):
            y = int(candidates[-1]) + search_start   # lowest strong edge
            cv2.line(debug, (0, y), (w, y), (0, 165, 255), 8)
            return y, "Gradient Scan", debug

    # ── Method 4: Color discontinuity (liquid vs air brightness jump) ─────────
    # Liquid (saline) is darker/more uniform; air above is brighter
    gray_crop = gray[int(h*0.1):int(h*0.9), :]
    col_mean  = np.mean(gray_crop, axis=1)   # mean brightness per row
    # Smooth and find biggest brightness jump
    smoothed  = np.convolve(col_mean, np.ones(7)/7, mode='same')
    diff      = np.diff(smoothed)
    # Largest negative jump = going from liquid to air (top-down scan)
    # We want the lowest such jump
    threshold = np.std(diff) * 1.2
    jumps     = np.where(np.abs(diff) > threshold)[0]
    if len(jumps):
        y = int(jumps[-1]) + int(h * 0.1)
        cv2.line(debug, (0, y), (w, y), (255, 100, 0), 8)
        return y, "Color Boundary", debug

    return None, None, debug

# ── upload ────────────────────────────────────────────────────────────────────

img_file = st.file_uploader(
    "เลือกรูปขวดน้ำเกลือ (แนวตั้ง หรือแนวนอนก็ได้)",
    type=["jpg", "jpeg", "png"],
)

if img_file:
    pil_img = Image.open(img_file).convert("RGB")
    pil_img = fix_exif_rotation(pil_img)
    img_np  = np.array(pil_img)

    H, W = img_np.shape[:2]
    if W > H:
        img_np = rotate_image(img_np, 90)
        st.toast("📱 ตรวจพบภาพแนวนอน — หมุนอัตโนมัติแล้ว", icon="🔄")

    st.markdown("### 🔄 ปรับการหมุน (ถ้ายังไม่ตรง)")
    extra_rot = st.radio(
        "หมุนเพิ่มเติม",
        options=[0, 90, 180, 270],
        format_func=lambda x: {0:"ปกติ (0°)", 90:"หมุน 90° CW",
                                180:"หมุน 180°", 270:"หมุน 270° CW"}[x],
        horizontal=True, index=0,
    )
    if extra_rot:
        img_np = rotate_image(img_np, extra_rot)

    H, W = img_np.shape[:2]

    # ── crop sliders ──────────────────────────────────────────────────────────
    st.markdown("### ✂️ ตั้งค่ากรอบ Crop")
    st.info("💡 ปรับ slider ให้ขอบบนตรงขีด 900 ml และขอบล่างตรงขีด 100 ml")

    col1, col2 = st.columns(2)
    with col1:
        top_pct    = st.slider("ขอบบน (900 ml) %",    0,  49,  5, key="top")
        left_pct   = st.slider("ขอบซ้าย %",           0,  49, 10, key="left")
    with col2:
        bottom_pct = st.slider("ขอบล่าง (100 ml) %", 51, 100, 95, key="bot")
        right_pct  = st.slider("ขอบขวา %",           51, 100, 90, key="right")

    top    = int(top_pct    / 100 * H)
    bottom = int(bottom_pct / 100 * H)
    left   = int(left_pct   / 100 * W)
    right  = int(right_pct  / 100 * W)

    # live preview
    preview = img_np.copy()
    cv2.rectangle(preview, (left, top), (right, bottom), (255, 0, 0), 3)
    for i in range(9):
        vol = 900 - i * 100
        py  = int(top + i * (bottom - top) / 8)
        color = (0, 0, 255) if vol in [900, 100] else (255, 150, 0)
        cv2.line(preview, (left, py), (right, py), color, 1)
        cv2.putText(preview, f"{vol}", (right + 5, py + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    st.image(preview, caption="Preview — ปรับ slider จนเส้นตรงกับขีดบนขวด",
             use_container_width=True)
    st.write("---")

    # ── sensitivity slider ────────────────────────────────────────────────────
    st.markdown("### 🎚️ ความไวในการตรวจจับ")
    sensitivity = st.select_slider(
        "ถ้าหาไม่เจอ ให้เพิ่มความไว",
        options=["ปกติ", "สูง", "สูงมาก"],
        value="ปกติ",
    )

    # ── calculate ─────────────────────────────────────────────────────────────
    if st.button("🚀 คำนวณปริมาตร"):
        cropped = img_np[top:bottom, left:right]
        img_cv  = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
        h, w    = img_cv.shape[:2]

        # enhance contrast before detection if sensitivity raised
        if sensitivity != "ปกติ":
            lab   = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clip  = 2.0 if sensitivity == "สูง" else 3.5
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
            l     = clahe.apply(l)
            img_cv = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        detected_y, method, output = find_water_surface(img_cv)

        # draw scale lines
        for i in range(9):
            vol_label = 900 - i * 100
            cy    = int(i * (h / 8))
            color = (0, 0, 255) if vol_label in [900, 100] else (255, 150, 0)
            cv2.line(output, (0, cy), (w, cy), color, 2)
            cv2.putText(output, f"{vol_label} ml", (10, cy + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if detected_y is not None:
            volume = 100 + ((h - detected_y) / h * 800)
            volume = max(100, min(900, int(volume)))   # clamp to valid range
            st.markdown(
                f"<h1 style='text-align:center;color:#00ff00;'>{volume} ml</h1>",
                unsafe_allow_html=True)
            st.caption(f"🔍 วิธีที่ใช้: **{method}**")
            st.image(output, channels="BGR",
                     caption="เส้นสีบนภาพ = ระดับน้ำที่ตรวจพบ",
                     use_container_width=True)

            # confidence hint
            if method == "Hough Line":
                st.success("✅ ความแม่นยำสูง — ตรวจพบเส้นขอบน้ำชัดเจน")
            elif method == "Relaxed Hough":
                st.warning("⚠️ ความแม่นยำปานกลาง — ลองถ่ายให้แสงสว่างขึ้นเพื่อผลดีขึ้น")
            else:
                st.warning("⚠️ ความแม่นยำต่ำ — ลองเพิ่มความไวหรือปรับกรอบให้แน่นขึ้น")
        else:
            st.error("❌ หาผิวน้ำไม่เจอแม้ใช้ทุกวิธีแล้ว")
            st.markdown("""
**แนะนำ:**
- 📷 ถ่ายให้แสงสว่าง ไม่มีแสงสะท้อน
- 🔲 ปรับกรอบ Crop ให้แน่น ครอบคลุมผิวน้ำ
- 🎚️ ลองเพิ่มความไวเป็น "สูงมาก"
- 📐 ถ่ายตรงๆ ไม่เอียง
""")
            st.image(output, channels="BGR",
                     caption="ภาพที่วิเคราะห์ (ไม่พบผิวน้ำ)",
                     use_container_width=True)
