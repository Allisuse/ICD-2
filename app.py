import streamlit as st
import cv2
import numpy as np
from PIL import Image, ExifTags

st.set_page_config(page_title="ICD Precision V2", layout="centered")

st.markdown("""
<h2 style='text-align:center; color:#00f2fe;'>📊 ICD Bottle Detector 2.0</h2>
""", unsafe_allow_html=True)
st.write("---")

# ── helpers ──────────────────────────────────────────────────────────────────

def fix_exif_rotation(img: Image.Image) -> Image.Image:
    """Auto-rotate based on EXIF orientation tag."""
    try:
        exif = img._getexif()
        if exif is None:
            return img
        orient_key = next(
            k for k, v in ExifTags.TAGS.items() if v == "Orientation"
        )
        orientation = exif.get(orient_key)
        rotations = {3: 180, 6: 270, 8: 90}
        if orientation in rotations:
            img = img.rotate(rotations[orientation], expand=True)
    except Exception:
        pass
    return img


def rotate_image(img_np: np.ndarray, angle: int) -> np.ndarray:
    """Rotate numpy image by 0/90/180/270 degrees."""
    k = (angle // 90) % 4
    return np.rot90(img_np, k=-k)   # rot90 positive = CCW, so negate for CW


# ── upload ────────────────────────────────────────────────────────────────────

img_file = st.file_uploader(
    "เลือกรูปขวดน้ำเกลือ (ถ่ายแนวตั้ง หรือแนวนอนก็ได้)",
    type=["jpg", "jpeg", "png"],
)

if img_file:
    pil_img = Image.open(img_file).convert("RGB")

    # 1️⃣  Auto-fix EXIF orientation (handles most mobile photos)
    pil_img = fix_exif_rotation(pil_img)
    img_np  = np.array(pil_img)

    # 2️⃣  If still landscape → rotate 90° CW automatically
    H, W = img_np.shape[:2]
    if W > H:
        img_np = rotate_image(img_np, 90)
        st.toast("📱 ตรวจพบภาพแนวนอน — หมุนอัตโนมัติแล้ว", icon="🔄")

    # 3️⃣  Manual rotation override
    st.markdown("### 🔄 ปรับการหมุน (ถ้ายังไม่ตรง)")
    extra_rot = st.radio(
        "หมุนเพิ่มเติม",
        options=[0, 90, 180, 270],
        format_func=lambda x: {0: "ปกติ (0°)", 90: "หมุน 90° CW",
                                180: "หมุน 180°", 270: "หมุน 270° CW"}[x],
        horizontal=True,
        index=0,
    )
    if extra_rot:
        img_np = rotate_image(img_np, extra_rot)

    H, W = img_np.shape[:2]   # refresh after rotation

    # ── crop sliders ─────────────────────────────────────────────────────────

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

    # ── live preview ──────────────────────────────────────────────────────────

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

    # ── calculate ─────────────────────────────────────────────────────────────

    if st.button("🚀 คำนวณปริมาตร"):
        cropped = img_np[top:bottom, left:right]
        img_cv  = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
        h, w    = img_cv.shape[:2]
        output  = img_cv.copy()

        for i in range(9):
            vol_label = 900 - i * 100
            cy    = int(i * (h / 8))
            color = (0, 0, 255) if vol_label in [900, 100] else (255, 150, 0)
            cv2.line(output, (0, cy), (w, cy), color, 2)
            cv2.putText(output, f"{vol_label} ml", (10, cy + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        gray  = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(cv2.GaussianBlur(gray, (7, 7), 0), 30, 100)
        lines = cv2.HoughLinesP(edged, 1, np.pi / 180, 40,
                                minLineLength=w // 3, maxLineGap=20)

        detected_y = None
        if lines is not None:
            lines = sorted(lines, key=lambda l: l[0][1], reverse=True)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y1 - y2) < 10:
                    detected_y = y1
                    cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 8)
                    break

        if detected_y is not None:
            volume = 100 + ((h - detected_y) / h * 800)
            st.markdown(
                f"<h1 style='text-align:center;color:#00ff00;'>{int(volume)} ml</h1>",
                unsafe_allow_html=True,
            )
            st.image(output, channels="BGR",
                     caption="เส้นเขียว = ระดับน้ำที่ตรวจพบ",
                     use_container_width=True)
        else:
            st.error("❌ หาผิวน้ำไม่เจอ — ลองขยับ slider ให้ครอบคลุมผิวน้ำให้ชัดขึ้น")
