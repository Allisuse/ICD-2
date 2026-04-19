import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_cropper import st_cropper

st.set_page_config(page_title="ICD Precision V2", layout="centered")
st.markdown("<h2 style='text-align: center; color: #00f2fe;'>📊 ICD Bottle Detector 2.0</h2>", unsafe_allow_html=True)
st.write("---")

# 1. อัปโหลดรูปภาพ
img_file = st.file_uploader("เลือกรูปขวดน้ำเกลือ (ถ่ายแนวตั้ง)", type=["jpg", "jpeg", "png"])

if img_file:
    img = Image.open(img_file).convert("RGB")

    st.info("💡 วิธีใช้: ปรับกรอบสีแดงให้ 'ขอบบน' ตรงกับขีด 900 และ 'ขอบล่าง' ตรงกับขีด 100")

    # 2. ส่วนการ Crop (Manual Alignment)
    cropped_img = st_cropper(
        img,
        realtime_update=True,
        box_color="#FF0000",
        aspect_ratio=None,
        return_type="image",   # ← required by streamlit-cropper
    )

    if st.button("🚀 คำนวณปริมาตร"):
        # แปลงรูปที่ Crop เป็น OpenCV
        img_cv = cv2.cvtColor(np.array(cropped_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        h, w = img_cv.shape[:2]

        output = img_cv.copy()

        # วาดเส้นสเกลอ้างอิง 100-900 ml
        for i in range(9):
            vol_label = 900 - (i * 100)
            current_y = int(i * (h / 8))
            color = (0, 0, 255) if vol_label in [900, 100] else (255, 150, 0)
            cv2.line(output, (0, current_y), (w, current_y), color, 2)
            cv2.putText(output, f"{vol_label}", (10, current_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 3. AI ค้นหาระดับน้ำ (เส้นสีเขียว)
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

        # 4. คำนวณและแสดงผล
        if detected_y is not None:
            volume = 100 + ((h - detected_y) / h * 800)
            st.markdown(
                f"<h1 style='text-align:center; color:#00ff00;'>{int(volume)} ml</h1>",
                unsafe_allow_html=True,
            )
            st.image(output, channels="BGR",
                     caption="ผลการวิเคราะห์ (เส้นเขียวคือระดับน้ำที่ตรวจพบ)",
                     use_container_width=True)
        else:
            st.error("❌ หาผิวน้ำไม่เจอ ลองปรับกรอบให้ครอบคลุมผิวน้ำชัดๆ")
