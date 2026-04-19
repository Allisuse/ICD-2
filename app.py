import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_cropper import st_cropper

st.set_page_config(page_title="ICD Precision Measure", layout="centered")

# ปรับสไตล์ UI ให้ดูสะอาดตา
st.markdown("""
<style>
    .main { background: #0a0a1a; }
    h3 { color: #00f2fe; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h3>📊 ICD Bottle: Manual Alignment</h3>", unsafe_allow_html=True)

# 1. อัปโหลดรูปภาพ
img_file = st.file_uploader("เลือกรูปขวดน้ำเกลือที่ต้องการวิเคราะห์", type=["jpg", "jpeg", "png"])

if img_file:
    img = Image.open(img_file)
    
    st.warning("⚠️ ขั้นตอนสำคัญ: ปรับกรอบสีแดงให้ 'ขอบบน' ตรงกับขีด 900 ml และ 'ขอบล่าง' ตรงกับขีด 100 ml ของขวดจริง")
    
    # 2. ส่วนการ Crop (ใช้เป็นตัวกำหนดมาตรวัด)
    # เราไม่ฟิก aspect ratio เพื่อให้ผู้ใช้ปรับความสูงตามขนาดขวดในรูปได้อิสระ
    cropped_img = st_cropper(img, realtime_update=True, box_color='#FF0000', aspect_ratio=None)
    
    if st.button("✅ ยืนยันตำแหน่งขีดและคำนวณ"):
        # แปลงรูปที่ Crop เป็น OpenCV
        img_array = np.array(cropped_img.convert('RGB'))
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        h, w = img_cv.shape[:2]

        # มาตรวัด: ขอบบนคือ 900, ขอบล่างคือ 100
        px_900 = 0
        px_100 = h

        # 3. ประมวลผลหาผิวน้ำ
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edged = cv2.Canny(blurred, 30, 100)
        
        # หาเส้นผิวน้ำ (เส้นแนวนอน)
        lines = cv2.HoughLinesP(edged, 1, np.pi/180, threshold=40, minLineLength=w//3, maxLineGap=20)

        detected_y = None
        output = img_cv.copy()

        # วาดเส้นสเกลอ้างอิง (200-800 ml) เหมือนในรูปที่คุณส่งมา
        # สเกลมีทั้งหมด 8 ช่วง (900 - 100 = 800 ml)
        for i in range(0, 9):
            vol_label = 900 - (i * 100)
            current_y = int(px_900 + (i * (h / 8)))
            
            # สีเส้น: 900 และ 100 เป็นสีแดง, อื่นๆ เป็นสีน้ำเงิน (ตามรูปต้นฉบับ)
            color = (0, 0, 255) if vol_label in [900, 100] else (255, 100, 0)
            thickness = 3 if vol_label in [900, 100] else 1
            
            cv2.line(output, (0, current_y), (w, current_y), color, thickness)
            cv2.putText(output, f"{vol_label} ml", (10, current_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # ค้นหาเส้นระดับน้ำจริง
        if lines is not None:
            # เรียงจากล่างขึ้นบนเพื่อให้เจอผิวน้ำที่อยู่ล่างสุดก่อน (กรณีมีฟอง)
            lines = sorted(lines, key=lambda l: l[0][1], reverse=True)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y1 - y2) < 10: # กรองเฉพาะเส้นแนวนอน
                    detected_y = y1
                    # วาดเส้นระดับน้ำที่ตรวจพบเป็นสีเขียว
                    cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 6)
                    break

        # 4. คำนวณและแสดงลัพธ์
        if detected_y is not None:
            # คำนวณปริมาตรตามสัดส่วน
            dist_from_bottom = px_100 - detected_y
            volume = 100 + (dist_from_bottom / h * 800)
            
            st.markdown(f"<h1 style='text-align:center; color:#00ff00;'>{int(volume)} ml</h1>", unsafe_allow_html=True)
            st.image(output, channels="BGR", use_container_width=True)
        else:
            st.error("❌ หาผิวน้ำไม่เจอ ลองปรับกรอบให้ครอบคลุมผิวน้ำชัดๆ หรือเลือกรูปที่แสงสะท้อนน้อยลง")
            st.image(output, channels="BGR", caption="เส้นสเกลที่คุณตั้งไว้", use_container_width=True)