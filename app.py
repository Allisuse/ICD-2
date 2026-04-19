import streamlit as st
import cv2
import numpy as np
from PIL import Image, ExifTags
import base64, io
import streamlit.components.v1 as components

st.set_page_config(page_title="ICD Precision V2", layout="centered")
st.markdown("<h2 style='text-align:center; color:#00f2fe;'>📊 ICD Bottle Detector 2.0</h2>",
            unsafe_allow_html=True)
st.write("---")

# ── helpers ───────────────────────────────────────────────────────────────────

def fix_exif_rotation(img):
    try:
        exif = img._getexif()
        if exif is None: return img
        orient_key = next(k for k, v in ExifTags.TAGS.items() if v == "Orientation")
        orientation = exif.get(orient_key)
        for tag, deg in {3:180, 6:270, 8:90}.items():
            if orientation == tag:
                return img.rotate(deg, expand=True)
    except: pass
    return img

def rotate_image(img_np, angle):
    return np.rot90(img_np, k=-(angle // 90) % 4)

def find_water_surface(img_cv):
    h, w  = img_cv.shape[:2]
    debug = img_cv.copy()
    gray  = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (7,7), 0)
    edged = cv2.Canny(blur, 30, 100)

    for thresh, gap, min_len, tol, color, name in [
        (40, 20, w//3,  10, (0,255,0),   "Hough Line"),
        (20, 40, w//6,  15, (0,200,255), "Relaxed Hough"),
    ]:
        lines = cv2.HoughLinesP(edged, 1, np.pi/180, thresh,
                                minLineLength=min_len, maxLineGap=gap)
        if lines is not None:
            valid = [(l, l[0][1]) for l in lines
                     if abs(l[0][1]-l[0][3]) < tol and l[0][1] > h*0.2]
            if valid:
                best, y = sorted(valid, key=lambda x: x[1], reverse=True)[0]
                x1,y1,x2,y2 = best[0]
                cv2.line(debug,(x1,y1),(x2,y2),color,8)
                return y, name, debug

    sobelx     = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
    row_energy = np.sum(np.abs(sobelx), axis=1)
    s, e       = int(h*0.20), int(h*0.95)
    roi        = row_energy[s:e]
    if roi.max() > 0:
        cands = np.where(roi > roi.max()*0.35)[0]
        if len(cands):
            y = int(cands[-1]) + s
            cv2.line(debug,(0,y),(w,y),(0,165,255),8)
            return y, "Gradient Scan", debug

    gray_c   = gray[int(h*0.1):int(h*0.9),:]
    smoothed = np.convolve(np.mean(gray_c,axis=1), np.ones(7)/7, mode='same')
    diff     = np.diff(smoothed)
    jumps    = np.where(np.abs(diff) > np.std(diff)*1.2)[0]
    if len(jumps):
        y = int(jumps[-1]) + int(h*0.1)
        cv2.line(debug,(0,y),(w,y),(255,100,0),8)
        return y, "Color Boundary", debug

    return None, None, debug

def img_to_b64(img_np_rgb):
    pil = Image.fromarray(img_np_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()

def calc_volume(y_px, h):
    return max(100, min(900, int(100 + ((h - y_px) / h * 800))))

def draw_scale_lines(img_cv):
    """Draw only scale reference lines (no water line)."""
    h, w   = img_cv.shape[:2]
    output = img_cv.copy()
    for i in range(9):
        vol   = 900 - i*100
        cy    = int(i*(h/8))
        color = (0,0,255) if vol in [900,100] else (255,150,0)
        cv2.line(output,(0,cy),(w,cy),color,2)
        cv2.putText(output,f"{vol} ml",(10,cy+22),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)
    return output

# ── drag-line canvas component ────────────────────────────────────────────────

def drag_line_canvas(img_b64, auto_y_frac, canvas_w=360):
    """
    Renders image + draggable cyan line in an iframe.
    Returns fractional Y position (0-1) chosen by user via URL fragment trick +
    st.query_params (Streamlit ≥1.30) or falls back to a number_input.
    """
    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#111; display:flex; flex-direction:column;
          align-items:center; font-family:sans-serif; color:#eee; }}
  #wrap {{ position:relative; display:inline-block; touch-action:none; }}
  canvas {{ display:block; max-width:100%; cursor:ns-resize; }}
  #info  {{ margin:8px 0; font-size:1.1em; color:#00f2fe; font-weight:bold; }}
  #hint  {{ font-size:0.8em; color:#aaa; margin-bottom:6px; }}
  #copyBtn {{
    margin:8px; padding:8px 22px; background:#00f2fe; color:#111;
    border:none; border-radius:6px; font-size:1em; font-weight:bold;
    cursor:pointer;
  }}
  #copied {{ color:#0f0; font-size:0.85em; height:1.2em; }}
</style>
</head>
<body>
<div id="hint">👆 แตะหรือลากเส้นสีฟ้าให้ตรงกับผิวน้ำ</div>
<div id="wrap"><canvas id="c"></canvas></div>
<div id="info">ผิวน้ำ: <span id="vol">--</span> ml</div>
<button id="copyBtn" onclick="copyVal()">✅ ยืนยันตำแหน่งนี้</button>
<div id="copied"></div>

<script>
const IMG_B64  = "{img_b64}";
const AUTO_Y   = {auto_y_frac};   // 0-1 fraction
const CANVAS_W = {canvas_w};

const canvas = document.getElementById('c');
const ctx    = canvas.getContext('2d');
const img    = new Image();
let lineY    = 0;   // pixels on canvas
let dragging = false;
let scale    = 1;

img.onload = () => {{
  scale          = CANVAS_W / img.naturalWidth;
  canvas.width   = CANVAS_W;
  canvas.height  = Math.round(img.naturalHeight * scale);
  lineY          = AUTO_Y * canvas.height;
  draw();
}};
img.src = 'data:image/jpeg;base64,' + IMG_B64;

function draw() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  // scale reference lines
  for (let i = 0; i < 9; i++) {{
    const vol = 900 - i*100;
    const y   = Math.round(i * canvas.height / 8);
    ctx.strokeStyle = (vol===900||vol===100) ? '#ff3333' : '#ff9500';
    ctx.lineWidth   = 1.5;
    ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(canvas.width,y); ctx.stroke();
    ctx.fillStyle = ctx.strokeStyle;
    ctx.font = '13px sans-serif';
    ctx.fillText(vol+' ml', 6, y+14);
  }}

  // draggable water line
  ctx.strokeStyle = '#00ffff';
  ctx.lineWidth   = 3;
  ctx.setLineDash([10,5]);
  ctx.beginPath(); ctx.moveTo(0,lineY); ctx.lineTo(canvas.width,lineY); ctx.stroke();
  ctx.setLineDash([]);

  // handle
  ctx.fillStyle = '#00ffff';
  ctx.beginPath();
  ctx.arc(canvas.width/2, lineY, 10, 0, Math.PI*2);
  ctx.fill();

  // volume label on line
  const frac = lineY / canvas.height;
  const vol  = Math.round(Math.max(100, Math.min(900, 100 + (1-frac)*800)));
  document.getElementById('vol').textContent = vol;
  document.getElementById('copyBtn').dataset.vol  = vol;
  document.getElementById('copyBtn').dataset.frac = frac.toFixed(4);
}}

function getY(e) {{
  const r = canvas.getBoundingClientRect();
  if (e.touches) return e.touches[0].clientY - r.top;
  return e.clientY - r.top;
}}
function clamp(v) {{ return Math.max(0, Math.min(canvas.height, v)); }}

canvas.addEventListener('mousedown',  e => {{ dragging=true; lineY=clamp(getY(e)); draw(); }});
canvas.addEventListener('mousemove',  e => {{ if(dragging){{ lineY=clamp(getY(e)); draw(); }} }});
canvas.addEventListener('mouseup',    ()=> dragging=false);
canvas.addEventListener('touchstart', e => {{ e.preventDefault(); dragging=true; lineY=clamp(getY(e)); draw(); }}, {{passive:false}});
canvas.addEventListener('touchmove',  e => {{ e.preventDefault(); if(dragging){{ lineY=clamp(getY(e)); draw(); }} }}, {{passive:false}});
canvas.addEventListener('touchend',   ()=> dragging=false);

function copyVal() {{
  const frac = parseFloat(document.getElementById('copyBtn').dataset.frac || AUTO_Y);
  const vol  = document.getElementById('copyBtn').dataset.vol || '--';
  // post to parent Streamlit via postMessage
  window.parent.postMessage({{type:'streamlit:setComponentValue', value: frac}}, '*');
  document.getElementById('copied').textContent = '✅ บันทึกแล้ว — กด "คำนวณจากเส้น Manual" ด้านล่าง';
}}
</script>
</body>
</html>
"""
    return components.html(html, height=700, scrolling=False)

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

img_file = st.file_uploader(
    "เลือกรูปขวดน้ำเกลือ",
    type=["jpg","jpeg","png"],
)

if img_file:
    pil_img = Image.open(img_file).convert("RGB")
    pil_img = fix_exif_rotation(pil_img)
    img_np  = np.array(pil_img)

    H, W = img_np.shape[:2]
    if W > H:
        img_np = rotate_image(img_np, 90)
        st.toast("📱 ตรวจพบภาพแนวนอน — หมุนอัตโนมัติแล้ว", icon="🔄")

    st.markdown("### 🔄 ปรับการหมุน")
    extra_rot = st.radio(
        "หมุนเพิ่มเติม",
        [0,90,180,270],
        format_func=lambda x:{0:"ปกติ",90:"90° CW",180:"180°",270:"270° CW"}[x],
        horizontal=True, index=0,
    )
    if extra_rot:
        img_np = rotate_image(img_np, extra_rot)
    H, W = img_np.shape[:2]

    # ── crop ─────────────────────────────────────────────────────────────────
    st.markdown("### ✂️ ตั้งค่ากรอบ Crop")
    c1, c2 = st.columns(2)
    with c1:
        top_pct    = st.slider("ขอบบน (900 ml) %",    0, 49,  5, key="top")
        left_pct   = st.slider("ขอบซ้าย %",           0, 49, 10, key="left")
    with c2:
        bottom_pct = st.slider("ขอบล่าง (100 ml) %", 51,100, 95, key="bot")
        right_pct  = st.slider("ขอบขวา %",           51,100, 90, key="right")

    top    = int(top_pct    /100*H)
    bottom = int(bottom_pct /100*H)
    left   = int(left_pct   /100*W)
    right  = int(right_pct  /100*W)

    # preview
    preview = img_np.copy()
    cv2.rectangle(preview,(left,top),(right,bottom),(255,0,0),3)
    for i in range(9):
        vol = 900-i*100
        py  = int(top+i*(bottom-top)/8)
        clr = (0,0,255) if vol in [900,100] else (255,150,0)
        cv2.line(preview,(left,py),(right,py),clr,1)
        cv2.putText(preview,f"{vol}",(right+5,py+6),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,clr,1)
    st.image(preview, caption="Preview", use_container_width=True)
    st.write("---")

    sensitivity = st.select_slider(
        "🎚️ ความไวในการตรวจจับ",
        ["ปกติ","สูง","สูงมาก"], value="ปกติ")

    # ── AUTO detect ──────────────────────────────────────────────────────────
    if st.button("🚀 ตรวจจับอัตโนมัติ"):
        cropped = img_np[top:bottom, left:right]
        img_cv  = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)

        if sensitivity != "ปกติ":
            lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
            l,a,b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0 if sensitivity=="สูง" else 3.5,
                                     tileGridSize=(8,8))
            img_cv = cv2.cvtColor(cv2.merge([clahe.apply(l),a,b]), cv2.COLOR_LAB2BGR)

        detected_y, method, _ = find_water_surface(img_cv)
        st.session_state.img_cv     = img_cv
        st.session_state.detected_y = detected_y
        st.session_state.method     = method
        st.session_state.manual_frac = (detected_y / img_cv.shape[0]
                                         if detected_y is not None else 0.5)
        st.session_state.calculated  = True

    # ── show canvas + results ─────────────────────────────────────────────────
    if st.session_state.get("calculated"):
        img_cv     = st.session_state.img_cv
        detected_y = st.session_state.detected_y
        method     = st.session_state.method
        h, w       = img_cv.shape[:2]

        if detected_y is not None:
            st.success(f"✅ AI ตรวจพบผิวน้ำ — วิธีที่ใช้: **{method}**")
            auto_vol = calc_volume(detected_y, h)
            st.markdown(
                f"<h2 style='text-align:center;color:#00ff00;'>"
                f"🤖 Auto: {auto_vol} ml</h2>", unsafe_allow_html=True)
        else:
            st.error("❌ AI หาผิวน้ำไม่เจอ — ลากเส้นสีฟ้าให้ตรงผิวน้ำด้านล่าง")

        # ── draggable canvas ──────────────────────────────────────────────────
        st.markdown("### 🖐️ ลากเส้นสีฟ้าให้ตรงผิวน้ำ แล้วกด ✅ ยืนยัน")

        # render the cropped image with scale lines baked in for canvas
        scaled_img = draw_scale_lines(img_cv)
        # convert BGR → RGB for display
        scaled_rgb = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2RGB)
        img_b64    = img_to_b64(scaled_rgb)

        auto_frac  = st.session_state.get("manual_frac", 0.5)
        drag_line_canvas(img_b64, auto_frac, canvas_w=360)

        # ── manual Y input (fallback for postMessage) ─────────────────────────
        st.markdown("**หรือพิมพ์ตำแหน่งผิวน้ำ (% จากบน):**")
        manual_pct = st.number_input(
            "% จากขอบบนของภาพ Crop (0 = บนสุด, 100 = ล่างสุด)",
            min_value=1, max_value=99,
            value=int(st.session_state.get("manual_frac", 0.5)*100),
            step=1, key="manual_pct_input",
        )

        if st.button("📐 คำนวณจากเส้น Manual"):
            manual_y   = int(manual_pct / 100 * h)
            manual_vol = calc_volume(manual_y, h)
            st.markdown(
                f"<h1 style='text-align:center;color:#ffff00;'>"
                f"🖐️ Manual: {manual_vol} ml</h1>", unsafe_allow_html=True)

            # final output image
            output = draw_scale_lines(img_cv)
            if detected_y is not None:
                cv2.line(output,(0,detected_y),(w,detected_y),(0,255,0),4)
                cv2.putText(output,"AUTO",(w-100,detected_y-8),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
            cv2.line(output,(0,manual_y),(w,manual_y),(0,255,255),5)
            cv2.putText(output,"MANUAL",(w-140,manual_y-8),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
            st.image(output, channels="BGR",
                     caption="🟢 Auto  |  🟡 Manual  |  เส้นแดง/ส้ม = สเกล",
                     use_container_width=True)

            if detected_y is not None:
                st.caption(f"🟢 Auto: {calc_volume(detected_y,h)} ml  |  "
                           f"🟡 Manual: {manual_vol} ml")
