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
    h, w = img_cv.shape[:2]
    gray  = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (7,7), 0)
    edged = cv2.Canny(blur, 30, 100)

    for thresh, gap, min_len, tol, name in [
        (40, 20, w//3,  10, "Hough Line"),
        (20, 40, w//6,  15, "Relaxed Hough"),
    ]:
        lines = cv2.HoughLinesP(edged, 1, np.pi/180, thresh,
                                minLineLength=min_len, maxLineGap=gap)
        if lines is not None:
            valid = [(l, l[0][1]) for l in lines
                     if abs(l[0][1]-l[0][3]) < tol and l[0][1] > h*0.2]
            if valid:
                _, y = sorted(valid, key=lambda x: x[1], reverse=True)[0]
                return y, name

    sobelx     = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
    row_energy = np.sum(np.abs(sobelx), axis=1)
    s, e       = int(h*0.20), int(h*0.95)
    roi        = row_energy[s:e]
    if roi.max() > 0:
        cands = np.where(roi > roi.max()*0.35)[0]
        if len(cands):
            return int(cands[-1]) + s, "Gradient Scan"

    gray_c   = gray[int(h*0.1):int(h*0.9),:]
    smoothed = np.convolve(np.mean(gray_c,axis=1), np.ones(7)/7, mode='same')
    diff     = np.diff(smoothed)
    jumps    = np.where(np.abs(diff) > np.std(diff)*1.2)[0]
    if len(jumps):
        return int(jumps[-1]) + int(h*0.1), "Color Boundary"

    return None, None

def img_to_b64(img_np_rgb):
    pil = Image.fromarray(img_np_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()

def calc_volume(water_y, top_y, bottom_y):
    """Calculate volume using draggable top/bottom as 900/100 ml references."""
    if bottom_y == top_y: return 0
    frac = (bottom_y - water_y) / (bottom_y - top_y)
    vol  = 100 + frac * 800
    return max(50, min(950, int(vol)))

# ── three-line draggable canvas ───────────────────────────────────────────────

def three_line_canvas(img_b64, top_frac, water_frac, bottom_frac, canvas_w=360):
    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
  background:#111; display:flex; flex-direction:column;
  align-items:center; font-family:sans-serif; color:#eee;
  padding-bottom: 12px;
}}
#hint {{ font-size:0.82em; color:#aaa; margin:8px 4px 4px; text-align:center; }}
#wrap {{ position:relative; touch-action:none; }}
canvas {{ display:block; cursor:ns-resize; }}
#legend {{
  display:flex; gap:14px; justify-content:center;
  margin:8px 0 4px; font-size:0.85em;
}}
.dot {{ width:12px; height:12px; border-radius:50%; display:inline-block; margin-right:4px; vertical-align:middle; }}
#vol {{
  font-size:1.6em; font-weight:bold; color:#00ff88;
  margin:4px 0; text-align:center;
}}
#confirmBtn {{
  margin:8px; padding:9px 28px; background:#00f2fe; color:#111;
  border:none; border-radius:8px; font-size:1em; font-weight:bold; cursor:pointer;
}}
#saved {{ color:#0f0; font-size:0.85em; min-height:1.2em; text-align:center; }}
#inputs {{ display:flex; gap:8px; margin-top:6px; font-size:0.82em; flex-wrap:wrap; justify-content:center; }}
#inputs label {{ color:#aaa; }}
#inputs input {{ width:60px; padding:3px 5px; background:#222; color:#eee; border:1px solid #555; border-radius:4px; }}
</style>
</head>
<body>
<div id="hint">👆 แตะหรือลากเส้นใดก็ได้เพื่อปรับตำแหน่ง</div>

<div id="legend">
  <span><span class="dot" style="background:#ff3333"></span>900 ml (บน)</span>
  <span><span class="dot" style="background:#00cc00"></span>ผิวน้ำ</span>
  <span><span class="dot" style="background:#3399ff"></span>100 ml (ล่าง)</span>
</div>

<div id="wrap"><canvas id="c"></canvas></div>
<div id="vol">-- ml</div>

<div id="inputs">
  <label>🔴 บน %: <input type="number" id="inTop"   min="0" max="99" value="{int(top_frac*100)}"></label>
  <label>🟢 น้ำ %: <input type="number" id="inWater" min="0" max="99" value="{int(water_frac*100)}"></label>
  <label>🔵 ล่าง %: <input type="number" id="inBot"   min="0" max="99" value="{int(bottom_frac*100)}"></label>
  <button onclick="applyInputs()" style="padding:3px 10px;background:#555;color:#eee;border:none;border-radius:4px;cursor:pointer;">ใช้ค่า</button>
</div>

<button id="confirmBtn" onclick="confirm()">✅ ยืนยัน — บันทึกค่า</button>
<div id="saved"></div>

<script>
const IMG_B64     = "{img_b64}";
const CANVAS_W    = {canvas_w};
let topFrac       = {top_frac};
let waterFrac     = {water_frac};
let bottomFrac    = {bottom_frac};

const canvas  = document.getElementById('c');
const ctx     = canvas.getContext('2d');
const img     = new Image();
let imgH      = 0;
let dragging  = null;  // 'top' | 'water' | 'bottom' | null
const HIT     = 18;    // px hit zone

img.onload = () => {{
  const scale   = CANVAS_W / img.naturalWidth;
  canvas.width  = CANVAS_W;
  canvas.height = imgH = Math.round(img.naturalHeight * scale);
  draw();
}};
img.src = 'data:image/jpeg;base64,' + IMG_B64;

function fracToY(f)  {{ return f * imgH; }}
function yToFrac(y)  {{ return Math.max(0.01, Math.min(0.99, y / imgH)); }}
function clampY(y)   {{ return Math.max(0, Math.min(imgH, y)); }}

function calcVol() {{
  const tY = fracToY(topFrac), bY = fracToY(bottomFrac), wY = fracToY(waterFrac);
  if (bY === tY) return 0;
  const frac = (bY - wY) / (bY - tY);
  return Math.round(Math.max(50, Math.min(950, 100 + frac * 800)));
}}

function drawLine(y, color, label, dashed) {{
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth   = 3.5;
  if (dashed) ctx.setLineDash([12, 6]);
  else        ctx.setLineDash([]);
  ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
  ctx.restore();

  // handle circle
  ctx.save();
  ctx.fillStyle   = color;
  ctx.strokeStyle = '#fff';
  ctx.lineWidth   = 1.5;
  ctx.beginPath();
  ctx.arc(canvas.width / 2, y, 11, 0, Math.PI * 2);
  ctx.fill(); ctx.stroke();
  ctx.restore();

  // label
  ctx.save();
  ctx.fillStyle = color;
  ctx.font      = 'bold 13px sans-serif';
  ctx.fillText(label, 6, y - 5);
  ctx.restore();
}}

function draw() {{
  ctx.clearRect(0, 0, canvas.width, imgH);
  ctx.drawImage(img, 0, 0, canvas.width, imgH);

  const tY = fracToY(topFrac);
  const bY = fracToY(bottomFrac);
  const wY = fracToY(waterFrac);

  // shaded liquid zone (top → water)
  ctx.save();
  ctx.fillStyle = 'rgba(0,200,100,0.08)';
  ctx.fillRect(0, tY, canvas.width, wY - tY);
  ctx.restore();

  drawLine(tY, '#ff3333', '🔴 900 ml', false);
  drawLine(bY, '#3399ff', '🔵 100 ml', false);
  drawLine(wY, '#00cc00', '🟢 ผิวน้ำ',  true);

  const vol = calcVol();
  document.getElementById('vol').textContent = vol + ' ml';

  // sync number inputs
  document.getElementById('inTop').value   = Math.round(topFrac   * 100);
  document.getElementById('inWater').value = Math.round(waterFrac * 100);
  document.getElementById('inBot').value   = Math.round(bottomFrac* 100);
}}

function getY(e) {{
  const r = canvas.getBoundingClientRect();
  const clientY = e.touches ? e.touches[0].clientY : e.clientY;
  return clientY - r.top;
}}

function hitLine(y, frac) {{ return Math.abs(y - fracToY(frac)) < HIT; }}

function whichLine(y) {{
  // priority: water > top > bottom (water is most frequently adjusted)
  if (hitLine(y, waterFrac))  return 'water';
  if (hitLine(y, topFrac))    return 'top';
  if (hitLine(y, bottomFrac)) return 'bottom';
  return null;
}}

function onDown(e) {{
  e.preventDefault();
  const y = getY(e);
  dragging = whichLine(y);
  if (!dragging) {{
    // If no line hit, move nearest line
    const dists = [
      ['top',    Math.abs(y - fracToY(topFrac))],
      ['water',  Math.abs(y - fracToY(waterFrac))],
      ['bottom', Math.abs(y - fracToY(bottomFrac))],
    ];
    dragging = dists.sort((a,b) => a[1]-b[1])[0][0];
  }}
  onMove(e);
}}

function onMove(e) {{
  if (!dragging) return;
  e.preventDefault();
  const f = yToFrac(clampY(getY(e)));
  if      (dragging === 'top')    topFrac    = Math.min(f, waterFrac - 0.02, bottomFrac - 0.04);
  else if (dragging === 'water')  waterFrac  = Math.max(topFrac + 0.02, Math.min(f, bottomFrac - 0.02));
  else if (dragging === 'bottom') bottomFrac = Math.max(f, waterFrac + 0.02, topFrac + 0.04);
  draw();
}}

function onUp() {{ dragging = null; }}

canvas.addEventListener('mousedown',  onDown, {{passive:false}});
canvas.addEventListener('mousemove',  onMove, {{passive:false}});
canvas.addEventListener('mouseup',    onUp);
canvas.addEventListener('touchstart', onDown, {{passive:false}});
canvas.addEventListener('touchmove',  onMove, {{passive:false}});
canvas.addEventListener('touchend',   onUp);

function applyInputs() {{
  topFrac    = Math.max(0.01, Math.min(0.97, parseInt(document.getElementById('inTop').value)   / 100));
  waterFrac  = Math.max(0.01, Math.min(0.97, parseInt(document.getElementById('inWater').value) / 100));
  bottomFrac = Math.max(0.01, Math.min(0.99, parseInt(document.getElementById('inBot').value)   / 100));
  draw();
}}

function confirm() {{
  const vol = calcVol();
  window.parent.postMessage({{
    type: 'streamlit:setComponentValue',
    value: {{ top: topFrac, water: waterFrac, bottom: bottomFrac, vol: vol }}
  }}, '*');
  document.getElementById('saved').textContent =
    '✅ บันทึกแล้ว — ' + vol + ' ml  |  กด "แสดงผลสุดท้าย" ด้านล่าง';
}}
</script>
</body>
</html>
"""
    return components.html(html, height=780, scrolling=False)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

img_file = st.file_uploader("เลือกรูปขวดน้ำเกลือ", type=["jpg","jpeg","png"])

if img_file:
    pil_img = Image.open(img_file).convert("RGB")
    pil_img = fix_exif_rotation(pil_img)
    img_np  = np.array(pil_img)

    H, W = img_np.shape[:2]
    if W > H:
        img_np = rotate_image(img_np, 90)
        st.toast("📱 หมุนอัตโนมัติแล้ว", icon="🔄")

    st.markdown("### 🔄 ปรับการหมุน")
    extra_rot = st.radio("หมุนเพิ่มเติม", [0,90,180,270],
        format_func=lambda x:{0:"ปกติ",90:"90° CW",180:"180°",270:"270° CW"}[x],
        horizontal=True, index=0)
    if extra_rot:
        img_np = rotate_image(img_np, extra_rot)
    H, W = img_np.shape[:2]

    # ── sensitivity ───────────────────────────────────────────────────────────
    sensitivity = st.select_slider("🎚️ ความไวในการตรวจจับ",
                                   ["ปกติ","สูง","สูงมาก"], value="ปกติ")

    st.write("---")

    # ── auto detect then show canvas ──────────────────────────────────────────
    if st.button("🚀 วิเคราะห์รูปภาพ"):
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        if sensitivity != "ปกติ":
            lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
            l,a,b = cv2.split(lab)
            clahe = cv2.createCLAHE(
                clipLimit=2.0 if sensitivity=="สูง" else 3.5,
                tileGridSize=(8,8))
            img_cv = cv2.cvtColor(cv2.merge([clahe.apply(l),a,b]), cv2.COLOR_LAB2BGR)

        detected_y, method = find_water_surface(img_cv)

        st.session_state.img_cv      = img_cv
        st.session_state.detected_y  = detected_y
        st.session_state.method      = method
        st.session_state.h           = img_cv.shape[0]
        st.session_state.w           = img_cv.shape[1]
        # default fracs
        st.session_state.top_frac    = 0.05
        st.session_state.bottom_frac = 0.95
        st.session_state.water_frac  = (detected_y / img_cv.shape[0]
                                        if detected_y is not None else 0.50)
        st.session_state.ready       = True

    # ── canvas section ────────────────────────────────────────────────────────
    if st.session_state.get("ready"):
        img_cv     = st.session_state.img_cv
        detected_y = st.session_state.detected_y
        method     = st.session_state.method
        h          = st.session_state.h
        w          = st.session_state.w

        if detected_y is not None:
            st.success(f"✅ AI ตรวจพบผิวน้ำ (วิธี: {method}) — ปรับเส้นให้แม่นยำด้านล่าง")
        else:
            st.warning("⚠️ AI หาผิวน้ำไม่เจอ — ลากเส้นสีเขียวให้ตรงผิวน้ำด้วยตนเอง")

        st.markdown("""
**วิธีใช้:**
- 🔴 **เส้นแดง** = ขีด 900 ml → ลากให้ตรงกับขีดบนสุดของขวด
- 🔵 **เส้นน้ำเงิน** = ขีด 100 ml → ลากให้ตรงกับขีดล่างสุดของขวด  
- 🟢 **เส้นเขียว** = ผิวน้ำ → ลากให้ตรงกับระดับน้ำ
- กด **✅ ยืนยัน** แล้วกด **แสดงผลสุดท้าย** ด้านล่าง
""")

        # render full image (no crop needed — lines define the scale)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_b64 = img_to_b64(img_rgb)

        three_line_canvas(
            img_b64,
            top_frac   = st.session_state.top_frac,
            water_frac = st.session_state.water_frac,
            bottom_frac= st.session_state.bottom_frac,
        )

        st.write("---")
        st.markdown("### 📊 แสดงผลสุดท้าย")
        st.caption("กรอกค่า % จากด้านบน หลังจากปรับเส้นในภาพแล้ว")

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            top_pct   = st.number_input("🔴 900 ml (%)",  0, 98,
                                         int(st.session_state.top_frac*100),   key="f_top")
        with fc2:
            water_pct = st.number_input("🟢 ผิวน้ำ (%)", 1, 99,
                                         int(st.session_state.water_frac*100), key="f_water")
        with fc3:
            bot_pct   = st.number_input("🔵 100 ml (%)", 2,100,
                                         int(st.session_state.bottom_frac*100),key="f_bot")

        if st.button("📐 แสดงผลสุดท้าย"):
            top_y   = int(top_pct   / 100 * h)
            water_y = int(water_pct / 100 * h)
            bot_y   = int(bot_pct   / 100 * h)
            volume  = calc_volume(water_y, top_y, bot_y)

            # draw final output
            output = img_cv.copy()

            # filled zone: top → water (light green tint)
            overlay = output.copy()
            cv2.rectangle(overlay, (0, top_y), (w, water_y), (0,180,80), -1)
            cv2.addWeighted(overlay, 0.15, output, 0.85, 0, output)

            # red line — 900 ml
            cv2.line(output, (0,top_y),   (w,top_y),   (0,0,255), 4)
            cv2.putText(output, "900 ml (Top)", (10, top_y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            # blue line — 100 ml
            cv2.line(output, (0,bot_y),   (w,bot_y),   (255,100,0), 4)
            cv2.putText(output, "100 ml (Bottom)", (10, bot_y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,100,0), 2)
            # green line — water surface
            cv2.line(output, (0,water_y), (w,water_y), (0,200,0), 5)
            cv2.putText(output, f"Water: {volume} ml", (10, water_y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,0), 2)

            st.markdown(
                f"<h1 style='text-align:center;color:#00ff88;'>{volume} ml</h1>",
                unsafe_allow_html=True)
            st.image(output, channels="BGR",
                     caption="🔴 900 ml  |  🟢 ผิวน้ำ  |  🔵 100 ml",
                     use_container_width=True)
