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
        if exif is None:
            return img
        orient_key = next(k for k, v in ExifTags.TAGS.items() if v == "Orientation")
        orientation = exif.get(orient_key)
        for tag, deg in {3: 180, 6: 270, 8: 90}.items():
            if orientation == tag:
                return img.rotate(deg, expand=True)
    except:
        pass
    return img


def find_water_surface(img_cv):
    h, w  = img_cv.shape[:2]
    gray  = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blur, 30, 100)

    for thresh, gap, min_len, tol in [(40, 20, w // 3, 10), (20, 40, w // 6, 15)]:
        lines = cv2.HoughLinesP(edged, 1, np.pi / 180, thresh,
                                minLineLength=min_len, maxLineGap=gap)
        if lines is not None:
            valid = [(l, l[0][1]) for l in lines
                     if abs(l[0][1] - l[0][3]) < tol and l[0][1] > h * 0.2]
            if valid:
                _, y = sorted(valid, key=lambda x: x[1], reverse=True)[0]
                return y

    sobelx     = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
    row_energy = np.sum(np.abs(sobelx), axis=1)
    s, e       = int(h * 0.20), int(h * 0.95)
    roi        = row_energy[s:e]
    if roi.max() > 0:
        cands = np.where(roi > roi.max() * 0.35)[0]
        if len(cands):
            return int(cands[-1]) + s

    gray_c   = gray[int(h * 0.1):int(h * 0.9), :]
    smoothed = np.convolve(np.mean(gray_c, axis=1), np.ones(7) / 7, mode='same')
    diff     = np.diff(smoothed)
    jumps    = np.where(np.abs(diff) > np.std(diff) * 1.2)[0]
    if len(jumps):
        return int(jumps[-1]) + int(h * 0.1)

    return None


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
  padding-bottom:14px;
}}
#hint {{ font-size:0.82em; color:#aaa; margin:8px 4px 4px; text-align:center; }}
#wrap {{ position:relative; touch-action:none; }}
canvas {{ display:block; }}
#legend {{
  display:flex; gap:14px; justify-content:center;
  margin:8px 0 4px; font-size:0.85em;
}}
.dot {{ width:12px; height:12px; border-radius:50%; display:inline-block; margin-right:4px; vertical-align:middle; }}
#vol {{ font-size:1.6em; font-weight:bold; color:#00ff88; margin:6px 0 4px; text-align:center; }}

#locks {{
  display:flex; gap:10px; justify-content:center; margin:6px 0;
  flex-wrap:wrap;
}}
.lock-btn {{
  padding:5px 14px; border-radius:20px; border:2px solid #555;
  background:#222; color:#bbb; cursor:pointer; font-size:0.82em;
  transition: all 0.15s;
}}
.lock-btn.locked {{
  background:#333; border-color:#ffcc00; color:#ffcc00;
}}

#confirmBtn {{
  margin:10px 8px 4px; padding:10px 32px; background:#00f2fe; color:#111;
  border:none; border-radius:8px; font-size:1em; font-weight:bold; cursor:pointer;
}}
#saved {{ color:#0f0; font-size:0.85em; min-height:1.2em; text-align:center; margin-top:4px; }}
</style>
</head>
<body>
<div id="hint">👆 ลากเส้นเพื่อปรับตำแหน่ง · กด 🔒 เพื่อล็อคเส้น</div>
<div id="legend">
  <span><span class="dot" style="background:#ff3333"></span>900 ml (บน)</span>
  <span><span class="dot" style="background:#00cc00"></span>ผิวน้ำ</span>
  <span><span class="dot" style="background:#3399ff"></span>100 ml (ล่าง)</span>
</div>
<div id="wrap"><canvas id="c"></canvas></div>
<div id="vol">-- ml</div>

<div id="locks">
  <button class="lock-btn" id="lockTop"   onclick="toggleLock('top')"   >🔓 บน (900 ml)</button>
  <button class="lock-btn" id="lockWater" onclick="toggleLock('water')" >🔓 ผิวน้ำ</button>
  <button class="lock-btn" id="lockBot"   onclick="toggleLock('bottom')">🔓 ล่าง (100 ml)</button>
</div>

<button id="confirmBtn" onclick="doConfirm()">✅ ยืนยันตำแหน่ง</button>
<div id="saved"></div>

<script>
const IMG_B64  = "{img_b64}";
const CANVAS_W = {canvas_w};
let topFrac    = {top_frac};
let waterFrac  = {water_frac};
let bottomFrac = {bottom_frac};

const locked = {{ top: false, water: false, bottom: false }};

const canvas = document.getElementById('c');
const ctx    = canvas.getContext('2d');
const img    = new Image();
let imgH     = 0;
let dragging = null;

img.onload = () => {{
  const scale   = CANVAS_W / img.naturalWidth;
  canvas.width  = CANVAS_W;
  canvas.height = imgH = Math.round(img.naturalHeight * scale);
  draw();
}};
img.src = 'data:image/jpeg;base64,' + IMG_B64;

function fracToY(f) {{ return f * imgH; }}
function yToFrac(y) {{ return Math.max(0.01, Math.min(0.99, y / imgH)); }}
function clampY(y)  {{ return Math.max(0, Math.min(imgH, y)); }}

function calcVol() {{
  const tY = fracToY(topFrac), bY = fracToY(bottomFrac), wY = fracToY(waterFrac);
  if (bY === tY) return 0;
  return Math.round(Math.max(50, Math.min(950, 100 + (bY - wY) / (bY - tY) * 800)));
}}

function drawLine(y, color, label, dashed, isLocked) {{
  ctx.save();
  ctx.globalAlpha = isLocked ? 0.5 : 1.0;
  ctx.strokeStyle = color; ctx.lineWidth = 3.5;
  ctx.setLineDash(dashed ? [12, 6] : []);
  ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
  ctx.restore();

  // handle circle
  ctx.save();
  ctx.globalAlpha = isLocked ? 0.65 : 1.0;
  ctx.fillStyle   = isLocked ? '#555' : color;
  ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.arc(canvas.width / 2, y, 12, 0, Math.PI * 2);
  ctx.fill(); ctx.stroke();
  // icon inside circle
  ctx.fillStyle = '#fff'; ctx.font = '12px sans-serif';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(isLocked ? '🔒' : '↕', canvas.width / 2, y);
  ctx.restore();

  // label
  ctx.save();
  ctx.globalAlpha = isLocked ? 0.55 : 1.0;
  ctx.fillStyle   = color; ctx.font = 'bold 13px sans-serif';
  ctx.textAlign   = 'left'; ctx.textBaseline = 'alphabetic';
  ctx.fillText(label + (isLocked ? '  🔒' : ''), 6, y - 6);
  ctx.restore();
}}

function draw() {{
  ctx.clearRect(0, 0, canvas.width, imgH);
  ctx.drawImage(img, 0, 0, canvas.width, imgH);
  const tY = fracToY(topFrac), bY = fracToY(bottomFrac), wY = fracToY(waterFrac);
  ctx.save();
  ctx.fillStyle = 'rgba(0,200,100,0.08)';
  ctx.fillRect(0, tY, canvas.width, wY - tY);
  ctx.restore();
  drawLine(tY, '#ff3333', '🔴 900 ml', false, locked.top);
  drawLine(bY, '#3399ff', '🔵 100 ml', false, locked.bottom);
  drawLine(wY, '#00cc00', '🟢 ผิวน้ำ',  true,  locked.water);
  document.getElementById('vol').textContent = calcVol() + ' ml';
}}

function toggleLock(line) {{
  locked[line] = !locked[line];
  const ids    = {{ top:'lockTop', water:'lockWater', bottom:'lockBot' }};
  const labels = {{ top:'บน (900 ml)', water:'ผิวน้ำ', bottom:'ล่าง (100 ml)' }};
  const btn = document.getElementById(ids[line]);
  if (locked[line]) {{
    btn.classList.add('locked');
    btn.textContent = '🔒 ' + labels[line];
  }} else {{
    btn.classList.remove('locked');
    btn.textContent = '🔓 ' + labels[line];
  }}
  draw();
}}

function getY(e) {{
  const r = canvas.getBoundingClientRect();
  return (e.touches ? e.touches[0].clientY : e.clientY) - r.top;
}}

function onDown(e) {{
  e.preventDefault();
  const y = getY(e);
  const candidates = [
    ['top',    Math.abs(y - fracToY(topFrac)),    locked.top],
    ['water',  Math.abs(y - fracToY(waterFrac)),  locked.water],
    ['bottom', Math.abs(y - fracToY(bottomFrac)), locked.bottom],
  ].filter(c => !c[2]).sort((a, b) => a[1] - b[1]);
  dragging = candidates.length ? candidates[0][0] : null;
  if (dragging) onMove(e);
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

function doConfirm() {{
  const vol = calcVol();
  window.parent.postMessage({{
    type: 'streamlit:setComponentValue',
    value: {{ top: topFrac, water: waterFrac, bottom: bottomFrac, vol: vol }}
  }}, '*');
  document.getElementById('saved').textContent = '✅ ยืนยันแล้ว — ' + vol + ' ml';
}}
</script>
</body>
</html>
"""
    return components.html(html, height=800, scrolling=False)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

img_file = st.file_uploader("เลือกรูปขวดน้ำเกลือ", type=["jpg", "jpeg", "png"])

if img_file:
    pil_img = fix_exif_rotation(Image.open(img_file).convert("RGB"))
    img_np  = np.array(pil_img)

    # Auto-rotate if landscape
    if img_np.shape[1] > img_np.shape[0]:
        img_np = np.rot90(img_np, k=1)
        st.toast("📱 หมุนอัตโนมัติแล้ว", icon="🔄")

    st.markdown("### 🔄 ปรับการหมุน")
    extra_rot = st.radio(
        "หมุนเพิ่มเติม", [0, 90, 180, 270],
        format_func=lambda x: {0: "ปกติ", 90: "90° CW", 180: "180°", 270: "270° CW"}[x],
        horizontal=True, index=0
    )
    if extra_rot:
        img_np = np.rot90(img_np, k=-(extra_rot // 90) % 4)

    st.write("---")

    # ── auto-analyze on upload (no button) ───────────────────────────────────
    img_cv     = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    detected_y = find_water_surface(img_cv)
    h, w       = img_cv.shape[:2]

    # Reset fracs only when a new file is loaded
    file_key = img_file.name + str(img_file.size)
    if st.session_state.get("_file_key") != file_key:
        st.session_state._file_key   = file_key
        st.session_state.top_frac    = 0.05
        st.session_state.bottom_frac = 0.95
        st.session_state.water_frac  = detected_y / h if detected_y is not None else 0.50

    if detected_y is not None:
        st.success("✅ AI ตรวจพบผิวน้ำ — ปรับเส้นให้แม่นยำด้านล่าง")
    else:
        st.warning("⚠️ AI หาผิวน้ำไม่เจอ — ลากเส้นสีเขียวให้ตรงผิวน้ำด้วยตนเอง")

    st.markdown("""
**วิธีใช้:**
- 🔴 **เส้นแดง** = ขีด 900 ml → ลากให้ตรงกับขีดบนสุดของขวด
- 🔵 **เส้นน้ำเงิน** = ขีด 100 ml → ลากให้ตรงกับขีดล่างสุดของขวด
- 🟢 **เส้นเขียว** = ผิวน้ำ → ลากให้ตรงกับระดับน้ำ
- กด **🔒** เพื่อล็อคเส้นที่ตั้งไว้แล้ว ป้องกันการเลื่อนพลาด
- กด **✅ ยืนยันตำแหน่ง** เพื่อบันทึกผล
""")

    # Encode image for canvas
    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)).save(buf, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    result = three_line_canvas(
        img_b64,
        top_frac    = st.session_state.top_frac,
        water_frac  = st.session_state.water_frac,
        bottom_frac = st.session_state.bottom_frac,
    )

    # Show final result after user confirms inside canvas
    if result and isinstance(result, dict) and "vol" in result:
        vol     = result["vol"]
        top_y   = int(result["top"]    * h)
        water_y = int(result["water"]  * h)
        bot_y   = int(result["bottom"] * h)

        st.markdown(
            f"<h1 style='text-align:center;color:#00ff88;'>{vol} ml</h1>",
            unsafe_allow_html=True)

        output  = img_cv.copy()
        overlay = output.copy()
        cv2.rectangle(overlay, (0, top_y), (w, water_y), (0, 180, 80), -1)
        cv2.addWeighted(overlay, 0.15, output, 0.85, 0, output)

        cv2.line(output, (0, top_y),   (w, top_y),   (0, 0, 255),   4)
        cv2.putText(output, "900 ml", (10, top_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.line(output, (0, bot_y),   (w, bot_y),   (255, 100, 0), 4)
        cv2.putText(output, "100 ml", (10, bot_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

        cv2.line(output, (0, water_y), (w, water_y), (0, 200, 0),   5)
        cv2.putText(output, f"Water: {vol} ml", (10, water_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

        st.image(output, channels="BGR",
                 caption="🔴 900 ml  |  🟢 ผิวน้ำ  |  🔵 100 ml",
                 use_container_width=True)
