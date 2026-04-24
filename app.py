# ═══════════════════════════════════════════════════════════════
#  WEED ROVER COMMAND CENTER — Flask Web App
#  Run: python app.py
#  Then open http://127.0.0.1:5000 in your browser
# ═══════════════════════════════════════════════════════════════

import os, cv2, csv, time, json, threading, base64
import numpy as np
from datetime import datetime
from flask import (Flask, render_template, request,
                   jsonify, send_from_directory, Response)
from werkzeug.utils import secure_filename

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER  = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER  = os.path.join(BASE_DIR, "outputs")
MODEL_PATH     = os.path.join(BASE_DIR, "MY_WEED_ROVER_BRAIN.pt")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ── Detection config ───────────────────────────────────────────
CONF_THRESHOLD         = 0.30
MAX_WEED_AREA_FRACTION = 0.15
MIN_CONF_TO_TARGET     = 0.55
SIZE_RATIO_THRESHOLD   = 2.5
CLASS_CROP             = 0
CLASS_WEED             = 1

COLOUR_CROP      = (40,  200,  40)
COLOUR_WEED      = (0,    60, 220)
COLOUR_RECLASSED = (0,   165, 255)
COLOUR_TARGET    = (0,   255, 255)

IMAGE_EXTS = {'.jpg','.jpeg','.png','.bmp','.tiff','.tif','.webp'}
VIDEO_EXTS = {'.mp4','.avi','.mov','.mkv','.webm'}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024   # 500 MB

# ── Progress store (per-job) ───────────────────────────────────
_progress = {}   # job_id → dict

# ══════════════════════════════════════════════════════════════
#  DETECTION HELPERS
# ══════════════════════════════════════════════════════════════

def select_target(boxes, class_ids, confidences, w, h):
    img_area = w * h
    target_idx = None
    min_area   = float('inf')
    for i, cls in enumerate(class_ids):
        if cls != CLASS_WEED:            continue
        if confidences[i] < MIN_CONF_TO_TARGET: continue
        x1,y1,x2,y2 = boxes[i]
        area = (x2-x1)*(y2-y1)
        if area / img_area > MAX_WEED_AREA_FRACTION: continue
        if area < min_area:
            min_area   = area
            target_idx = i
    return target_idx


def apply_size_check(boxes, class_ids, w, h):
    updated   = list(class_ids)
    reclassed = set()
    img_area  = w * h
    crop_areas = [(boxes[i][2]-boxes[i][0])*(boxes[i][3]-boxes[i][1])
                  for i,c in enumerate(class_ids) if c == CLASS_CROP]
    for i,cls in enumerate(class_ids):
        if cls != CLASS_WEED: continue
        wa = (boxes[i][2]-boxes[i][0])*(boxes[i][3]-boxes[i][1])
        if crop_areas:
            if wa > np.mean(crop_areas)*SIZE_RATIO_THRESHOLD:
                updated[i] = CLASS_CROP; reclassed.add(i)
        else:
            if wa/img_area > MAX_WEED_AREA_FRACTION:
                updated[i] = CLASS_CROP; reclassed.add(i)
    return updated, reclassed


def pixel_to_pantilt(cx, cy, w, h):
    pan  = np.interp(cx, [0, w], [0,   180])
    tilt = np.interp(cy, [0, h], [30,  150])
    return round(float(pan),1), round(float(tilt),1)


def draw_frame(frame, boxes, class_ids, confidences,
               reclassed, target_idx, pan, tilt):
    out  = frame.copy()
    fh, fw = frame.shape[:2]
    for i,(box,cls,conf) in enumerate(zip(boxes,class_ids,confidences)):
        x1,y1,x2,y2 = map(int,box)
        cx = (x1+x2)//2; cy = (y1+y2)//2
        if i in reclassed:
            colour = COLOUR_RECLASSED; label = f"CROP* {conf:.0%}"
        elif cls == CLASS_WEED:
            if i == target_idx:
                colour = COLOUR_TARGET
                label  = f"TARGET {conf:.0%} Pan={pan} Tilt={tilt}"
            else:
                colour = COLOUR_WEED; label = f"WEED {conf:.0%}"
        else:
            colour = COLOUR_CROP; label = f"CROP {conf:.0%}"
        thickness = 3 if i == target_idx else 2
        cv2.rectangle(out,(x1,y1),(x2,y2),colour,thickness)
        if i == target_idx:
            cl=18
            for (px,py),(dx,dy) in [((x1,y1),(1,1)),((x2,y1),(-1,1)),
                                     ((x1,y2),(1,-1)),((x2,y2),(-1,-1))]:
                cv2.line(out,(px,py),(px+dx*cl,py),colour,3)
                cv2.line(out,(px,py),(px,py+dy*cl),colour,3)
            cv2.arrowedLine(out,(fw//2,fh//2),(cx,cy),COLOUR_TARGET,2,tipLength=0.03)
        cv2.drawMarker(out,(cx,cy),colour,cv2.MARKER_CROSS,14,2)
        (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.52,1)
        lx=max(x1,0); ly=max(y1-th-10,0)
        cv2.rectangle(out,(lx,ly),(lx+tw+8,ly+th+8),colour,-1)
        cv2.putText(out,label,(lx+4,ly+th+2),
                    cv2.FONT_HERSHEY_SIMPLEX,0.52,(255,255,255),1,cv2.LINE_AA)
        area_pct = (x2-x1)*(y2-y1)/(fw*fh)*100
        cv2.putText(out,f"{area_pct:.1f}% frame",(x1,y2+14),
                    cv2.FONT_HERSHEY_SIMPLEX,0.40,colour,1,cv2.LINE_AA)
    # HUD
    total_weeds = sum(1 for c in class_ids if c==CLASS_WEED)
    total_crops = sum(1 for c in class_ids if c==CLASS_CROP)
    hud = [f"Weeds   : {total_weeds}",f"Crops   : {total_crops}",
           f"Fixed   : {len(reclassed)}",f"Conf    : {CONF_THRESHOLD}"]
    if target_idx is not None:
        hud.append(f"Pan={pan}  Tilt={tilt}")
        hud.append("LASER LOCKED")
    panel_h = len(hud)*26+16
    overlay = out.copy()
    cv2.rectangle(overlay,(0,0),(230,panel_h),(0,0,0),-1)
    cv2.addWeighted(overlay,0.60,out,0.40,0,out)
    for j,line in enumerate(hud):
        col = (0,255,255) if line=="LASER LOCKED" else (255,255,255)
        cv2.putText(out,line,(6,22+j*26),cv2.FONT_HERSHEY_SIMPLEX,0.54,col,1,cv2.LINE_AA)
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(out,ts,(fw-240,fh-10),cv2.FONT_HERSHEY_SIMPLEX,0.44,(160,160,160),1,cv2.LINE_AA)
    return out


def run_inference_image(model, img_path):
    """Run inference on one image. Returns annotated image + stats dict."""
    image = cv2.imread(img_path)
    if image is None:
        return None, {}
    h,w = image.shape[:2]
    results = model(image, conf=CONF_THRESHOLD, verbose=False)[0]
    stats = {"weeds":0,"crops":0,"fixed":0,"target":False,
             "pan":90.0,"tilt":90.0,"target_cx":None,"target_cy":None,
             "target_conf":None,"detections":[]}
    pan,tilt = 90.0,90.0
    if results.boxes is not None and len(results.boxes)>0:
        boxes       = results.boxes.xyxy.cpu().numpy()
        class_ids   = results.boxes.cls.cpu().numpy().astype(int)
        confidences = results.boxes.conf.cpu().numpy()
        updated_ids, reclassed = apply_size_check(boxes,class_ids,w,h)
        target_idx = select_target(boxes,updated_ids,confidences,w,h)
        if target_idx is not None:
            x1,y1,x2,y2 = boxes[target_idx]
            cx=(x1+x2)/2; cy=(y1+y2)/2
            pan,tilt = pixel_to_pantilt(cx,cy,w,h)
            stats["target_cx"] = round(float(cx),1)
            stats["target_cy"] = round(float(cy),1)
            stats["target_conf"] = float(confidences[target_idx])
        annotated = draw_frame(image,boxes,updated_ids,confidences,
                               reclassed,target_idx,pan,tilt)
        stats["weeds"]  = sum(1 for c in updated_ids if c==CLASS_WEED)
        stats["crops"]  = sum(1 for c in updated_ids if c==CLASS_CROP)
        stats["fixed"]  = len(reclassed)
        stats["target"] = target_idx is not None
        stats["pan"]    = pan
        stats["tilt"]   = tilt
        for i,(box,cls,conf) in enumerate(zip(boxes,updated_ids,confidences)):
            x1,y1,x2,y2 = map(int,box)
            cls_name = model.names.get(cls,str(cls))
            if i in reclassed: cls_name+="*"
            stats["detections"].append({
                "id":i,"class":cls_name,"conf":round(float(conf),4),
                "box":[x1,y1,x2,y2],"is_target":(i==target_idx)
            })
    else:
        annotated = image
    return annotated, stats


def run_inference_video(model, vid_path, out_path, job_id):
    """Process video file, save annotated copy, return aggregate stats."""
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        _progress[job_id]["error"] = "Cannot open video"
        return {}
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    writer  = cv2.VideoWriter(out_path,fourcc,fps_src,(width,height))
    frame_n=0; weed_frames=0; total_weeds=0; total_crops=0
    last_stats = {"pan":90.0,"tilt":90.0,"target_cx":None,"target_cy":None,"target_conf":None}
    while True:
        ret,frame = cap.read()
        if not ret: break
        frame_n+=1
        _progress[job_id]["pct"] = int(frame_n/total_f*100)
        results = model(frame,conf=CONF_THRESHOLD,verbose=False)[0]
        pan,tilt=90.0,90.0
        if results.boxes is not None and len(results.boxes)>0:
            boxes       = results.boxes.xyxy.cpu().numpy()
            class_ids   = results.boxes.cls.cpu().numpy().astype(int)
            confidences = results.boxes.conf.cpu().numpy()
            updated_ids,reclassed = apply_size_check(boxes,class_ids,width,height)
            target_idx = select_target(boxes,updated_ids,confidences,width,height)
            if target_idx is not None:
                x1,y1,x2,y2=boxes[target_idx]
                cx=(x1+x2)/2; cy=(y1+y2)/2
                pan,tilt=pixel_to_pantilt(cx,cy,width,height)
                last_stats["pan"]=pan; last_stats["tilt"]=tilt
                last_stats["target_cx"]=round(float(cx),1)
                last_stats["target_cy"]=round(float(cy),1)
                last_stats["target_conf"]=float(confidences[target_idx])
                weed_frames+=1
            total_weeds += sum(1 for c in updated_ids if c==CLASS_WEED)
            total_crops += sum(1 for c in updated_ids if c==CLASS_CROP)
            annotated = draw_frame(frame,boxes,updated_ids,confidences,
                                   reclassed,target_idx,pan,tilt)
        else:
            annotated=frame.copy()
        writer.write(annotated)
    cap.release(); writer.release()
    _progress[job_id]["pct"] = 100
    return {
        "frames":frame_n,"weed_frames":weed_frames,
        "weeds":total_weeds,"crops":total_crops,
        "target":weed_frames>0,
        "weed_frame_rate": round(weed_frames/max(frame_n,1)*100,1),
        **last_stats
    }

# ══════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error":"No file part"}),400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error":"Empty filename"}),400
    fname   = secure_filename(f.filename)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname   = f"{ts}_{fname}"
    save_to = os.path.join(UPLOAD_FOLDER, fname)
    f.save(save_to)
    ext = os.path.splitext(fname)[1].lower()
    ftype = "image" if ext in IMAGE_EXTS else "video" if ext in VIDEO_EXTS else "unknown"
    return jsonify({"filename":fname,"type":ftype,"path":save_to})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data     = request.json or {}
    filename = data.get("filename")
    if not filename:
        return jsonify({"error":"No filename provided"}),400

    src_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(src_path):
        return jsonify({"error":"File not found"}),404

    # Check model
    model_path = data.get("model_path", MODEL_PATH)
    if not os.path.exists(model_path):
        return jsonify({"error":f"Model not found: {model_path}. Place best.pt next to app.py or specify path."}),400

    model = YOLO(model_path)
    ext   = os.path.splitext(filename)[1].lower()
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    name  = os.path.splitext(filename)[0]

    if ext in IMAGE_EXTS:
        out_filename = f"{name}_result{ext}"
        out_path     = os.path.join(OUTPUT_FOLDER, out_filename)
        annotated, stats = run_inference_image(model, src_path)
        if annotated is None:
            return jsonify({"error":"Could not read image"}),500
        cv2.imwrite(out_path, annotated)

        # Save CSV
        csv_name = f"{name}_results_{ts}.csv"
        csv_path = os.path.join(OUTPUT_FOLDER, csv_name)
        with open(csv_path,"w",newline="",encoding="utf-8") as cf:
            w = csv.writer(cf)
            w.writerow(["Class","Confidence","X1","Y1","X2","Y2","Is_Target","Pan","Tilt"])
            for d in stats.get("detections",[]):
                x1,y1,x2,y2=d["box"]
                w.writerow([d["class"],d["conf"],x1,y1,x2,y2,
                             d["is_target"],
                             stats["pan"] if d["is_target"] else "",
                             stats["tilt"] if d["is_target"] else ""])

        # Encode annotated image as base64 for preview
        _,buf = cv2.imencode(".jpg", annotated,[cv2.IMWRITE_JPEG_QUALITY,85])
        b64   = base64.b64encode(buf).decode()

        return jsonify({
            "type":"image",
            "stats":stats,
            "output_file": out_filename,
            "csv_file":    csv_name,
            "preview_b64": b64,
            "output_folder": OUTPUT_FOLDER
        })

    elif ext in VIDEO_EXTS:
        # Run video inference in background, poll for progress
        job_id       = f"job_{ts}"
        out_filename = f"{name}_result.mp4"
        out_path     = os.path.join(OUTPUT_FOLDER, out_filename)
        csv_name     = f"{name}_results_{ts}.csv"
        _progress[job_id] = {"pct":0,"done":False,"stats":{},"error":None,
                              "out_filename":out_filename,"csv_name":csv_name}

        def _worker():
            try:
                stats = run_inference_video(model, src_path, out_path, job_id)
                # Write CSV
                csv_path2 = os.path.join(OUTPUT_FOLDER, csv_name)
                with open(csv_path2,"w",newline="",encoding="utf-8") as cf:
                    cw = csv.writer(cf)
                    cw.writerow(["Metric","Value"])
                    for k,v in stats.items():
                        cw.writerow([k,v])
                _progress[job_id]["stats"] = stats
            except Exception as e:
                _progress[job_id]["error"] = str(e)
            finally:
                _progress[job_id]["done"] = True

        threading.Thread(target=_worker, daemon=True).start()
        return jsonify({"type":"video","job_id":job_id})

    else:
        return jsonify({"error":"Unsupported file type"}),400


@app.route("/api/progress/<job_id>")
def progress(job_id):
    p = _progress.get(job_id)
    if not p:
        return jsonify({"error":"Unknown job"}),404
    return jsonify(p)


@app.route("/api/model_check", methods=["POST"])
def model_check():
    data = request.json or {}
    path = data.get("path", MODEL_PATH)
    return jsonify({"exists": os.path.exists(path), "path": path})


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  🌿 WEED ROVER COMMAND CENTER")
    print("="*55)
    if not YOLO_AVAILABLE:
        print("  ⚠  ultralytics not installed!")
        print("     pip install ultralytics")
    if not os.path.exists(MODEL_PATH):
        print(f"  ⚠  Model not found: {MODEL_PATH}")
        print("     Place your best.pt next to app.py")
    else:
        print(f"  ✓  Model found: {MODEL_PATH}")
    print(f"  ✓  Uploads  : {UPLOAD_FOLDER}")
    print(f"  ✓  Outputs  : {OUTPUT_FOLDER}")
    print(f"\n  Open browser → http://127.0.0.1:5000")
    print("="*55 + "\n")
    app.run(debug=False, host="127.0.0.1", port=5000)
