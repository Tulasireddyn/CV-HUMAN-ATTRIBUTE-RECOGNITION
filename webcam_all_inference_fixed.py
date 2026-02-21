# webcam_all_inference_fixed.py
import os, json, cv2, torch, tempfile, numpy as np
import torchvision.transforms as T
from PIL import Image

from models.model import AgeGenderModel
from models.emotion_model import EmotionModel
from models.clothing_model import ClothingModel
from models.pose_model import PoseModel

# --------------------
# 0) Defaults (used only if we can't read mapping from ckpt or labels file)
# --------------------
DEFAULT_GENDER = ["Male", "Female"]
DEFAULT_EMOTION = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]
DEFAULT_POSE = [
    "sleeping","drinking","running","eating","sitting",
    "clapping","fighting","laughing","calling",
    "listening_to_music","using_laptop","texting","dancing"
]
DEFAULT_CLOTHING = ["T-shirt","Shirt","Sweater","Tank top","Jackets_Vests"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# 1) Helpers: label mapping
# --------------------
def _try_load_labels_from_file(path):
    if not os.path.exists(path):
        return None
    try:
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            # support {"idx_to_class": {...}} or {"classes": [...]}
            if isinstance(data, dict):
                if "classes" in data and isinstance(data["classes"], list):
                    return data["classes"]
                if "idx_to_class" in data and isinstance(data["idx_to_class"], dict):
                    # convert {idx: name}
                    m = data["idx_to_class"]
                    # keys might be str
                    items = sorted(((int(k), v) for k, v in m.items()), key=lambda x: x[0])
                    return [v for _, v in items]
                if "class_to_idx" in data and isinstance(data["class_to_idx"], dict):
                    m = data["class_to_idx"]
                    items = sorted(((int(v), k) for k, v in m.items()), key=lambda x: x[0])
                    return [k for _, k in items]
        else:
            # labels.txt: one class per line
            with open(path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            return lines if lines else None
    except Exception:
        return None

def _try_load_labels_from_checkpoint(ckpt_path):
    try:
        sd = torch.load(ckpt_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            meta = {k: v for k, v in sd.items() if k != "state_dict"}
        else:
            meta = sd if isinstance(sd, dict) else {}
        # Try common fields
        for k in ("idx_to_class","classes","class_names","labels"):
            if k in meta:
                v = meta[k]
                if isinstance(v, list):
                    return v
                if isinstance(v, dict):
                    # idx_to_class dict
                    try:
                        items = sorted(((int(i), name) for i, name in v.items()), key=lambda x: x[0])
                        return [name for _, name in items]
                    except Exception:
                        pass
        if "class_to_idx" in meta and isinstance(meta["class_to_idx"], dict):
            m = meta["class_to_idx"]
            items = sorted(((int(idx), cls) for cls, idx in m.items()), key=lambda x: x[0])
            return [cls for idx, cls in items]
    except Exception:
        pass
    return None

def resolve_labels(task_name, ckpt_path, defaults, labels_file_candidates):
    # 1) try checkpoint metadata
    labels = _try_load_labels_from_checkpoint(ckpt_path)
    # 2) try sidecar files
    if labels is None:
        for cand in labels_file_candidates:
            labels = _try_load_labels_from_file(cand)
            if labels:
                break
    # 3) fallback
    if labels is None:
        labels = defaults
    print(f"[{task_name}] labels:", labels)
    return labels

# --------------------
# 2) Load models safely
# --------------------
def load_state_dict_safely(model, ckpt_path, map_location):
    sd = torch.load(ckpt_path, map_location=map_location)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    new_sd = {}
    for k, v in sd.items():
        new_sd[k.replace("module.", "") if k.startswith("module.") else k] = v
    model.load_state_dict(new_sd, strict=False)
    return model

# Age/Gender + Emotion (unchanged mapping)
age_gender_model = AgeGenderModel()
load_state_dict_safely(age_gender_model, "checkpoints/age_gender_model.pth", device)
age_gender_model.to(device).eval()

emotion_model = EmotionModel()
load_state_dict_safely(emotion_model, "checkpoints/emotion_model.pth", device)
emotion_model.to(device).eval()

# Resolve clothing & pose labels from ckpt sidecar if possible
CLOTHING_LABELS = resolve_labels(
    "clothing",
    "checkpoints/best_clothing_model.pth",
    DEFAULT_CLOTHING,
    labels_file_candidates=[
        "checkpoints/best_clothing_model.labels.json",
        "checkpoints/best_clothing_model.labels.txt",
        "config/clothing_labels.json",
        "config/clothing_labels.txt"
    ],
)
POSE_LABELS = resolve_labels(
    "pose",
    "checkpoints/best_pose_model.pth",
    DEFAULT_POSE,
    labels_file_candidates=[
        "checkpoints/best_pose_model.labels.json",
        "checkpoints/best_pose_model.labels.txt",
        "config/pose_labels.json",
        "config/pose_labels.txt"
    ],
)

clothing_model = ClothingModel(num_classes=len(CLOTHING_LABELS), pretrained=False)
load_state_dict_safely(clothing_model, "checkpoints/best_clothing_model.pth", device)
clothing_model.to(device).eval()

pose_model = PoseModel(num_classes=len(POSE_LABELS), pretrained=False)
load_state_dict_safely(pose_model, "checkpoints/best_pose_model.pth", device)
pose_model.to(device).eval()

# --------------------
# 3) Preprocess
# --------------------
tf224 = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# --------------------
# 4) Detectors + display
# --------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

GUI_AVAILABLE = True
try:
    test = np.zeros((2,2,3), np.uint8)
    cv2.imshow("__test__", test); cv2.waitKey(1); cv2.destroyWindow("__test__")
except Exception:
    GUI_AVAILABLE = False
_fallback_path = os.path.join(tempfile.gettempdir(), "cv_har_last_frame.jpg")
_opened_once = False

def show_frame(name, frame):
    global _opened_once
    if GUI_AVAILABLE:
        try: cv2.imshow(name, frame); return
        except Exception: pass
    cv2.imwrite(_fallback_path, frame)
    if not _opened_once:
        try:
            if os.name == "nt": os.startfile(_fallback_path)
            else:
                import subprocess; subprocess.Popen(["xdg-open", _fallback_path])
        except Exception: pass
        _opened_once = True

# --------------------
# 5) Inference helpers (with confidence + smoothing)
# --------------------
from collections import deque

def softmax_np(x):
    e = np.exp(x - np.max(x))
    return e / (np.sum(e) + 1e-9)

# temporal buffers
pose_buf = deque(maxlen=5)
cloth_buf = deque(maxlen=5)

@torch.inference_mode()
def predict_face(face_bgr):
    x = tf224(Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
    age_out, gender_out = age_gender_model(x)
    emo_out = emotion_model(x)
    age = float(age_out.item())
    gender = DEFAULT_GENDER[int(torch.argmax(gender_out, 1))]
    emotion = DEFAULT_EMOTION[int(torch.argmax(emo_out, 1))]
    return age, gender, emotion

@torch.inference_mode()
def predict_person(person_bgr):
    x = tf224(Image.fromarray(cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
    cloth_logits = clothing_model(x).cpu().numpy()[0]
    pose_logits = pose_model(x).cpu().numpy()[0]
    cloth_probs = softmax_np(cloth_logits)
    pose_probs  = softmax_np(pose_logits)

    cloth_buf.append(cloth_probs)
    pose_buf.append(pose_probs)

    # EMA / mean smoothing
    cloth_avg = np.mean(np.stack(cloth_buf, 0), 0)
    pose_avg  = np.mean(np.stack(pose_buf, 0), 0)

    ci = int(cloth_avg.argmax()); cc = float(cloth_avg[ci])
    pi = int(pose_avg.argmax());  pc = float(pose_avg[pi])

    # top-3 (for on-screen debugging)
    top3_c = np.argsort(-cloth_avg)[:3]
    top3_p = np.argsort(-pose_avg)[:3]

    top3_c_txt = ", ".join([f"{CLOTHING_LABELS[i]}:{cloth_avg[i]:.2f}" for i in top3_c])
    top3_p_txt = ", ".join([f"{POSE_LABELS[i]}:{pose_avg[i]:.2f}" for i in top3_p])

    return (CLOTHING_LABELS[ci], cc, top3_c_txt), (POSE_LABELS[pi], pc, top3_p_txt)

def expand_bbox(x, y, w, h, img_w, img_h, pad=0.25):
    # add padding on each side, keep inside image
    cx, cy = x + w/2, y + h/2
    half_w = w*(1+pad)/2; half_h = h*(1+pad)/2
    x1 = max(0, int(cx - half_w)); y1 = max(0, int(cy - half_h))
    x2 = min(img_w, int(cx + half_w)); y2 = min(img_h, int(cy + half_h))
    return x1, y1, x2 - x1, y2 - y1

def center_fallback(frame, scale=0.75):
    H, W = frame.shape[:2]
    s = int(min(H, W)*scale)
    x = (W - s)//2; y = (H - s)//2
    return frame[y:y+s, x:x+s]

# --------------------
# 6) Main loop
# --------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam"); raise SystemExit

while True:
    ok, frame = cap.read()
    if not ok: break
    draw = frame.copy()
    H, W = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        try:
            age, gender, emotion = predict_face(face_roi)
            cv2.rectangle(draw, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(draw, f"Age {age:.1f} | {gender} | {emotion}",
                        (x, max(0,y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        except Exception: pass

    # Person (HOG + padding; fallback center crop)
    rects, weights = hog.detectMultiScale(frame, winStride=(8,8), padding=(8,8), scale=1.05)
    person_roi = None; px=py=pw=ph=0
    if len(rects):
        best = int(np.argmax(weights)) if len(weights) else 0
        (x,y,w,h) = rects[best]
        x,y,w,h = expand_bbox(x,y,w,h,W,H,pad=0.35)
        person_roi = frame[y:y+h, x:x+w]; px,py,pw,ph = x,y,w,h
    else:
        person_roi = center_fallback(frame, scale=0.8)

    try:
        (cloth_label, cloth_conf, cloth_top3), (pose_label, pose_conf, pose_top3) = predict_person(person_roi)
        msg = f"Clothing: {cloth_label} ({cloth_conf:.2f}) | Pose: {pose_label} ({pose_conf:.2f})"
        if len(rects):
            cv2.rectangle(draw, (px,py), (px+pw,py+ph), (255,128,0), 2)
            cv2.putText(draw, msg, (px, max(0, py-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,128,0), 2)
        else:
            cv2.putText(draw, msg, (10, H-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,128,0), 2)

        # DEBUG top-3 (press 'd' to toggle if you like; here always on for calibration)
        debug_y = 40
        cv2.putText(draw, f"[Top3 Clothing] {cloth_top3}", (10, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.putText(draw, f"[Top3 Pose]     {pose_top3}", (10, debug_y+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    except Exception:
        pass

    cv2.putText(draw, "HAR — Age/Gender/Emotion + Clothing/Pose", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220,220,220), 2)

    show_frame("Human Attribute Recognition (4-in-1)", draw)

    if GUI_AVAILABLE:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        try:
            import msvcrt
            if msvcrt.kbhit() and msvcrt.getwch().lower() == 'q':
                break
        except Exception: pass

cap.release()
try: cv2.destroyAllWindows()
except Exception: pass