import os
import cv2
import torch
import tempfile
import numpy as np
import torchvision.transforms as T
from PIL import Image

from models.model import AgeGenderModel
from models.emotion_model import EmotionModel
from models.clothing_model import ClothingModel
from models.pose_model import PoseModel

# --------------------
# 0. Labels (EDIT IF NEEDED)
# --------------------
GENDER_LABELS = ["Male", "Female"]
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# From your screenshot of the pose CSV
POSE_LABELS = [
    "sleeping", "drinking", "running", "eating", "sitting",
    "clapping", "fighting", "laughing", "calling",
    "listening_to_music", "using_laptop", "texting", "dancing"
]

# From your clothing CSV (product_type); tweak to match your training classes
CLOTHING_LABELS = ["T-shirt", "Shirt", "Sweater", "Tank top", "Jackets_Vests"]

# --------------------
# 1. Device
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# 2. Models + safe state_dict loader
# --------------------
def load_state_dict_safely(model, ckpt_path, map_location):
    sd = torch.load(ckpt_path, map_location=map_location)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    # Strip "module." if saved from DataParallel
    new_sd = {}
    for k, v in sd.items():
        new_k = k.replace("module.", "") if k.startswith("module.") else k
        new_sd[new_k] = v
    model.load_state_dict(new_sd, strict=False)
    return model

# Age-Gender
age_gender_model = AgeGenderModel()
load_state_dict_safely(age_gender_model, "checkpoints/age_gender_model.pth", device)
age_gender_model.to(device).eval()

# Emotion
emotion_model = EmotionModel()
load_state_dict_safely(emotion_model, "checkpoints/emotion_model.pth", device)
emotion_model.to(device).eval()

# Clothing (ResNet-50)
clothing_model = ClothingModel(num_classes=len(CLOTHING_LABELS), pretrained=False)
load_state_dict_safely(clothing_model, "checkpoints/best_clothing_model.pth", device)
clothing_model.to(device).eval()

# Pose (ResNet-18)
pose_model = PoseModel(num_classes=len(POSE_LABELS), pretrained=False)
load_state_dict_safely(pose_model, "checkpoints/best_pose_model.pth", device)
pose_model.to(device).eval()

# --------------------
# 3. Preprocessing
# --------------------
common_tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# --------------------
# 4. Detectors (faces + people)
# --------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# HOG person detector (classic opencv)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# --------------------
# 5. Display safety (headless-safe)
# --------------------
GUI_AVAILABLE = True
def _init_display_test():
    global GUI_AVAILABLE
    try:
        dummy = np.zeros((2,2,3), dtype=np.uint8)
        cv2.namedWindow("__cv_test__")
        cv2.imshow("__cv_test__", dummy)
        cv2.waitKey(1)
        cv2.destroyWindow("__cv_test__")
    except Exception:
        GUI_AVAILABLE = False
        print("OpenCV GUI not available; using file preview fallback.")

_fallback_path = os.path.join(tempfile.gettempdir(), "cv_har_last_frame.jpg")
_fallback_launched = False

def display_frame_safe(window_name, frame):
    global _fallback_launched
    if GUI_AVAILABLE:
        try:
            cv2.imshow(window_name, frame)
            return
        except Exception:
            print("cv2.imshow failed; switching to fallback.")
    # Fallback: write to temp and open once
    try:
        cv2.imwrite(_fallback_path, frame)
        if not _fallback_launched:
            try:
                if os.name == "nt":
                    os.startfile(_fallback_path)
                else:
                    import subprocess
                    subprocess.Popen(["xdg-open", _fallback_path])
                _fallback_launched = True
            except Exception:
                pass
    except Exception as e:
        print("Failed to write fallback frame:", e)

_init_display_test()

# --------------------
# 6. Helpers
# --------------------
def pil_from_bgr(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

@torch.inference_mode()
def predict_face_attrs(face_bgr):
    x = common_tf(pil_from_bgr(face_bgr)).unsqueeze(0).to(device)
    age_out, gender_out = age_gender_model(x)
    emotion_out = emotion_model(x)

    age = float(age_out.item())
    gender_idx = int(torch.argmax(gender_out, dim=1).item())
    emotion_idx = int(torch.argmax(emotion_out, dim=1).item())

    return age, GENDER_LABELS[gender_idx], EMOTION_LABELS[emotion_idx]

@torch.inference_mode()
def predict_clothing_pose(person_bgr):
    x = common_tf(pil_from_bgr(person_bgr)).unsqueeze(0).to(device)
    clothing_out = clothing_model(x)
    pose_out = pose_model(x)

    clothing_idx = int(torch.argmax(clothing_out, dim=1).item())
    pose_idx = int(torch.argmax(pose_out, dim=1).item())

    return CLOTHING_LABELS[clothing_idx], POSE_LABELS[pose_idx]

def letterbox_center_crop(img, scale=0.6):
    """Take a centered crop (scale fraction of min(H,W)) to reduce background."""
    h, w = img.shape[:2]
    side = int(min(h, w) * scale)
    y1 = max(0, h//2 - side//2); y2 = y1 + side
    x1 = max(0, w//2 - side//2); x2 = x1 + side
    return img[y1:y2, x1:x2]

# --------------------
# 7. Webcam Loop
# --------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam")
    raise SystemExit

while True:
    ok, frame = cap.read()
    if not ok:
        break

    draw = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Face attributes
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    face_texts = []
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        try:
            age, gender, emotion = predict_face_attrs(face_roi)
            face_text = f"Age {age:.1f} | {gender} | {emotion}"
            face_texts.append(face_text)

            cv2.rectangle(draw, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(draw, face_text, (x, max(0, y-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        except Exception:
            pass

    # --- Person region(s) for clothing/pose
    # Try HOG first
    rects, weights = hog.detectMultiScale(frame, winStride=(8,8), padding=(8,8), scale=1.05)
    person_text = ""
    if len(rects) > 0:
        # Use the highest-score detection
        best_idx = int(np.argmax(weights)) if len(weights) else 0
        (px, py, pw, ph) = rects[best_idx]
        person_roi = frame[py:py+ph, px:px+pw]
        try:
            clothing, pose = predict_clothing_pose(person_roi)
            person_text = f"Clothing: {clothing} | Pose: {pose}"
            cv2.rectangle(draw, (px, py), (px+pw, py+ph), (255, 128, 0), 2)
            cv2.putText(draw, person_text, (px, max(0, py-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,128,0), 2)
        except Exception:
            pass
    else:
        # Fallback: centered crop if no person found
        person_roi = letterbox_center_crop(frame, scale=0.6)
        try:
            clothing, pose = predict_clothing_pose(person_roi)
            person_text = f"Clothing: {clothing} | Pose: {pose}"
            cv2.putText(draw, person_text, (10, draw.shape[0]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,128,0), 2)
        except Exception:
            pass

    # A compact title bar
    title = "HAR — Age/Gender/Emotion + Clothing/Pose"
    cv2.putText(draw, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

    # Show
    display_frame_safe("Human Attribute Recognition (4-in-1)", draw)

    # Quit on 'q'
    if GUI_AVAILABLE:
        try:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception:
            GUI_AVAILABLE = False
    else:
        try:
            import msvcrt
            if msvcrt.kbhit() and msvcrt.getwch().lower() == 'q':
                break
        except Exception:
            pass

cap.release()
try:
    cv2.destroyAllWindows()
except Exception:
    pass
