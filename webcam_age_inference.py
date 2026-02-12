import os
import tempfile
import cv2
import torch
import torchvision.transforms as transforms
from models.model import AgeGenderModel
from models.emotion_model import EmotionModel
from PIL import Image

# --------------------
# 1. Load models
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Age-Gender Model
age_gender_model = AgeGenderModel()
age_gender_model.load_state_dict(torch.load("checkpoints/age_gender_model.pth", map_location=device))
age_gender_model.to(device)
age_gender_model.eval()

# Emotion Model
emotion_model = EmotionModel()
emotion_model.load_state_dict(torch.load("checkpoints/emotion_model.pth", map_location=device))
emotion_model.to(device)
emotion_model.eval()

# --------------------
# 2. Preprocessing
# --------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --------------------
# 3. Load Haar Cascade for face detection
# --------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# --------------------
# 4. Open webcam
# --------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam")
    exit()

# --- Display safety: detect whether cv2.imshow is available (some wheels are headless)
GUI_AVAILABLE = True

def _init_display_test():
    global GUI_AVAILABLE
    try:
        # small dummy image
        test_img = (255 * torch.zeros((1, 1, 3), dtype=torch.uint8).numpy()).reshape((1,1,3))
        cv2.namedWindow("__cv_test__")
        cv2.imshow("__cv_test__", test_img)
        cv2.waitKey(1)
        cv2.destroyWindow("__cv_test__")
    except Exception:
        GUI_AVAILABLE = False
        print("OpenCV GUI functions appear unavailable (cv2.imshow/waitKey failed). Falling back to disk-based display.")
        print("To enable native windows, install the standard OpenCV wheel: pip uninstall opencv-python-headless -y; pip install --force-reinstall opencv-python")

_fallback_path = os.path.join(tempfile.gettempdir(), "cv_har_last_frame.jpg")
_fallback_launched = False

def display_frame_safe(window_name, frame):
    global _fallback_launched
    if GUI_AVAILABLE:
        try:
            cv2.imshow(window_name, frame)
            return
        except Exception:
            # mark GUI unavailable for future frames
            print("cv2.imshow raised error; switching to fallback display.")
            # fall through to fallback
    # fallback: write to temp file, open once
    try:
        cv2.imwrite(_fallback_path, frame)
        if not _fallback_launched:
            try:
                if os.name == 'nt':
                    os.startfile(_fallback_path)
                else:
                    import subprocess
                    opener = 'xdg-open' if os.name == 'posix' else None
                    if opener:
                        subprocess.Popen([opener, _fallback_path])
                _fallback_launched = True
            except Exception:
                pass
    except Exception as e:
        print("Failed to write fallback frame:", e)

gender_labels = ["Male", "Female"]
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract face ROI
        face_img = frame[y:y+h, x:x+w]
        face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

        # Preprocess
        img_tensor = transform(face_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            # Age-Gender Prediction
            age_pred, gender_pred = age_gender_model(img_tensor)
            predicted_age = age_pred.item()
            predicted_gender = torch.argmax(gender_pred, dim=1).item()

            # Emotion Prediction
            emotion_pred = emotion_model(img_tensor)
            predicted_emotion = torch.argmax(emotion_pred, dim=1).item()

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Label text
        label = f"Age: {predicted_age:.1f}, {gender_labels[predicted_gender]}, {emotion_labels[predicted_emotion]}"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    display_frame_safe("Age, Gender & Emotion Detection", frame)

    # Press 'q' to quit
    if GUI_AVAILABLE:
        try:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception:
            GUI_AVAILABLE = False
            print("cv2.waitKey raised error; switching to console keyboard fallback.")
    else:
        # fallback: on Windows, allow pressing 'q' in console (msvcrt)
        try:
            import msvcrt
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch.lower() == 'q':
                    break
        except Exception:
            # no console key detection available; user can Ctrl-C to quit
            pass

cap.release()
try:
    cv2.destroyAllWindows()
except Exception:
    pass
