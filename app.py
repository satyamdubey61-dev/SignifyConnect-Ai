"""
Flask backend for SignifyConnect - Sign Language Detection.
Exposes:
- /api/predict for image-based sign language recognition.
- /api/signup and /api/login for simple user authentication.
- /api/contact for "Get In Touch" form (sends email to company).
"""
import os
import base64
import math
import smtplib
import sqlite3
from email.mime.text import MIMEText
from email.utils import formataddr

import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from werkzeug.security import generate_password_hash, check_password_hash

# Paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_dotenv():
    """Load .env file from project root so you don't have to set GMAIL_APP_PASSWORD in CMD every time."""
    env_path = os.path.join(BASE_DIR, ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and os.environ.get(key) in (None, ""):
                    os.environ[key] = value


_load_dotenv()

MODEL_PATH = os.path.join(BASE_DIR, "converted_keras", "keras_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "converted_keras", "labels.txt")
USERS_DB_PATH = os.path.join(BASE_DIR, "users.db")

# Contact form: messages are sent to this address (read from env after .env is loaded)
CONTACT_TO_EMAIL = os.environ.get("CONTACT_TO_EMAIL", "info.evolvora@gmail.com")
CONTACT_FROM_EMAIL = os.environ.get("CONTACT_FROM_EMAIL", "info.evolvora@gmail.com")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")

app = Flask(__name__, static_folder="Frontend", static_url_path="")
app.secret_key = os.environ.get("SIGNIFYCONNECT_SECRET_KEY", "dev-secret-key")
CORS(app, resources={r"/api/*": {"origins": "*"}})


def get_db_connection():
    """Return a new SQLite connection for the users database."""
    conn = sqlite3.connect(USERS_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_users_db():
    """Create the users table if it does not already exist."""
    conn = get_db_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


# Ensure the users table exists when the app starts
init_users_db()

# Load model and labels once at startup
# IMPORTANT: This backend receives independent images over HTTP, not a continuous video stream.
# Use staticMode=True to avoid MediaPipe timestamp ordering issues ("Packet timestamp mismatch").
detector = HandDetector(staticMode=True, maxHands=1)
classifier = Classifier(MODEL_PATH, LABELS_PATH)

# Parse labels from labels.txt (format: "0 Hello", "1 Thank You", etc.)
labels = []
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    labels.append(parts[1])  # e.g. "Hello", "Thank You"
                else:
                    labels.append(line)
if not labels:
    labels = ["Hello", "Thank You", "Yes"]

IMG_SIZE = 300
OFFSET = 20


def preprocess_and_predict(img_array):
    """Run hand detection + classification on a BGR image. Returns label or None."""
    if img_array is None or not hasattr(img_array, "shape") or len(img_array.shape) != 3:
        return None

    hands, _ = detector.findHands(img_array)
    if not hands:
        return None

    hand = hands[0]
    bbox = hand.get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    x, y, w, h = bbox

    if w <= 0 or h <= 0:
        return None

    img_white = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255

    y1 = max(0, y - OFFSET)
    y2 = min(img_array.shape[0], y + h + OFFSET)
    x1 = max(0, x - OFFSET)
    x2 = min(img_array.shape[1], x + w + OFFSET)

    img_crop = img_array[y1:y2, x1:x2]
    if img_crop.size == 0:
        return None

    aspect_ratio = h / float(w)
    if aspect_ratio > 1:
        k = IMG_SIZE / h
        w_cal = math.ceil(k * w)
        img_resize = cv2.resize(img_crop, (w_cal, IMG_SIZE))
        w_gap = math.ceil((IMG_SIZE - w_cal) / 2)
        img_white[:, w_gap : w_cal + w_gap] = img_resize
    else:
        k = IMG_SIZE / w
        h_cal = math.ceil(k * h)
        img_resize = cv2.resize(img_crop, (IMG_SIZE, h_cal))
        h_gap = math.ceil((IMG_SIZE - h_cal) / 2)
        img_white[h_gap : h_cal + h_gap, :] = img_resize

    prediction, index = classifier.getPrediction(img_white, draw=False)
    if index is None or index < 0 or index >= len(labels):
        return None
    return labels[index]


@app.route("/api/signup", methods=["POST"])
def signup():
    """
    Simple signup endpoint.
    Expects JSON: { "name": "...", "email": "...", "password": "..." }
    """
    if not request.is_json:
        return jsonify({"success": False, "message": "Request must be JSON."}), 400

    data = request.get_json() or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not name or not email or not password:
        return jsonify({"success": False, "message": "Name, email and password are required."}), 400

    if len(password) < 6:
        return jsonify({"success": False, "message": "Password must be at least 6 characters long."}), 400

    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
            (name, email, generate_password_hash(password)),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        return jsonify({"success": False, "message": "An account with this email already exists."}), 400
    finally:
        conn.close()

    return jsonify({"success": True, "message": "Account created successfully."})


@app.route("/api/login", methods=["POST"])
def login():
    """
    Simple login endpoint.
    Expects JSON: { "email": "...", "password": "..." }
    """
    if not request.is_json:
        return jsonify({"success": False, "message": "Request must be JSON."}), 400

    data = request.get_json() or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"success": False, "message": "Email and password are required."}), 400

    conn = get_db_connection()
    try:
        user = conn.execute(
            "SELECT id, name, email, password_hash FROM users WHERE email = ?",
            (email,),
        ).fetchone()
    finally:
        conn.close()

    if user is None or not check_password_hash(user["password_hash"], password):
        return jsonify({"success": False, "message": "Invalid email or password."}), 401

    return jsonify(
        {
            "success": True,
            "message": "Login successful.",
            "user": {
                "id": user["id"],
                "name": user["name"],
                "email": user["email"],
            },
        }
    )


@app.route("/api/contact", methods=["POST"])
def contact():
    """
    "Get In Touch" form: expects JSON { "name": "...", "email": "...", "message": "..." }.
    Sends an email to CONTACT_TO_EMAIL (info.evolvora@gmail.com) via Gmail SMTP.
    Set GMAIL_APP_PASSWORD in the environment for the Gmail account that sends the email.
    """
    if not request.is_json:
        return jsonify({"success": False, "message": "Request must be JSON."}), 400

    data = request.get_json() or {}
    name = (data.get("name") or "").strip()
    sender_email = (data.get("email") or "").strip()
    message_body = (data.get("message") or "").strip()

    if not name or not sender_email or not message_body:
        return jsonify({"success": False, "message": "Name, email and message are required."}), 400

    if not GMAIL_APP_PASSWORD:
        return jsonify({
            "success": False,
            "message": "Contact form is not configured. Administrator: set GMAIL_APP_PASSWORD.",
        }), 503

    subject = f"SignifyConnect – Get In Touch from {name}"
    body = f"""You received a message from the SignifyConnect contact form.

Name: {name}
Email: {sender_email}

Message:
{message_body}
"""
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = formataddr(("SignifyConnect Contact", CONTACT_FROM_EMAIL))
    msg["To"] = CONTACT_TO_EMAIL
    msg["Reply-To"] = sender_email

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(CONTACT_FROM_EMAIL, GMAIL_APP_PASSWORD)
            server.sendmail(CONTACT_FROM_EMAIL, CONTACT_TO_EMAIL, msg.as_string())
    except Exception as e:
        return jsonify({"success": False, "message": f"Failed to send email: {str(e)}"}), 500

    return jsonify({"success": True, "message": "Message sent successfully. We'll get back to you soon."})


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "demo.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Accept JSON: { "image": "base64_encoded_string" }
    or form-data with file field "image".
    Returns { "label": "Hello" } or { "label": null, "error": "..." }.
    """
    img_data = None

    if request.is_json:
        data = request.get_json()
        img_data = data.get("image")
        if isinstance(img_data, str) and img_data.startswith("data:"):
            # Strip data URL prefix if present
            img_data = img_data.split(",", 1)[-1]
    elif request.files:
        file = request.files.get("image")
        if file:
            img_data = base64.b64encode(file.read()).decode("utf-8")

    if not img_data:
        return jsonify({"label": None, "error": "No image provided. Send JSON { \"image\": \"<base64>\" } or form-data 'image'."}), 400

    try:
        raw = base64.b64decode(img_data)
        nparr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"label": None, "error": "Invalid image data."}), 400

        label = preprocess_and_predict(img)
        return jsonify({"label": label})
    except Exception as e:
        return jsonify({"label": None, "error": str(e)}), 500


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "labels": labels})


if __name__ == "__main__":
    # We already load .env above via _load_dotenv(); skip Flask's dotenv so it doesn't require python-dotenv
    os.environ.setdefault("FLASK_SKIP_DOTENV", "1")
    print("Starting SignifyConnect backend. Open http://127.0.0.1:5000 in your browser.")
    app.run(host="0.0.0.0", port=5000, debug=True)
