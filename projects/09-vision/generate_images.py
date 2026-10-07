"""projects/09-vision/generate_images.py

Render a handful of synthetic *error screenshots* (window chrome + an error dialog)
with PIL, so the ground-truth text and intent are known exactly. Used by run.py to
test vision extraction + intent routing. Writes images/ + labels.jsonl.

    python generate_images.py
"""
import json
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(HERE, "images")

# (id, title, error_line, key_phrase the model must surface, gold intent)
CASES = [
    ("err1", "Billing", "Payment Declined: your card ending 4242 was declined.",
     "card", "billing"),
    ("err2", "API Console", "HTTP 429: Too Many Requests - rate limit exceeded.",
     "429", "technical"),
    ("err3", "Sign in", "Login Failed: incorrect password (3 attempts remaining).",
     "password", "account"),
    ("err4", "Dashboard", "Application Error: NullPointerException at Dashboard.render",
     "nullpointerexception", "technical"),
    ("err5", "Account", "Subscription Expired - renew your plan to restore access.",
     "subscription", "billing"),
]


def font(size):
    for p in ["/System/Library/Fonts/Supplemental/Arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc",
              "/Library/Fonts/Arial.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def render(case):
    _id, title, err, _kp, _intent = case
    W, H = 640, 360
    img = Image.new("RGB", (W, H), "#e9edf2")
    d = ImageDraw.Draw(img)
    # window chrome
    d.rectangle([20, 20, W - 20, H - 20], fill="white", outline="#c2c8d0", width=2)
    d.rectangle([20, 20, W - 20, 56], fill="#3a4252")
    for i, col in enumerate(["#ff5f57", "#febc2e", "#28c840"]):
        d.ellipse([34 + i * 22, 31, 48 + i * 22, 45], fill=col)
    d.text((110, 30), title, fill="white", font=font(16))
    # red error banner
    d.rectangle([44, 110, W - 44, 170], fill="#fdecea", outline="#e0b4b0", width=2)
    d.text((60, 122), "⚠  Error", fill="#b3261e", font=font(18))
    d.text((44, 196), err, fill="#202124", font=font(17))
    d.text((44, 250), "If the problem persists, contact support.",
           fill="#5f6368", font=font(14))
    os.makedirs(IMG_DIR, exist_ok=True)
    path = os.path.join(IMG_DIR, f"{_id}.png")
    img.save(path)
    return path


def main():
    rows = []
    for c in CASES:
        path = render(c)
        rows.append({"id": c[0], "image": os.path.relpath(path, HERE),
                     "error_text": c[2], "key_phrase": c[3], "intent": c[4]})
        print(f"{c[0]}: {c[2][:50]}...  -> {c[4]}")
    with open(os.path.join(HERE, "labels.jsonl"), "w") as f:
        f.write("\n".join(json.dumps(r) for r in rows))
    print(f"wrote {len(rows)} images + labels.jsonl")


if __name__ == "__main__":
    main()
