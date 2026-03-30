from fastapi import FastAPI
from pydantic import BaseModel
import os

# Project imports
from env import EmailEnv
from tasks import tasks
from grader import grade
from models import Action
from fastapi.responses import FileResponse

# Optional OpenAI (REAL AI)
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except:
    client = None  # fallback if no API key

app = FastAPI()

# -----------------------------
# 📌 Input Models
# -----------------------------

class GraderInput(BaseModel):
    predictions: list
    labels: list

class EmailInput(BaseModel):
    email: str

# -----------------------------
# 🏠 Home Route
# -----------------------------

@app.get("/")
def home():
    return {"message": "MailMind API is running"}

# -----------------------------
# 📋 Tasks
# -----------------------------

@app.get("/tasks")
def get_tasks():
    return tasks

# -----------------------------
# 🤖 Baseline AI
# -----------------------------

@app.get("/baseline")
def get_baseline():
    env = EmailEnv()
    observation = env.reset()

    predictions = []
    labels = [email["label"] for email in env.emails]

    done = False

    while not done:
        email_text = observation.email.lower()

        # Simple rule-based AI
        if "win" in email_text or "offer" in email_text:
            action = "spam"
        else:
            action = "important"

        predictions.append(action)

        observation, reward, done, _ = env.step(Action(action=action))

    score = grade(predictions, labels)

    return {"baseline_score": score}

# -----------------------------
# 📊 Grader
# -----------------------------

@app.post("/grader")
def run_grader(data: GraderInput):
    score = grade(data.predictions, data.labels)
    return {"score": score}

# -----------------------------
# 🧠 Analyze Email (REAL-TIME)
# -----------------------------

@app.post("/analyze")
def analyze_email(data: EmailInput):
    email = data.email

    # 🔥 If OpenAI key exists → use AI
    if client:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Classify email as spam or important and suggest action (reply or ignore)."
                },
                {
                    "role": "user",
                    "content": email
                }
            ]
        )
        return {"result": response.choices[0].message.content}

    # 🟡 Fallback rule-based logic
    keywords = ["offer", "win", "free", "prize", "click", "urgent"]

    if any(word in email.lower() for word in keywords):
        label = "spam"
        action = "ignore"
    else:
        label = "important"
        action = "reply"

    return {
        "email": email,
        "label": label,
        "action": action
    }

# -----------------------------
# 🌐 UI Route
# -----------------------------

@app.get("/ui")
def get_ui():
    return FileResponse("index.html")