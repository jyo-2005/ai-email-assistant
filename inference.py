import os
from openai import OpenAI
from env import EmailEnv
from models import Action
from grader import grade

# -----------------------------
# 🔑 Setup LLM (MANDATORY)
# -----------------------------
client = OpenAI(
    base_url=os.environ.get("API_BASE_URL"),
    api_key=os.environ.get("API_KEY")
)

# -----------------------------
# 🤖 PREDICT FUNCTION
# -----------------------------
def predict(observation):
    email = observation.email

    # 🔥 TRY LLM FIRST
    try:
        response = client.chat.completions.create(
            model=os.environ.get("MODEL_NAME", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": "Classify email as spam or important. Reply only 'spam' or 'important'."
                },
                {
                    "role": "user",
                    "content": email
                }
            ],
            timeout=10
        )

        result = response.choices[0].message.content.strip().lower()

        if "spam" in result:
            return Action(action="spam")
        else:
            return Action(action="important")

    except Exception as e:
        # ⚠️ VERY IMPORTANT: fallback (prevents crash)
        email_text = email.lower()
        spam_keywords = ["win", "offer", "free", "prize", "click", "urgent"]

        if any(word in email_text for word in spam_keywords):
            return Action(action="spam")

        return Action(action="important")


# -----------------------------
# 🚀 RUN ENVIRONMENT
# -----------------------------
def run():
    env = EmailEnv()
    observation = env.reset()

    print("[START] task=email_classification", flush=True)

    predictions = []
    labels = [email["label"] for email in env.emails]

    done = False
    step = 0

    while not done:
        action = predict(observation)

        observation, reward, done, _ = env.step(action)

        predictions.append(action.action)
        step += 1

        print(f"[STEP] step={step} reward={reward}", flush=True)

    score = grade(predictions, labels)

    print(f"[END] task=email_classification score={score} steps={step}", flush=True)


# 🔥 FORCE RUN
run()
