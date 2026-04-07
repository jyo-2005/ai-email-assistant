from env import EmailEnv
from models import Action
from grader import grade

# -----------------------------
# 🤖 PREDICT FUNCTION
# -----------------------------
def predict(observation):
    email = observation.email.lower()

    spam_keywords = ["win", "offer", "free", "prize", "click", "urgent"]

    if any(word in email for word in spam_keywords):
        return Action(action="spam")

    return Action(action="important")


# -----------------------------
# 🚀 RUN FUNCTION (IMPORTANT)
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


# 🔥 IMPORTANT: FORCE EXECUTION
run()
