from env import EmailEnv
from models import Action
from grader import grade

def predict(observation):
    email_text = observation.email.lower()
    spam_keywords = ["win", "offer", "free", "prize", "click", "urgent"]

    if any(word in email_text for word in spam_keywords):
        return Action(action="spam")
    
    return Action(action="important")


if __name__ == "__main__":
    env = EmailEnv()
    observation = env.reset()

    print("[START] task=email_classification", flush=True)

    predictions = []
    labels = [email["label"] for email in env.emails]

    done = False
    step_count = 0

    while not done:
        action = predict(observation)

        observation, reward, done, _ = env.step(action)

        predictions.append(action.action)
        step_count += 1

        print(f"[STEP] step={step_count} reward={reward}", flush=True)

    score = grade(predictions, labels)

    print(f"[END] task=email_classification score={score} steps={step_count}", flush=True)
