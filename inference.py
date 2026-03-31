from env import EmailEnv
from models import Action
from grader import grade

def run_baseline():
    # create environment
    env = EmailEnv()

    # start environment
    observation = env.reset()

    predictions = []
    labels = [email["label"] for email in env.emails]

    done = False

    while not done:
        # simple rule-based logic
        email_text = observation.email.lower()

        if "win" in email_text or "offer" in email_text:
            action = Action(action="spam")
        else:
            action = Action(action="important")

        # store prediction
        predictions.append(action.action)

        # take step
        observation, reward, done, _ = env.step(action)

    # evaluate performance
    score = grade(predictions, labels)

    print("Baseline Score:", score)
