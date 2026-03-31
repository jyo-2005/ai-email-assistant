from models import Action

def predict(observation):
    email_text = observation.email.lower()

    # 🔑 Keywords for spam detection
    spam_keywords = ["win", "offer", "free", "prize", "click", "urgent"]

    # 🚫 If spam-like words found → mark as spam
    if any(word in email_text for word in spam_keywords):
        return Action(action="spam")

    # ✅ Otherwise → important
    return Action(action="important")
