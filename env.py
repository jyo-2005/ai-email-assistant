from models import Observation,Action,Reward

class EmailEnv:
    def __init__(self):
        self.emails = [
    {"email": "Win money now!!!", "sender": "spam@xyz.com", "label": "spam"},
    {"email": "Meeting at 10 AM", "sender": "boss@company.com", "label": "important"},
    {"email": "Limited offer just for you", "sender": "ads@shop.com", "label": "spam"},
    {"email": "Project deadline tomorrow", "sender": "manager@company.com", "label": "important"},
    {"email": "Congratulations! You won a gift card", "sender": "promo@deal.com", "label": "spam"}
]
        self.index = 0
    def reset(self):
        self.index = 0
        return self._get_observation()
    def state(self):
        return self._get_observation()
    def _get_observation(self):
        data = self.emails[self.index]
        return Observation(email=data["email"],sender=data["sender"])
    def step(self, action: Action):
        current_email = self.emails[self.index]
        correct_label = current_email["label"]

        # reward logic
        if action.action == correct_label:
            reward = Reward(value=1.0)
        else:
            reward = Reward(value=-1.0)

        # move to next email
        self.index += 1
        done = self.index >= len(self.emails)

        # next observation
        observation = None if done else self._get_observation()

        return observation, reward, done, {}