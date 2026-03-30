from pydantic import BaseModel

# What the agent sees
class Observation(BaseModel):
    email:str
    sender:str

# What the agent does
class Action(BaseModel):
    action:str

# What the agent gets as feedback
class Reward(BaseModel):
    value:float