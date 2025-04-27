from umbrella.speculation.auto_engine import AutoEngine
from umbrella.logging_config import setup_logger
import json
from umbrella.templates import Prompts, SysPrompts
import sys

DEVICE = "cuda:0"
configuration = "./configs/chat_config_12gb.json"
with open(configuration, "r") as f:
    config = json.load(f)

engine = AutoEngine.from_config(DEVICE, **config)
engine.initialize()


res = engine.generate(context="Hello, how are you?", max_new_tokens=30)
print(res['generated_text'])