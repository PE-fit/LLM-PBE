from attacks import Jailbreak
from data import JailbreakQueries
from metrics import JailbreakRate
from models import TogetherAIModels

data = JailbreakQueries()
# Fill api_key
llm = TogetherAIModels(model="togethercomputer/llama-2-7b-chat", api_key="")
attack = Jailbreak()
results = attack.execute_attack(data, llm)
rate = JailbreakRate(results).compute_metric()
print("rate:", rate)
