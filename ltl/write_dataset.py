import json

with open("czw_test") as f:
    data = json.load(f)

for s in data["data"]:
    print(s["sentence"])
