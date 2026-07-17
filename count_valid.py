import json

with open("./evaluation_manual/benjamin50_guilherme.json", "r", encoding="utf-8") as file:
    guilherme = json.load(file)

with open("./evaluation_manual/benjamin50_leticia.json", "r", encoding="utf-8") as file:
    leticia = json.load(file)

merged = []
for gui, let in zip(guilherme, leticia):
    merged.append({
        "question": gui.get("question"),
        "guilherme": gui.get("guilherme"),
        "leticia": let.get("leticia")
    })

valid = 0
mh = 0
both = 0
for i in merged:
    print(i)
    is_valid = False
    is_mh = False
    if i["guilherme"]["answerable"] and i["leticia"]["answerable"]:
        valid += 1
        is_valid = True
    if i["guilherme"]["multi-hop"] and i["leticia"]["multi-hop"]:
        mh += 1
        is_mh = True
    if is_valid and is_mh:
        both += 1

print(valid)
print(mh)
print(both)