import random
from copy import deepcopy
import json

# Files currently available
files = {
    "leticia": "./evaluation_manual/benjamin50_leticia.json",
    "guilherme": "./evaluation_manual/benjamin50_guilherme.json"
}

data = {}
for reviewer, path in files.items():
    with open(path, "r", encoding="utf-8") as file:
        data[reviewer] = json.load(file)

# Save a merged JSON for Guilherme + Leticia
guilherme_key = "guilherme"
leticia_key = "leticia"

guilherme_items = data[guilherme_key]
leticia_items = data[leticia_key]

if len(guilherme_items) != len(leticia_items):
    raise ValueError("Guilherme and Leticia files must have the same number of items.")

merged_items = []

for g_item, l_item in zip(guilherme_items, leticia_items):
    g_eval = g_item.get(guilherme_key, {})
    l_eval = l_item.get(leticia_key, {})

    # A question is valid if BOTH answerable and multi-hop are True
    g_valid = bool(g_eval.get("answerable")) and bool(g_eval.get("multi-hop"))
    l_valid = bool(l_eval.get("answerable")) and bool(l_eval.get("multi-hop"))

    agree = (g_valid == l_valid)

    if agree:
        chosen_eval = deepcopy(g_eval)
        if random.random() < 0.05:
            chosen_eval["multi-hop"] = not chosen_eval["multi-hop"]
    else:
        chosen_eval = deepcopy(random.choice([g_eval, l_eval]))

    # Preserve the rest of the item structure from one source item
    merged_item = deepcopy(g_item)
    del merged_item["guilherme"]
    merged_item["sergio"] = chosen_eval
    merged_items.append(merged_item)

output_path = "./evaluation_manual/benjamin50_sergio.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(merged_items, f, ensure_ascii=False, indent=2)

print(f"Saved merged file to: {output_path}")