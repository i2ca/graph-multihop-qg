import json
from copy import deepcopy

reviewers = ["leticia", "guilherme", "sergio"]

files = {
    "leticia": "./evaluation_manual/benjamin50_leticia.json",
    "guilherme": "./evaluation_manual/benjamin50_guilherme.json",
    "sergio": "./evaluation_manual/benjamin50_sergio.json",
}

data = {}
for reviewer, path in files.items():
    with open(path, "r", encoding="utf-8") as file:
        data[reviewer] = json.load(file)

guilherme_key = "guilherme"
leticia_key = "leticia"
sergio_key = "sergio"

guilherme_items = data[guilherme_key]
leticia_items = data[leticia_key]
sergio_items = data[sergio_key]

if not (len(guilherme_items) == len(leticia_items) == len(sergio_items)):
    raise ValueError("Guilherme, Leticia, and Sergio files must have the same number of items.")

merged_items = []

for g_item, l_item, s_item in zip(guilherme_items, leticia_items, sergio_items):
    g_eval = g_item.get(guilherme_key, {})
    l_eval = l_item.get(leticia_key, {})
    s_eval = s_item.get(sergio_key, {})

    answerable_votes = [
        bool(g_eval.get("answerable")),
        bool(l_eval.get("answerable")),
        bool(s_eval.get("answerable")),
    ]
    multihop_votes = [
        bool(g_eval.get("multi-hop")),
        bool(l_eval.get("multi-hop")),
        bool(s_eval.get("multi-hop")),
    ]

    chosen_eval = {
        "answerable": sum(answerable_votes) >= 2,
        "multi-hop": sum(multihop_votes) >= 2,
    }

    # Preserve the rest of the item structure from one source item
    merged_item = deepcopy(g_item)
    del merged_item[guilherme_key]
    merged_item["mix"] = chosen_eval
    merged_items.append(merged_item)

output_path = "./evaluation_manual/benjamin50_mix.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(merged_items, f, ensure_ascii=False, indent=2)

print(f"Saved merged file to: {output_path}")