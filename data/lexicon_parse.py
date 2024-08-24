#!/usr/bin/env python3

import pandas as pd
import json

data = pd.read_excel("lexiconwex2023.xls", sheet_name="ATUS 2023 Lexicon", skiprows=(1), names=["code", "activity", "examples1", "examples2"])

current = None
current_activity = None
examples = []

activity_map = {}
for _, row in data.iterrows():
    if pd.isnull(row["code"]) == False and pd.isnull(row["activity"]) == False:
        if current is not None:
            activity_map[current] = {}
            activity_map[current]["activity"] = current_activity.strip()
            activity_map[current]["examples"] = examples
            examples = []

        current = row["code"]
        current_activity = row["activity"].replace(", n.e.c.*", "")
    else:
        if pd.isnull(row["examples1"]) == False:
            examples.append(row["examples1"].strip())
        if pd.isnull(row["examples2"]) == False:
            examples.append(row["examples2"].strip())

all_data = []
for a in activity_map:
    for ex in activity_map[a]["examples"]:
        all_data.append({"code":a, "activity":activity_map[a]["activity"], "example":ex})
df = pd.DataFrame(all_data)
df.to_csv("lexiconwex2023.csv", index=False)

# with open("lexiconwex2023.json", 'w') as out:
#     json.dump(activity_map, out, indent=3)