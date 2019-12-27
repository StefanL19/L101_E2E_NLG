import pandas as pd
from tqdm import tqdm

df = pd.read_csv("data/inp_and_gt_name_near_food_area_no_inform_fixed.csv")
for index, row in tqdm(df.iterrows()):
	if (("area" not in row["source_language"]) and ("area" in row["target_language"]) and (row["split"]=="train")): #and ("x-area" not in row["target_language"]):
		print(row["target_language"])
		# inp_gts = row["inp_gt"].split(", ")
		# inp_gts = list(filter(lambda a: "area" not in a, inp_gts))
		# inp_gt = ", ".join(inp_gts)
		# df.loc[index, "inp_gt"] = inp_gt

#df.to_csv("data/inp_and_gt_name_near_food_area_no_inform_fixed.csv", encoding='utf-8', index=False)
