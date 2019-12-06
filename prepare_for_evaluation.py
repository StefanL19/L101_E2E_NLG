import pandas as pd

predictions_path = "data/res.csv"

df = pd.read_csv(predictions_path)

grp = df.groupby(['MR_GT'])


for name, group in grp:
	with open("data/results/gt_ref.txt", "a") as f:
		for samp in group["Reference_Ground_Truth"]:
			f.write(samp)
			f.write("\n")

		f.write("\n")

	with open("data/results/sampled_ref.txt", "a") as f_s:
		f_s.write(group["Sample_No_Delex"].iloc[0])
		f_s.write("\n")


	# for samp in group["Sample_No_Delex"]:
	# 	print(samp)
