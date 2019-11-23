import pandas as pd
import numpy as np
import string
import re
from tqdm import tqdm

# inspired by https://github.com/UFAL-DSG/tgen/blob/master/e2e-challenge/input/convert.py 

def preprocess_input_no_delexicalization(input_string):
	pass

def full_delexicalization(csv_path, type_subs_dict):
	df = pd.read_csv(csv_path)

	training_input_sentences = []
	training_output_sentences = []

	for index, row in tqdm(df.iterrows()):
		input_text = ""

		mr = row["mr"]
		mr_parts = mr.split(",")

		inp = ""
		target = row["ref"].translate(str.maketrans('', '', "#%&'()*+,./:;<=>?@[\]^_`{|}~")).lower()
		for mr_part in mr_parts:
			slot_type = mr_part.split("[")[0].strip().lower()
			slot_value = mr_part.split("[")[1].strip("]").strip().lower()

			slot_type_token = type_subs_dict[slot_type]
			inp += " info " + slot_type_token + " " + slot_value


			if slot_type != "familyfriendly":
				target = target.replace(slot_value, "<value_"+slot_type+">")
				target = re.sub(r"£([0-9]|-|£)+","<value_pricerange>", target)

		# print(inp)
		# print(target)
		# print(row["ref"])
		# print("------------------")

		training_input_sentences.append(inp)
		training_output_sentences.append(target)

	return training_input_sentences, training_output_sentences
		
			

type_subs_dict = {
	"name":"<name>",
	"eattype":"<eattype>",
	"pricerange":"<pricerange>",
	"customer rating":"<customerrating>",
	"near":"<near>",
	"food":"<food>",
	"area":"<area>",
	"familyfriendly":"<familyfriendly>"
}

train_inp_sent, train_out_sent = full_delexicalization("data/e2e-dataset/trainset.csv", type_subs_dict)

for idx, inp in enumerate(train_inp_sent):
	with open("data/full_delexicalization_train_input.txt", "a") as f:
		f.write(inp+"\n")

	with open("data/full_delexicalization_train_output.txt", "a") as f:
		f.write(train_out_sent[idx]+"\n")

