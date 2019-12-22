from collections import Counter
from nltk.util import ngrams 

text = ""
n_gram = 3

with open("../L101_metrics/my_metrics/13/sampled_ref.txt", "r") as f:
	for line in f:
		line = line.strip("\n").lower()
		text += line

	print(len(list(Counter(ngrams(text.split(), n_gram)))))

