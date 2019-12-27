from collections import Counter
from nltk.util import ngrams 

text = ""
n_gram = 1

with open("../L101_metrics/my_metrics/14/sampled_ref.txt", "r") as f:
	for line in f:
		line = line.strip("\n").lower()
		text += line

	print(len(list(Counter(ngrams(text.split(), n_gram)))))
	all_samples_area = list(Counter(ngrams(text.split(), n_gram)))

text = ""
print("-------------------------------------------------------------")
with open("../L101_metrics/my_metrics/25/sampled_ref.txt", "r") as f:
	for line in f:
		line = line.strip("\n").lower()
		text += line

	print(len(list(Counter(ngrams(text.split(), n_gram)))))
	all_samples = list(Counter(ngrams(text.split(), n_gram)))

diff = []
for sample in all_samples_area:
	if sample not in all_samples:	
		diff.append(sample)

print(diff)
