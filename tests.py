from data_processing import DataPreprocessor, Delexicalizer
from slot_aligner import SlotAligner
import re
from nltk.tokenize import word_tokenize

# data_processor = DataPreprocessor()
# data_processor.from_files(train_input_path="data/e2e-dataset/trainset.csv", validation_input_path="data/e2e-dataset/devset.csv", 
# 	test_input_path="data/e2e-dataset/testset.csv", delexicalization_type="partial", delexicalization_slots=["name", "near", "food"])


delexicalizer = Delexicalizer("partial", ["name", "near", "food"])

# inp_mr = "name[The Golden Curry], food[French], customer rating[low], area[city centre], familyFriendly[no], near[Café Rouge]"
# output = "a X-con-cuisine-food restaurant near X-near in city centre is X-name. it has a low customer rating and is not kid friendly."

# delexicalizer.reverse_delexicalize_sample(inp_mr, output)

# name[The Cambridge Blue], eatType[pub], food[Fast food], priceRange[high], near[Café Brazil]
# X-name pub is expensive and offers gourmet burgers and freshly cut fries

# name[The Golden Curry], food[Italian], customer rating[3 out of 5], area[riverside], familyFriendly[yes], near[Café Rouge]
# X-name is a kid-friendly X-vow-cuisine-food restaurant near the riverside. the customer rating is 3 out of 5.


aligner = SlotAligner()
some_test = "x-name is a kid friendly restaurant which serves amazing vitamin a"
some_test_tok = word_tokenize(some_test)

ref_text = "a X-con-cuisine-food restaurant near X-near in city centre is X-name. it has a low customer rating and is not kid friendly."
ref_text = ref_text.lower()
ref_text_tokenized = re.findall(r"[\w'-]+|[.,!?;]", ref_text)
#print(ref_text[110:130])

aligner.align_familyfriendly(ref_text, ref_text_tokenized, "no")
print(aligner.align_food(some_test, "american"))

print(some_test_tok[3])