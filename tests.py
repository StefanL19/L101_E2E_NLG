from data_processing import DataPreprocessor, Delexicalizer
from slot_aligner import SlotAligner
import re
from nltk.tokenize import word_tokenize
import data_augmentation
from data_loader import NMTDataset
import json
import pandas as pd
from alignment_utils import tokenize_mr


# ############################### Create a new data processor and vectorizer
processor = DataPreprocessor.from_files(train_input_path="data/e2e-dataset/trainset.csv", validation_input_path="data/e2e-dataset/devset.csv", 
 	test_input_path="data/e2e-dataset/testset.csv", delexicalization_type="partial", delexicalization_slots=["name", "near", "food", "pricerange"])

# processor.save_data("data/inp_and_gt.csv")
# dataset = NMTDataset.load_dataset_and_make_vectorizer("data/inp_and_gt.csv")
# dataset.save_vectorizer("data/vectorizer.json")


##################################### Add TUDA Samples 
# processor = DataPreprocessor.from_existing_df("data/inp_and_gt.csv", delexicalization_type="partial", delexicalization_slots=["name", "near", "food"])
# processor.add_samples("data/trainset.csv.predicted", "data/e2e-dataset/trainset.csv")
# processor.save_data("data/inp_and_gt_augmented.csv")

# # ################################# Load data processor and vectorizer
# processor = DataPreprocessor.from_existing_df("data/inp.csv", delexicalization_type="partial", delexicalization_slots=["name", "near", "food"])
# dataset = NMTDataset.load_dataset_and_load_vectorizer("data/inp.csv", "data/model_storage/vectorizer.json")

# vect = dataset.get_vectorizer()
# s_v = vect.target_vocab
# print(s_v.lookup_index(3))
# print(s_v.lookup_index(34))
#print(s_v.lookup_index(18))
# res = vect.vectorize("<inform> pricerange more than £30 <inform> position inner", "prices start at £30 .")
# print(res)
# 

# # ##################################### Test Reverse Delexicalization
# delexicalizer = Delexicalizer("partial", ["name", "near", "food"])
# text_df = pd.read_csv("data/inp_and_gt.csv")
# for row in text_df[0:1].iterrows():
# 	print(row[1]["source_language"])
# 	print("------------------------")
# 	print(row[1]["target_language"])
# 	print("------------------------")
# 	print(row[1]["inp_gt"])
# 	print("------------------------")

# 	pred_reverse_delexicalized = delexicalizer.reverse_delexicalize_sample(row[1]["inp_gt"], row[1]["target_language"])
# 	print(pred_reverse_delexicalized)
# 	print("-------------------------")
# 	print(row[1]["ref_gt"])


##################################### Test classification reranker
# mr = "name[The Golden Palace], eatType[coffee shop], food[Fast food], priceRange[high]"

# ref = "there is a high priced x-con-food place called x-name. with an average customer rating . it is like a coffee shop . and it is kid-friendly . "

# ######################################## Test Reranker 
# aligner = SlotAligner()
# inp = tokenize_mr(mr)
# res = aligner.alignment_reranker(inp, ref)
# print("The alignment result is: ", res)



# processor.save_data("data/inp.csv")

#processor = DataPreprocessor.from_existing_df("data/inp.csv", delexicalization_type="partial", delexicalization_slots=["name", "near", "food"])

# vect = dataset._vectorizer
# vectorizer_serializable = vect.to_serializable()



# delexicalizer = Delexicalizer("partial", ["name", "near", "food"])
# aligner = SlotAligner()

# inp_mr = "name[The Golden Curry], food[French], customer rating[low], area[city centre], familyFriendly[no], near[Café Rouge]"
# output = "a X-con-cuisine-food restaurant near X-near in city centre is X-name. it has a low customer rating and is not kid friendly."

# delexicalizer.reverse_delexicalize_sample(inp_mr, output)

# name[The Cambridge Blue], eatType[pub], food[Fast food], priceRange[high], near[Café Brazil]
# X-name pub is expensive and offers gourmet burgers and freshly cut fries

# name[The Golden Curry], food[Italian], customer rating[3 out of 5], area[riverside], familyFriendly[yes], near[Café Rouge]
# X-name is a kid-friendly X-vow-cuisine-food restaurant near the riverside. the customer rating is 3 out of 5.



# some_test = "x-name is a kid friendly restaurant which serves amazing vitamin a"
# some_test_tok = word_tokenize(some_test)

# ref_text = "a X-con-cuisine-food restaurant near X-near in city centre is X-name. it has a low customer rating and is not kid friendly."
# ref_text = ref_text.lower()
# ref_text_tokenized = re.findall(r"[\w'-]+|[.,!?;]", ref_text)
# #print(ref_text[110:130])

# aligner.align_familyfriendly(ref_text, ref_text_tokenized, "no")
# print(aligner.align_food(some_test, "american"))

# print(some_test_tok[3])


# ## Test Name alignment
# inp = "a X-con-cuisine-food restaurant near X-near in city centre is. it has a low customer rating and is not kid friendly.."
# pos, cnt = aligner.align_near(inp.lower(), "x-name")


# Test price range alignment


# # Test customer rating allignment
# inp = "a X-con-cuisine-food restaurant near X-near in city centre is X-name. it has a low customer rating and is not kid friendly."
# inp = "X-name is a kid-friendly X-vow-cuisine-food restaurant near the riverside. the customer rating is 3 out of 5"
# inp = inp.lower()
# inp_tok = word_tokenize(inp)

# pos = aligner.align_customer_rating(inp, inp_tok, "low")
# print(pos)

# # Test Family Friendly Alignment
# inp = "a X-con-cuisine-food restaurant near X-near in city centre is X-name. it has a low customer rating and is not kid friendly."
# inp = inp.lower()
# inp_tok = word_tokenize(inp)
# pos = aligner.align_familyfriendly(inp, inp_tok, "not")
# print(pos)

# # Test price range alignment
# #name[The Rice Boat], food[English], priceRange[¬£20-25], customer rating[high], area[riverside], familyFriendly[yes], near[Express by Holiday Inn],
# inp = "The Rice Boat serves English food and is highly rated by its customers. The prices are average but it is easily located close to the Express by Holiday Inn in Riverside."
# inp = inp.lower()
# inp_tok = word_tokenize(inp)
# pos = aligner.align_pricerange(inp, inp_tok, "¬£20-25")
# print(pos)


# # Test Area Alignment
# inp = "The Rice Boat serves English food and is highly rated by its customers. The prices are average but it is easily located close to the Express by Holiday Inn in Riverside."
# inp = inp.lower()
# inp_tok = word_tokenize(inp)
# pos = aligner.align_area(inp, inp_tok, "riverside")
# print(inp[159:170])


# # Test Eat type Alignment
# #"name[Strada], eatType[pub], food[Italian], customer rating[5 out of 5], near[Yippee Noodle Bar]"
# inp = "There is a pub called Strada which serves Italian food. It's near Yippee Noodle Bar and has a 5 out of 5 customer rating."		
# inp = inp.lower()
# inp_tok = word_tokenize(inp)
# pos = aligner.align_eattype(inp, inp_tok, "pub")
# print(inp[11:20])

## Test utterance splitting
# mr = "name[Strada], eatType[pub], food[Italian], customer rating[5 out of 5], near[Yippee Noodle Bar]"
# ref = "There is a pub called Strada which serves Italian food. It's near Yippee Noodle Bar and has a 5 out of 5 customer rating."
# ref = ref.lower()

# data_augmentation.split_utterance(mr, ref)






