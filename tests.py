from data_processing import DataPreprocessor, Delexicalizer
from slot_aligner import SlotAligner
import re
from nltk.tokenize import word_tokenize
import data_augmentation

# Test data processor
data_processor = DataPreprocessor()
data_processor.from_files(train_input_path="data/e2e-dataset/trainset.csv", validation_input_path="data/e2e-dataset/devset.csv", 
	test_input_path="data/e2e-dataset/testset.csv", delexicalization_type="partial", delexicalization_slots=["name", "near", "food"])


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






