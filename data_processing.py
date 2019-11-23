import pandas as pd
from tqdm import tqdm

def tokenize_mr(sample):
		"""
			Performs initial tokenization of the input sample
			sample: input, output pair to be tokenized
			mode: train, validation, or test mode
		"""
		output = {}

		mr_parts = sample.split(",")

		for mr_part in mr_parts:
			slot_type = mr_part.split("[")[0].strip().lower()
			slot_value = mr_part.split("[")[1].strip("]").strip().lower()
			
			output[slot_type] = slot_value.lower()

		return output



class DataPreprocessor(object):
	def __init__(self, dataframe=None, delexicalization_type="full", delexicalization_slots=None):
		self.delexicalization_type = delexicalization_type
		self.dataframe = dataframe
		self.delexicalization_slots = delexicalization_slots

	@classmethod
	def from_existing_df(cls, existing_df_path, delexicalization_type="full", delexicalization_slots=None):
		pass

	@classmethod 
	def from_files(cls, train_input_path, validation_input_path, test_input_path,  delexicalization_type="full", delexicalization_slots=None):
		"""
			Instantiates the DataPreprocessor class from a list of training, validation, and test data paths

			Args:
				train_input_path (string): Input path to csv file containing the training samples
				validation_input_path (string): Input path to the csv file containing the validation samples
				test_input_path (string): Input path to the csv file containing the test samples
				delexicalization_type (string): One of: full, partial, none
				delexicalization_slots (array): Slots to be delexicalized, none if delexicalization type is full or none
		"""
		train_df = pd.read_csv(train_input_path)
		validation_df = pd.read_csv(validation_input_path)
		test_df = pd.read_csv(test_input_path)

		delexicalizer = Delexicalizer(delexicalization_type, delexicalization_slots)

		c = 0
		for index, row in tqdm(train_df.iterrows()):
			input_mr = tokenize_mr(row[0])
			output_ref = row[1]
			
			# Check if output ref is present and convert it to lower
			if output_ref != None:
				output_ref = output_ref.lower()

			input_mr, output_ref, delexicalization_results = delexicalizer.delexicalize_sample(input_mr, output_ref)
			
			# print(input_mr)
			# print(output_ref)

			if False in delexicalization_results:
				print(row[0])
				print(output_ref)
				print("----------------------------------")
				c += 1

		print(c)

		


			# print(input_mr)
			# print(output_ref)
			# print(delexicalization_results)

	def add_samples(self, samples_file_path):
		pass

	def save_output(self, output_path):
		pass

class Delexicalizer(object):
	def __init__(self, delexicalization_type, delexicalization_slots=None):
		self.delexicalization_type = delexicalization_type
		self.delexicalization_slots = delexicalization_slots

	def delexicalize_sample(self, inp, output=None):
		"""
			Delexicalizes an input sample

			Args:
			inp(Dictionary): Dictionary containing the input slot types and the corresponding values
			output(string): The reference sentence
		"""

		if self.delexicalization_type == "full":
			inp, output, delexicalization_results = self._full_delexicalization(inp, output)
		elif self.delexicalization_type == "partial":
			inp, output, delexicalization_results = self._partial_delexicalization(inp, output)
		else:
			pass

		return inp, output, delexicalization_results


	def _full_delexicalization(self, inp, output):
		"""
			Performs full delexicalization of an input sample

			Args:
			inp(Dictionary): Dictionary containing the input slot types and the corresponding values
			output(string): The reference sentence
		"""
		pass

	def _partial_delexicalization(self, inp, output):
		"""
			Performs partial delexicalization of an input sample

			Args:
			inp(Dictionary): Dictionary containing the input slot types and the corresponding values
			output(string): The reference sentence
		"""
		# A list to keep track on successfull/unsuccesfull delexicalizations
		delexicalization_results = []
		if "name" in self.delexicalization_slots:
			inp, output, is_success = self._delexicalize_name(inp, output)
			delexicalization_results.append(is_success)

		if "near" in self.delexicalization_slots:
			inp, output, is_success = self._delexicalize_near(inp, output)
			delexicalization_results.append(is_success)

		if "food" in self.delexicalization_slots:
			inp, output, is_success = self._delexicalize_food_slug2slug(inp, output)

		return inp, output, delexicalization_results

	def _delexicalize_name(self, inp, output):
		"""
			Delexicalizes the name field in an input/output pair
			sample: input, output pair to be delexicalized
		"""
		# Variable to check if the delexicalization was successfull
		# A delexicalization is unsuccessfull iff the name slot is present in 
		# the mr but the value is not found in the output with the used technique
		
		is_success = True
		if "name" in inp.keys():
			name_value = inp["name"]
			inp["name"] = "X-name"
			
			if output != None:
				# We are not at testing time
				# Check if the name can be found in the output
				if name_value in output:
					output = output.replace(name_value, "X-name")
				else:
					is_success = False
		
		return inp, output, is_success

	def _delexicalize_eat_type(self, sample):
		"""
			Delexicalizes the name field in an input/output pair
			sample: input, output pair to be delexicalized
		"""
		pass

	def _delexicalize_price_range(self, sample):
		"""
			Delexicalizes the name field in an input/output pair
			sample: input, output pair to be delexicalized
		"""
		pass

	def _delexicalize_customer_rating(self, sample):
		"""
			Delexicalizes the name field in an input/output pair
			sample: input, output pair to be delexicalized
		"""
		pass

	def _delexicalize_near(self, inp, output):
		"""
			Delexicalizes the name field in an input/output pair
			sample: input, output pair to be delexicalized
		"""
		is_success = True
		if "near" in inp.keys():
			name_value = inp["near"]
			inp["near"] = "X-near"
			
			if output != None:
				# We are not at testing time
				# Check if the name can be found in the output
				if name_value in output or "crown plaza hotel" in output:
					output = output.replace(name_value, "X-near")

					# Search specifically for Crowne Hotel Plaza as
					# it is a problem in many of the samples
					output = output.replace("crown plaza hotel", "X-near")
				else:
					is_success = False
		
		return inp, output, is_success

	def _delexicalize_food_slug2slug(self, inp, output):
		"""
			Delexicalizes the name field in an input/output pair
			sample: input, output pair to be delexicalized
		"""
		# The food delexicalization idea was taken from the slug2slug model as the name of the function states
		# https://www.macs.hw.ac.uk/InteractionLab/E2E/final_papers/E2E-Slug2Slug.pdf

		is_success = False
		if "food" in inp.keys():
			food_value = inp["food"]

			# Different cuisine names starting with a vowel
			cuisine_vow = ["italian", "english"]

			# Different cuisine names starting with a consonant
			cuisine_con = ["japanese", "french", "chinese"]

			# Different types of food 
			food = ["fast food"]

			if food_value in cuisine_vow:
				inp["food"] = "X-vow-cuisine-food"

			elif food_value in cuisine_con:
				inp["food"] = "X-con-cuisine-food"

			else:
				inp["food"] = "X-con-food"

			if output != None:
				for cv in cuisine_vow:
					if cv in output:
						is_success = True
						output = output.replace(cv, "X-vow-cuisine-food")

				for cc in cuisine_con:
					if cc in output:
						is_success = True
						output = output.replace(cc, "X-con-cuisine-food")

				for f in food:
					if f in output:
						is_success = True
						output = output.replace(f, "X-con-food")
		else:
			is_success = True

		return inp, output, is_success

	def _delexicalize_area(self, sample):
		"""
			Delexicalizes the name field in an input/output pair
			sample: input, output pair to be delexicalized
		"""
		pass

	def _delexicalize_family_friendly(self, sample):
		"""
			Delexicalizes the name field in an input/output pair
			sample: input, output pair to be delexicalized
		"""
		pass

