from data_processing import DataPreprocessor

data_processor = DataPreprocessor()
data_processor.from_files(train_input_path="data/e2e-dataset/trainset.csv", validation_input_path="data/e2e-dataset/devset.csv", 
	test_input_path="data/e2e-dataset/testset.csv", delexicalization_type="partial", delexicalization_slots=["name", "near", "food"])