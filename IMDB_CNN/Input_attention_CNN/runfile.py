import os
os.system("python3 train.py -input_dataset_path input_data/total.tar -word2vec_model_path word2vec_model/IMDB_skipgram_model -output_path attention_cnn.txt -num_epochs 50 -batch_size 100") 
