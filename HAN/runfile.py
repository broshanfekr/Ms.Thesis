import os
os.system("python3 train_LSTM.py -input_dataset_path input_data/total.tar -word2vec_model_path word2vec_model/IMDB_skipgram_model -output_path han_result.txt -num_epochs 20 -batch_size 7")