# TextSummarization

This project aims at generating relevant summaries for Amazon product reviews using Encoder-Decoder LSTM Network. 
To run this follow the steps listed below.

1. Download the dataset from the following website and rename it as “Reviews.cv" 
https://www.kaggle.com/snap/amazon-fine-food-reviews

2. Download the wordembeddings from the following website and rename it as 
“numberbatch-en.txt”
https://github.com/commonsense/conceptnet-numberbatch

3. Run the following command to perform pre-processing
python preprocess_data.py

4. Run the following commands to train the model  with or without attention
python train_model.py --attention yes
python train_model.py --attention no
