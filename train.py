import os
import re
import numpy as np
import pandas as pd
import pickle
from nltk.tokenize import SyllableTokenizer
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

# set random seed for reproducibility
SEED = 42



### INTIALIZATION
# load datasets
datasets_folder_path = os.path.join(os.getcwd(), "datasets")
hongkong_username_df = pd.read_csv(os.path.join(datasets_folder_path, "hongkong_username.csv"))
non_hongkong_username_df = pd.read_csv(os.path.join(datasets_folder_path, "non_hongkong_username.csv"))

# label 0 for non-HK and 1 for HK
non_hongkong_username_df["Hong Kong"] = 0
hongkong_username_df["Hong Kong"] = 1

# merge datasets
df = pd.concat([hongkong_username_df, non_hongkong_username_df], axis=0, ignore_index=True)



# PREPROCESSING PIPELINE
# drop duplicates
original_size = len(df)
df = df.drop_duplicates(subset=["IG Username"])
print(f"{original_size - len(df)} duplicated entries removed. {len(df)} entries retained.")

# remove non-alpha characters
df["IG Username"] = df["IG Username"].astype(str)
df["IG Username"] = df["IG Username"].apply(lambda x: re.sub(r"[\d._]+", "", x))

# drop empty usernames
original_size = len(df)
df = df[df["IG Username"] != ""]
print(f"{original_size - len(df)} empty usernames removed. {len(df)} entries retained.")

# tokenize
tokenizer = SyllableTokenizer()
df["Tokenized IG Username"] = df["IG Username"].apply(tokenizer.tokenize)

# create a unique syllable vocabulary
syllable_vocab = list(set(df["Tokenized IG Username"].explode()))
print(f"Vocabulary size: {len(syllable_vocab)} unique syllables.")

# encoding
def manual_binarize(tokenized_username, syllable_vocab):
    binary_vector = [0] * len(syllable_vocab)
    for syllable in tokenized_username:
        if syllable in syllable_vocab:
            index = syllable_vocab.index(syllable)
            binary_vector[index] = 1
    return binary_vector

df["Encoded Username"] = df["Tokenized IG Username"].apply(lambda x: manual_binarize(x, syllable_vocab))



# MODEL TRAINING
# convert to approporiate data types
X = np.array(df["Encoded Username"].tolist())
y = df["Hong Kong"].values

# standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# initialize SVM with the tuned hyperparameters
svm_model = SVC(C=1, probability=True, random_state=SEED)

# train the model
print("Training the SVM model...")
svm_model.fit(X_scaled, y)
print("Training complete.")

# save the trained model, syllable vocabulary, and scaler
output_folder = os.getcwd()
with open(os.path.join(output_folder, "svm_model.pkl"), "wb") as model_file:
    pickle.dump(svm_model, model_file)
with open(os.path.join(output_folder, "syllable_vocab.pkl"), "wb") as vocab_file:
    pickle.dump(syllable_vocab, vocab_file)
with open(os.path.join(output_folder, "scaler.pkl"), "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Trained model, syllable vocabulary, and scaler saved successfully.")