from transformers import BertTokenizer
import nltk
import pandas as pd
nltk.download('wordnet')
nltk.download('wordnet_ic')
nltk.download('omw-1.4')



from nltk.corpus import wordnet as wn, wordnet_ic, stopwords

def get_wordnet_meaning(sense_key):
    """Convert a WordNet sense key to its definition."""
    try:
        return wn.lemma_from_key(sense_key).synset().definition()
    except:
        return "Definition not found"


def preprocess_wic():
    train_path = "C:/Users/amush/INLP_Project/WiC_dataset/train/train.data.txt"
    train_labels_path = "C:/Users/amush/INLP_Project/WiC_dataset/train/train.gold.txt"
    
    df_train = pd.read_csv(train_path, delimiter='\t', names=["word", "pos", "index-range", "sentence1", "sentence2"])
    df_train['label'] = pd.read_csv(train_labels_path, delimiter='\t', names=["label"])
    
    sentence_pairs = list(zip(df_train['sentence1'], df_train['sentence2']))
    target_words = df_train['word'].tolist()
    labels = df_train['label'].tolist()
    return sentence_pairs, target_words, labels



