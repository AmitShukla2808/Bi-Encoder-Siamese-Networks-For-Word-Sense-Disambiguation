import xml.etree.ElementTree as ET
from collections import defaultdict
from itertools import combinations
import pandas as pd

# === Load gold labels ===
# === Load gold labels ===
def load_gold_labels(gold_file):
    gold = {}
    with open(gold_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            inst_id = parts[0]
            senses = set(parts[1:])
            gold[inst_id] = senses
    return gold

# === Parse XML ===
def parse_instances(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    instances = {}
    for sentence in root.iter('sentence'):
        inst_id = sentence.attrib['id']
        lemma = None  # We will extract lemma from instance elements
        
        words = []
        
        # Iterate over all elements inside the sentence
        for word in sentence:
            if word.tag == 'wf':  # normal word form
                words.append(word.text)
            elif word.tag == 'instance':  # word with instance (i.e., sense)
                words.append(word.text)
                if not lemma:  # Capture the first lemma
                    lemma = word.attrib['lemma']

        # Join all words to form the full sentence
        full_sentence = ' '.join(words).strip()
        
        if lemma:  # Ensure we have a valid lemma
            instances[inst_id] = {'lemma': lemma, 'sentence': full_sentence}
        else:
            print(f"Warning: Missing lemma for sentence ID: {inst_id}")

    return instances

# === Build sentence pairs ===
def build_pairs(instances, gold):
    lemma_to_instances = defaultdict(list)

    # Group instances by lemma
    for inst_id, data in instances.items():
        lemma_to_instances[data['lemma']].append(inst_id)

    sentence_pairs = []
    
    # Iterate over all groups of instances for each lemma
    for lemma, inst_ids in lemma_to_instances.items():
        for id1, id2 in combinations(inst_ids, 2):
            sent1 = instances[id1]['sentence']
            sent2 = instances[id2]['sentence']
            senses1 = gold.get(id1, set())  # Get senses from gold for the first sentence
            senses2 = gold.get(id2, set())  # Get senses from gold for the second sentence
            
            # Label: 1 if there's overlap in senses, else 0
            label = int(bool(senses1 & senses2))  # 1 if overlap, 0 otherwise
            sentence_pairs.append((sent1, sent2, lemma, label))

    return sentence_pairs

# Load files
gold = load_gold_labels('C:/Users/amush/INLP_Project/Finetuning/WSD_Unified_Evaluation_Datasets/WSD_Unified_Evaluation_Datasets/semeval2015/semeval2015.gold.key.txt')
instances = parse_instances('C:/Users/amush/INLP_Project/Finetuning/WSD_Unified_Evaluation_Datasets/WSD_Unified_Evaluation_Datasets/semeval2015/semeval2015.data.xml')
pairs = build_pairs(instances, gold)

# Optional: Convert to DataFrame
df = pd.DataFrame(pairs, columns=["sent1", "sent2", "lemma", "ground_truth"])

# Save to CSV
df.to_csv("semeval2015.csv", index=False)


gold = load_gold_labels('C:/Users/amush/INLP_Project/Finetuning/WSD_Unified_Evaluation_Datasets/WSD_Unified_Evaluation_Datasets/semeval2013/semeval2013.gold.key.txt')
instances = parse_instances('C:/Users/amush/INLP_Project/Finetuning/WSD_Unified_Evaluation_Datasets/WSD_Unified_Evaluation_Datasets/semeval2013/semeval2013.data.xml')
pairs = build_pairs(instances, gold)

# Optional: Convert to DataFrame
df = pd.DataFrame(pairs, columns=["sent1", "sent2", "lemma", "ground_truth"])

# Save to CSV
df.to_csv("semeval2013.csv", index=False)


gold = load_gold_labels('C:/Users/amush/INLP_Project/Finetuning/WSD_Unified_Evaluation_Datasets/WSD_Unified_Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt')
instances = parse_instances('C:/Users/amush/INLP_Project/Finetuning/WSD_Unified_Evaluation_Datasets/WSD_Unified_Evaluation_Datasets/semeval2007/semeval2007.data.xml')
pairs = build_pairs(instances, gold)

# Optional: Convert to DataFrame
df = pd.DataFrame(pairs, columns=["sent1", "sent2", "lemma", "ground_truth"])

# Save to CSV
df.to_csv("semeval2007.csv", index=False)

