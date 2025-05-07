import xml.etree.ElementTree as ET
from collections import defaultdict

def parse_xml(xml_file):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    sentences = {}
    
    # Iterate over each sentence in the XML
    for sentence in root.findall('.//sentence'):
        sentence_id = sentence.get('id')
        lemmas = set()

        # Collect all lemmas from 'wf' and 'instance' tags
        for wf in sentence.findall('.//wf'):
            lemmas.add(wf.get('lemma'))
        
        for instance in sentence.findall('.//instance'):
            lemmas.add(instance.get('lemma'))

        # Store the sentence and its associated lemmas
        sentences[sentence_id] = lemmas
    
    return sentences

def generate_sentence_pairs(sentences):
    sentence_pairs = []

    # Create a mapping from lemma to sentence IDs
    lemma_to_sentences = defaultdict(list)
    for sentence_id, lemmas in sentences.items():
        for lemma in lemmas:
            lemma_to_sentences[lemma].append(sentence_id)

    # Generate pairs of sentences that share at least one lemma
    for lemma, sentence_ids in lemma_to_sentences.items():
        if len(sentence_ids) > 1:
            for i in range(len(sentence_ids)):
                for j in range(i + 1, len(sentence_ids)):
                    sentence_pairs.append((sentence_ids[i], sentence_ids[j]))

    return sentence_pairs


def read_gold_key(gold_key_file):
    gold_key = {}
    with open(gold_key_file, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            
            parts = line.split()
            if len(parts) != 2:
                print(f"Skipping invalid line: {line}")
                continue  # Skip invalid lines
            
            sentence_pair, label = parts
            try:
                sentence_1, sentence_2 = sentence_pair.split('-')
                gold_key[(sentence_1, sentence_2)] = int(label)
            except ValueError:
                print(f"Skipping invalid sentence pair: {sentence_pair}")
                continue  # Skip invalid sentence pairs
    return gold_key


def evaluate_sentence_pairs(sentence_pairs, gold_key):
    # Evaluate based on the gold key
    results = []
    for pair in sentence_pairs:
        pair_id = (pair[0], pair[1])
        if pair_id in gold_key:
            label = gold_key[pair_id]
            results.append((pair, label))
    return results

# Example usage
xml_file = r'C:\Users\amush\INLP_Project\Finetuning\WSD_Unified_Evaluation_Datasets\WSD_Unified_Evaluation_Datasets\senseval2\senseval2.data.xml'  # Path to your XML file

# Parse the XML and generate sentence pairs
sentences = parse_xml(xml_file)
sentence_pairs = generate_sentence_pairs(sentences)

# Optional: If a gold key file is available, evaluate the sentence pairs
gold_key_file = r'C:\Users\amush\INLP_Project\Finetuning\WSD_Unified_Evaluation_Datasets\WSD_Unified_Evaluation_Datasets\senseval2\senseval2.gold.key.txt'  # Provide the path to your gold key file
gold_key = read_gold_key(gold_key_file)
evaluation_results = evaluate_sentence_pairs(sentence_pairs, gold_key)

# Output the results
for pair, label in evaluation_results:
    print(f"Sentence Pair: {pair}, Label: {label}")
