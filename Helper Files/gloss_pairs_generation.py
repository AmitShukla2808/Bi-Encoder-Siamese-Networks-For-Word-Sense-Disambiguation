import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
import csv
import nltk
import random
import time
import re
import xml.etree.ElementTree as ET
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer, util
import spacy
from transformers import MarianMTModel, MarianTokenizer
from difflib import SequenceMatcher


#----------------------------------------------- BACK TRANSLATION -----------------------------------------------------------------

nlp = spacy.load("en_core_web_sm")
sentence_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def sequence_similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def extract_words_with_lemma(text, target_lemma):
    doc = nlp(text)
    return [token.text for token in doc if token.lemma_.lower() == target_lemma.lower()]

def get_synonyms(word, pos_tag):
    synonyms = set()
    for syn in wn.synsets(word, pos=pos_tag):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    return list(synonyms)

def basic_paraphrase(text, lemma):
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.lemma_.lower() == lemma.lower():
            tokens.append(token.text)
            continue
        if token.pos_ in ["ADJ", "ADV"] and random.random() < 0.5:
            pos_map = {"ADJ": wn.ADJ, "ADV": wn.ADV}
            synonyms = get_synonyms(token.text, pos_map[token.pos_])
            if synonyms:
                tokens.append(random.choice(synonyms))
                continue
        tokens.append(token.text)
    return " ".join(tokens)

def chained_back_translation(text, lemma, similarity_threshold=0.90, max_attempts=10):
    print(f"[Original English] {text}")

    attempts = 0
    while attempts < max_attempts:
        candidates = [basic_paraphrase(text, lemma) for _ in range(5)]
        en_emb = sentence_model.encode(text, convert_to_tensor=True)
        candidate_embeddings = sentence_model.encode(candidates, convert_to_tensor=True)
        cosine_scores = util.cos_sim(en_emb, candidate_embeddings)

        sorted_indices = cosine_scores[0].argsort(descending=True)

        for idx in sorted_indices:
            best_candidate = candidates[idx]
            matched_words = extract_words_with_lemma(best_candidate, lemma)
            similarity = sequence_similarity(best_candidate, text)

            if matched_words and similarity >= similarity_threshold:
                print(f"[Back-translated Candidate] {best_candidate}")
                return best_candidate, matched_words

        attempts += 1

    print("[Warning] No strong match found after multiple attempts. Falling back to original.")
    return text, extract_words_with_lemma(text, lemma)

#------------------------------------------------------------------------------------------------------------

# Load the sentence transformer model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ---------------------- Helpers ----------------------

def load_keyfile(keyfile_path="semcor.gold.key.txt"):
    id_to_sensekey = {}
    try:
        with open(keyfile_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    id_to_sensekey[parts[0]] = parts[1]
    except Exception as e:
        print(f"[ERROR] Failed to load keyfile: {e}")
    return id_to_sensekey

def get_gold_synset(sense_key):
    try:
        lemma = wn.lemma_from_key(sense_key)
        return lemma.synset()
    except Exception as e:
        print(f"[ERROR] Failed to get synset for sense key {sense_key}: {e}")
        return None

def highlight_word(sentence, target_word):
    pattern = r'\b' + re.escape(target_word) + r'\b'
    highlighted = re.sub(pattern, f'"{target_word}"', sentence, count=1)
    highlighted = highlighted.strip()
    if highlighted.startswith('"') and highlighted.endswith('"') and highlighted.count('"') == 2:
        highlighted = highlighted[1:-1]
    return highlighted

def format_gloss(word, gloss):
    return f'{word} : {gloss}'

def remove_punctuation_except_quotes(text):
    return ''.join(ch for ch in text if ch.isalnum() or ch.isspace() or ch == '"')

# ---------------------- SemCor Processing ----------------------

def process_semcor(xml_path="semcor.data.xml", keyfile_path="semcor.gold.key.txt", max_contexts=50000):
    id_to_sensekey = load_keyfile(keyfile_path)
    word_occurrences = []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        print(f"[DEBUG] Parsed XML file: {xml_path}")

        for sentence in root.findall('.//sentence'):
            words = []
            for elem in sentence:
                if elem.tag in ['wf', 'instance'] and elem.text:
                    words.append(elem)

            sentence_text = ' '.join([e.text for e in words])

            for elem in words:
                if elem.tag == 'instance':
                    local_id = elem.get('id', '')
                    lemma = elem.get('lemma', '')
                    surface_word = elem.text.strip() if elem.text else ''
                    sense_key = id_to_sensekey.get(local_id, None)

                    if lemma and sense_key and surface_word:
                        gold_synset = get_gold_synset(sense_key)
                        if gold_synset:
                            word_occurrences.append({
                                'sentence': sentence_text,
                                'lemma': lemma,
                                'word': surface_word,
                                'gold_synset': gold_synset,
                                'sense_key': sense_key
                            })

        print(f"[DEBUG] Total target words found: {len(word_occurrences)}")
        return random.sample(word_occurrences, min(max_contexts, len(word_occurrences)))

    except Exception as e:
        print(f"[ERROR] Failed to parse SemCor: {e}")
        return []

# ---------------------- Data Writers ----------------------

def generate_context_gloss_pairs(data, filename, k_hard=1, k_medium=1, k_easy=1):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Context Sentence', 'Synset Gloss Definition', 'Label'])

        for occ in data:
            context = highlight_word(remove_punctuation_except_quotes(occ['sentence']), occ['word'])
            all_contexts = []
            orig_context = occ['sentence']
            lemma = occ['lemma']
            for _ in range(3):
                new_context, match_words = chained_back_translation(orig_context, lemma)
                if len(match_words) == 0:
                    continue
                all_contexts.append(highlight_word(remove_punctuation_except_quotes(new_context),match_words[0]))     
            gold_syn = occ['gold_synset']
            correct_gloss = format_gloss(lemma, gold_syn.definition())
            correct_emb = sbert_model.encode(correct_gloss, convert_to_tensor=True)
            
            cntpos = 1
            writer.writerow([context, correct_gloss, 1])

            negs = []
            for syn in wn.synsets(lemma):
                if syn != gold_syn:
                    gloss = format_gloss(lemma, syn.definition())
                    emb = sbert_model.encode(gloss, convert_to_tensor=True)
                    sim = util.cos_sim(correct_emb, emb).item()
                    negs.append(([context, gloss, 0], sim))

            negs.sort(key=lambda x: x[1], reverse=True)
            all_negs = negs
            random.shuffle(all_negs)

            for pair, _ in all_negs:
                writer.writerow(pair)
            
            extra = len(all_negs) - cntpos
            
            for contexts in all_contexts:
                if extra <= 0:
                    break
                writer.writerow([contexts, correct_gloss, 1])
                extra -= 1
            
            while extra > 0:
                writer.writerow([context, correct_gloss, 1])
                extra -= 1

def generate_context_hypernym_pairs(data, filename, k_hard=1, k_medium=1, k_easy=1):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Context Sentence', 'Hypernym Gloss Definition', 'Label'])

        for occ in data:
            context = highlight_word(remove_punctuation_except_quotes(occ['sentence']), occ['word'])
            all_contexts = []
            orig_context = occ['sentence']
            lemma = occ['lemma']
            for _ in range(3):
                new_context, match_words = chained_back_translation(orig_context, lemma)
                if len(match_words) == 0:
                    continue
                all_contexts.append(highlight_word(remove_punctuation_except_quotes(new_context),match_words[0])) 
            
            gold_syn = occ['gold_synset']
            context_emb = sbert_model.encode(context, convert_to_tensor=True)

            pos_hyps = gold_syn.hypernyms()
            positives = []
            gloss = ''
            for hyp in pos_hyps:
                gloss = format_gloss(hyp.lemmas()[0].name(), hyp.definition())
                emb = sbert_model.encode(gloss, convert_to_tensor=True)
                sim = util.cos_sim(context_emb, emb).item()
                positives.append(([context, gloss, 1], sim))

            cntpos = 1
            writer.writerow([context, gloss, 1])
            correct_gloss = gloss

            negs = []
            for syn in wn.synsets(lemma):
                if syn != gold_syn:
                    for hyp in syn.hypernyms():
                        gloss = format_gloss(hyp.lemmas()[0].name(), hyp.definition())
                        emb = sbert_model.encode(gloss, convert_to_tensor=True)
                        sim = util.cos_sim(context_emb, emb).item()
                        negs.append(([context, gloss, 0], sim))

            negs.sort(key=lambda x: x[1])  # ascending = least similar = hardest
            all_negs = negs
            random.shuffle(all_negs)

            for pair, _ in all_negs:
                writer.writerow(pair)
                
            extra = len(all_negs) - cntpos
            
            for contexts in all_contexts:
                if extra <= 0:
                    break
                writer.writerow([contexts, gloss, 1])
                extra -= 1

            positives.sort(key=lambda x: -x[1])
            for pair, _ in positives[:k_hard]:
                if extra <= 0:
                    break
                writer.writerow(pair)
                extra -= 1
            
            while extra > 0:
                writer.writerow([context, correct_gloss, 1])
                extra -= 1

# ---------------------- Main ----------------------

def main():
    start_time = time.time()
    data = process_semcor("semcor.data.xml", "semcor.gold.key.txt", max_contexts=5000)

    if data:
        generate_context_gloss_pairs(data, "context_gloss_pairs_mixed.csv", k_hard=1, k_medium=1, k_easy=1)
        generate_context_hypernym_pairs(data, "context_hypernym_pairs_mixed.csv", k_hard=1, k_medium=1, k_easy=1)
        print("[SUCCESS] CSVs created successfully.")
    else:
        print("[FAILURE] No valid data found.")

    print(f"[DEBUG] Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
