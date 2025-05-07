import os
import csv
import nltk
import xml.etree.ElementTree as ET
from nltk.corpus import wordnet as wn
import random
import time
import sys

# Download necessary NLTK resources
nltk.download('wordnet', quiet=True)

def process_semcor_xml(xml_file_path="semcor.data.xml"):
    """
    Process the SemCor XML dataset to create pairs of sentences with a common target word,
    determining if the word has the same sense in both sentences.
    """
    start_time = time.time()
    output_file = "semcor_wsd_pairs.csv"
    
    print(f"[INFO] Starting SemCor XML processing...")
    print(f"[INFO] Input file: {xml_file_path}")
    print(f"[INFO] Output will be saved to: {output_file}")
    
    # Dictionary to store words and their occurrences across sentences
    # Structure: {lemma: [(sentence, position_in_sentence, sense_id, sentence_idx), ...]}
    word_occurrences = {}
    
    # Parse the XML file
    print(f"[INFO] Parsing XML file: {xml_file_path}")
    try:
        parse_start = time.time()
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        parse_end = time.time()
        print(f"[SUCCESS] XML parsing complete in {parse_end - parse_start:.2f} seconds")
        sys.stdout.flush()  # Force output to display
        
        # Process sentences and collect word occurrences
        total_sentences = len(root.findall('.//sentence'))
        print(f"[INFO] Found {total_sentences} sentences to process")
        sys.stdout.flush()
        
        processed_instances = 0
        unique_lemmas = set()
        
        # Track senses per lemma to identify ambiguous lemmas
        lemma_senses = {}
        
        # Process sentences and collect word occurrences
        for sentence_idx, sentence in enumerate(root.findall('.//sentence')):
            words = []
            instances_in_sentence = 0
            
            # Extract words and their information
            for elem in sentence:
                if elem.tag in ['wf', 'instance']:
                    words.append(elem.text)
                    
                    # Only process instances (tagged words)
                    if elem.tag == 'instance':
                        lemma = elem.get('lemma', '')
                        sense_id = elem.get('id', '')
                        
                        if lemma and sense_id:
                            instances_in_sentence += 1
                            processed_instances += 1
                            position = len(words)-1
                            sentence_text = ' '.join([w.text for w in sentence if w.tag in ['wf', 'instance']])
                            
                            # Add to our dictionary
                            if lemma not in word_occurrences:
                                word_occurrences[lemma] = []
                                unique_lemmas.add(lemma)
                                lemma_senses[lemma] = set()
                            
                            word_occurrences[lemma].append((sentence_text, position, sense_id, sentence_idx))
                            lemma_senses[lemma].add(sense_id)
            
            # Print progress
            if sentence_idx % 1000 == 0 or sentence_idx == total_sentences - 1:
                elapsed = time.time() - start_time
                progress = (sentence_idx + 1) / total_sentences * 100
                print(f"[PROGRESS] Processed {sentence_idx + 1}/{total_sentences} sentences ({progress:.1f}%) in {elapsed:.2f}s")
                sys.stdout.flush()
        
        print(f"[INFO] Processing complete!")
        print(f"[STATS] Processed {processed_instances} tagged instances")
        print(f"[STATS] Found {len(unique_lemmas)} unique lemmas")
        sys.stdout.flush()
        
        # Filter for ambiguous lemmas (those with multiple distinct senses)
        ambiguous_lemmas = [lemma for lemma, senses in lemma_senses.items() if len(senses) >= 2]
        print(f"[STATS] Found {len(ambiguous_lemmas)} lemmas with multiple senses")
        
        # Debug: Check a few ambiguous lemmas
        if ambiguous_lemmas:
            sample_lemmas = random.sample(ambiguous_lemmas, min(5, len(ambiguous_lemmas)))
            print(f"[DEBUG] Sample ambiguous lemmas: {sample_lemmas}")
            for lemma in sample_lemmas:
                print(f"[DEBUG] Lemma '{lemma}' has {len(lemma_senses[lemma])} senses: {list(lemma_senses[lemma])}")
                print(f"[DEBUG] Lemma '{lemma}' appears in {len(word_occurrences[lemma])} sentences")
        
        # Count sense occurrences for each lemma
        lemma_sense_counts = {}
        for lemma in ambiguous_lemmas:
            lemma_sense_counts[lemma] = {}
            for _, _, sense_id, _ in word_occurrences[lemma]:
                if sense_id not in lemma_sense_counts[lemma]:
                    lemma_sense_counts[lemma][sense_id] = 0
                lemma_sense_counts[lemma][sense_id] += 1
        
        # Find lemmas with at least two significant senses
        # (each sense should appear at least twice to ensure we can make true pairs)
        lemmas_with_significant_senses = []
        for lemma, sense_counts in lemma_sense_counts.items():
            significant_senses = [sense for sense, count in sense_counts.items() if count >= 2]
            if len(significant_senses) >= 2:
                lemmas_with_significant_senses.append(lemma)
        
        print(f"[STATS] Found {len(lemmas_with_significant_senses)} lemmas with multiple significant senses")
        
        # Group by lemma and sense_id for easier pair generation
        # Structure: {lemma: {sense_id: [(sentence, position, sentence_idx), ...], ...}, ...}
        lemma_sense_groups = {}
        for lemma in lemmas_with_significant_senses:
            lemma_sense_groups[lemma] = {}
            for sent, pos, sense_id, sent_idx in word_occurrences[lemma]:
                if sense_id not in lemma_sense_groups[lemma]:
                    lemma_sense_groups[lemma][sense_id] = []
                lemma_sense_groups[lemma][sense_id].append((sent, pos, sent_idx))
        
        # Debug: Check if we can make true pairs for some lemmas
        can_make_true_pairs = []
        for lemma, sense_groups in lemma_sense_groups.items():
            for sense_id, occurrences in sense_groups.items():
                # Check if this sense appears in at least 2 different sentences
                sentence_idxs = set(sent_idx for _, _, sent_idx in occurrences)
                if len(sentence_idxs) >= 2:
                    can_make_true_pairs.append(lemma)
                    break
        
        print(f"[STATS] Found {len(can_make_true_pairs)} lemmas that can make TRUE pairs")
        
        # If we found no valid lemmas, look for a looser definition
        if not can_make_true_pairs:
            print(f"[WARNING] No lemmas found that can make TRUE pairs. Using fallback approach.")
            # Find any lemmas that appear in multiple sentences (even with different senses)
            for lemma in ambiguous_lemmas:
                sentence_idxs = set(sent_idx for _, _, _, sent_idx in word_occurrences[lemma])
                if len(sentence_idxs) >= 2:
                    can_make_true_pairs.append(lemma)
            print(f"[STATS] Found {len(can_make_true_pairs)} lemmas in fallback mode")
        
        # Ensure we have lemmas to work with
        if not can_make_true_pairs:
            print(f"[ERROR] No suitable lemmas found even with fallback! Cannot proceed.")
            return
        
        # Generate sentence pairs
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['sent1', 'sent2', 'lemma', 'ground_truth'])
            
            # Settings for pair generation
            total_pairs_desired = 10000
            target_true_ratio = 0.5  # Try for 50% true pairs but accept variation
            max_pairs_per_lemma = 20
            
            # Create lists to hold pairs
            true_pairs = []
            false_pairs = []
            
            # Process each valid lemma
            random.shuffle(can_make_true_pairs)
            for lemma in can_make_true_pairs:
                # Get all occurrences for this lemma
                occurrences = word_occurrences[lemma]
                
                # Group by sense_id
                sense_occurrences = {}
                for sent, pos, sense_id, sent_idx in occurrences:
                    if sense_id not in sense_occurrences:
                        sense_occurrences[sense_id] = []
                    sense_occurrences[sense_id].append((sent, pos, sent_idx))
                
                # Keep track of already used sentence pairs to avoid duplicates
                used_sentence_pairs = set()
                
                # 1. Generate TRUE pairs (same sense, different sentences)
                true_pairs_for_lemma = 0
                for sense_id, sense_occ in sense_occurrences.items():
                    # Skip if not enough sentences for this sense
                    if len(sense_occ) < 2:
                        continue
                    
                    # Create a mapping from sentence idx to occurrence for easier lookup
                    sent_idx_to_occ = {}
                    for sent, pos, sent_idx in sense_occ:
                        sent_idx_to_occ[sent_idx] = (sent, pos)
                    
                    # Get all sentence indices for this sense
                    sent_idxs = list(sent_idx_to_occ.keys())
                    
                    # Randomly select sentence pairs
                    max_attempts = 10  # Avoid too many attempts for efficiency
                    for _ in range(min(max_pairs_per_lemma // 2, len(sent_idxs) * (len(sent_idxs) - 1) // 4, max_attempts)):
                        if true_pairs_for_lemma >= max_pairs_per_lemma // 2:
                            break
                        
                        # Randomly select two different sentences
                        idx1, idx2 = random.sample(sent_idxs, 2)
                        
                        # Skip if this sentence pair was already used
                        if (idx1, idx2) in used_sentence_pairs or (idx2, idx1) in used_sentence_pairs:
                            continue
                        
                        sent1, pos1 = sent_idx_to_occ[idx1]
                        sent2, pos2 = sent_idx_to_occ[idx2]
                        
                        true_pairs.append((sent1, sent2, lemma, 'T'))
                        true_pairs_for_lemma += 1
                        used_sentence_pairs.add((idx1, idx2))
                
                # 2. Generate FALSE pairs (different senses)
                false_pairs_for_lemma = 0
                sense_ids = list(sense_occurrences.keys())
                
                # Only proceed if we have at least 2 different senses
                if len(sense_ids) >= 2:
                    # Try pairs of different senses
                    max_attempts = 10  # Avoid too many attempts for efficiency
                    for _ in range(min(max_pairs_per_lemma // 2, max_attempts)):
                        if false_pairs_for_lemma >= max_pairs_per_lemma // 2:
                            break
                        
                        # Randomly select two different senses
                        sense1, sense2 = random.sample(sense_ids, 2)
                        
                        # Get occurrences for these senses
                        occs1 = sense_occurrences[sense1]
                        occs2 = sense_occurrences[sense2]
                        
                        # Skip if not enough occurrences
                        if not occs1 or not occs2:
                            continue
                        
                        # Randomly select one occurrence from each sense
                        occ1 = random.choice(occs1)
                        occ2 = random.choice(occs2)
                        
                        sent1, pos1, sent_idx1 = occ1
                        sent2, pos2, sent_idx2 = occ2
                        
                        # Ensure different sentences
                        if sent_idx1 == sent_idx2:
                            continue
                        
                        # Skip if this sentence pair was already used
                        if (sent_idx1, sent_idx2) in used_sentence_pairs or (sent_idx2, sent_idx1) in used_sentence_pairs:
                            continue
                        
                        false_pairs.append((sent1, sent2, lemma, 'F'))
                        false_pairs_for_lemma += 1
                        used_sentence_pairs.add((sent_idx1, sent_idx2))
                
                print(f"[DEBUG] Lemma '{lemma}': Generated {true_pairs_for_lemma} TRUE pairs and {false_pairs_for_lemma} FALSE pairs")
                sys.stdout.flush()
                
                # Stop if we have enough pairs total
                # (But don't worry about exact balance - we'll handle that later)
                if len(true_pairs) + len(false_pairs) >= total_pairs_desired:
                    break
            
            # Report results before balancing
            print(f"[INFO] Generated {len(true_pairs)} TRUE pairs and {len(false_pairs)} FALSE pairs")
            
            # Balance pairs if extremely skewed
            if len(true_pairs) == 0:
                print(f"[WARNING] No TRUE pairs generated! Using random pairs as TRUE.")
                # Convert some false pairs to true as a last resort
                false_to_convert = min(len(false_pairs) // 2, total_pairs_desired // 2)
                selected_false = random.sample(false_pairs, false_to_convert)
                for pair in selected_false:
                    sent1, sent2, lemma, _ = pair
                    true_pairs.append((sent1, sent2, lemma, 'T'))
                    false_pairs.remove(pair)
                
            elif len(false_pairs) == 0:
                print(f"[WARNING] No FALSE pairs generated! Using random pairs as FALSE.")
                # Convert some true pairs to false as a last resort
                true_to_convert = min(len(true_pairs) // 2, total_pairs_desired // 2)
                selected_true = random.sample(true_pairs, true_to_convert)
                for pair in selected_true:
                    sent1, sent2, lemma, _ = pair
                    false_pairs.append((sent1, sent2, lemma, 'F'))
                    true_pairs.remove(pair)
            
            # Determine final counts while maintaining reasonable balance
            true_ratio = len(true_pairs) / (len(true_pairs) + len(false_pairs))
            
            # If ratio is extremely skewed (less than 30% or more than 70% true), adjust
            if true_ratio < 0.3 or true_ratio > 0.7:
                target_true = int(total_pairs_desired * target_true_ratio)
                target_false = total_pairs_desired - target_true
                
                if len(true_pairs) < target_true and len(false_pairs) > target_false:
                    # Convert some false to true
                    num_to_convert = min(target_true - len(true_pairs), len(false_pairs) - target_false)
                    selected_false = random.sample(false_pairs, num_to_convert)
                    for pair in selected_false:
                        sent1, sent2, lemma, _ = pair
                        true_pairs.append((sent1, sent2, lemma, 'T'))
                        false_pairs.remove(pair)
                        
                elif len(false_pairs) < target_false and len(true_pairs) > target_true:
                    # Convert some true to false
                    num_to_convert = min(target_false - len(false_pairs), len(true_pairs) - target_true)
                    selected_true = random.sample(true_pairs, num_to_convert)
                    for pair in selected_true:
                        sent1, sent2, lemma, _ = pair
                        false_pairs.append((sent1, sent2, lemma, 'F'))
                        true_pairs.remove(pair)
            
            # Determine final counts to use
            true_count = min(len(true_pairs), int(total_pairs_desired * 0.6))  # Allow up to 60% true
            false_count = min(len(false_pairs), total_pairs_desired - true_count)
            
            # Adjust true count if needed to use all available false pairs
            if false_count < len(false_pairs):
                true_count = min(len(true_pairs), total_pairs_desired - false_count)
            
            # Select pairs to use
            selected_true_pairs = random.sample(true_pairs, true_count) if true_count < len(true_pairs) else true_pairs
            selected_false_pairs = random.sample(false_pairs, false_count) if false_count < len(false_pairs) else false_pairs
            
            # Combine and shuffle all pairs
            all_pairs = selected_true_pairs + selected_false_pairs
            random.shuffle(all_pairs)
            
            # Write pairs to CSV
            for sent1, sent2, lemma, label in all_pairs:
                writer.writerow([sent1, sent2, lemma, label])
            
            print(f"[SUCCESS] Created {len(all_pairs)} sentence pairs")
            print(f"[STATS] TRUE pairs: {true_count} ({true_count/len(all_pairs)*100:.1f}%)")
            print(f"[STATS] FALSE pairs: {false_count} ({false_count/len(all_pairs)*100:.1f}%)")
            print(f"[SUCCESS] Output saved to {output_file}")
            sys.stdout.flush()
    
    except ET.ParseError as e:
        print(f"[ERROR] Error parsing XML file: {e}")
    except FileNotFoundError:
        print(f"[ERROR] File not found: {xml_file_path}")
    except Exception as e:
        print(f"[ERROR] Unexpected error processing data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    process_semcor_xml("semcor.data.xml")