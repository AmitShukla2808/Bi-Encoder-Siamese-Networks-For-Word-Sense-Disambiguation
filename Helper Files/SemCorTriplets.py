import xml.etree.ElementTree as ET
import csv
import random
import time
import sys
import os
import re
import nltk
from collections import Counter

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.tokenize import word_tokenize

def get_text(elem):
    """
    Recursively extract all text content from an XML element, including its children and tails.
    """
    text = elem.text or ''
    for child in elem:
        text += get_text(child)
        text += child.tail or ''
    return text


def create_sentence_variation(sentence, target_word=None):
    """
    Create a variation of the given sentence by applying one or more
    transformation techniques while preserving meaning.
    
    Parameters:
    - sentence: The original sentence
    - target_word: Optional word to preserve in the variation
    
    Returns:
    - A modified version of the sentence
    """
    # Tokenize the sentence
    tokens = word_tokenize(sentence)
    
    # Choose a random transformation to apply
    transformation = random.choice([
        'add_punctuation', 
        'change_determiners',
        'add_filler_words',
        'reorder_clauses'
    ])
    
    if transformation == 'add_punctuation':
        # Add or change punctuation
        if not sentence.endswith(('.', '!', '?')):
            sentence = sentence + random.choice(['.', '!', '?'])
        else:
            # Replace ending punctuation
            ending_punct = random.choice(['.', '!', '?'])
            sentence = sentence[:-1] + ending_punct
    
    elif transformation == 'change_determiners':
        # Replace determiners (a/an/the)
        determiners = {'a': 'the', 'an': 'the', 'the': 'a'}
        new_tokens = []
        for token in tokens:
            if token.lower() in determiners and random.random() < 0.7:
                new_token = determiners[token.lower()]
                if token[0].isupper():
                    new_token = new_token.capitalize()
                new_tokens.append(new_token)
            else:
                new_tokens.append(token)
        sentence = ' '.join(new_tokens)
    
    elif transformation == 'add_filler_words':
        # Add filler words at the beginning or middle
        filler_beginnings = [
            'Well, ', 'Indeed, ', 'Actually, ', 'Certainly, ', 'Of course, ',
            'In fact, ', 'Basically, ', 'So, ', 'You see, ', 'Notably, '
        ]
        
        # Add at beginning with 70% probability, otherwise insert in middle
        if random.random() < 0.7:
            filler = random.choice(filler_beginnings)
            sentence = filler + sentence[0].lower() + sentence[1:]
        else:
            # Add in the middle after a comma or conjunction
            mid_fillers = [' actually', ' basically', ' in fact', ' indeed', ' certainly']
            comma_positions = [m.start() for m in re.finditer(r',', sentence)]
            if comma_positions:
                pos = random.choice(comma_positions)
                filler = random.choice(mid_fillers)
                sentence = sentence[:pos+1] + filler + sentence[pos+1:]
    
    elif transformation == 'reorder_clauses':
        # Split on commas and try to reorder parts that make sense
        parts = sentence.split(',')
        if len(parts) >= 3:
            # Swap two adjacent parts that aren't at the beginning or end
            i = random.randint(1, len(parts) - 2)
            parts[i], parts[i+1] = parts[i+1], parts[i]
            sentence = ','.join(parts)
    
    return sentence


# Function to generate a new sentence using the lemma with the same sense
def create_new_sentence_for_lemma(lemma, sense):
    """
    Create a completely new sentence that uses the given lemma with the same sense.
    This is used when we can't find different natural examples from the dataset.
    
    Parameters:
    - lemma: The lemma to use in the sentence
    - sense: The sense ID to preserve
    
    Returns:
    - A newly generated sentence using the lemma
    """
    # Define sentence templates for different lemmas
    # These templates ensure the lemma is used with a similar meaning to the original sense
    templates = [
        f"The {lemma} was clearly visible in the distance.",
        f"They discussed the {lemma} at length during the meeting.",
        f"I've never seen such a remarkable {lemma} before.",
        f"The professor explained the concept of {lemma} to the students.",
        f"Several examples of {lemma} were presented in the paper.",
        f"The community values this type of {lemma} highly.",
        f"The {lemma} can be found in many different contexts.",
        f"Researchers have studied this {lemma} for many years.",
        f"This particular {lemma} has unique characteristics.",
        f"The history of this {lemma} dates back centuries."
    ]
    
    # For verbs, use different templates
    verb_templates = [
        f"They {lemma} frequently when the situation calls for it.",
        f"I {lemma} whenever I have the opportunity.",
        f"She would {lemma} if given the chance.",
        f"The experts {lemma} in a very specific way.",
        f"You should {lemma} carefully in such circumstances.",
        f"We {lemma} according to the established guidelines.",
        f"The students {lemma} as part of their assignment.",
        f"To {lemma} effectively requires practice.",
        f"They have {lemma} for years with great success.",
        f"Sometimes you must {lemma} to achieve the best results."
    ]
    
    # For adjectives, use different templates
    adj_templates = [
        f"The {lemma} nature of the problem was challenging.",
        f"It was a {lemma} solution to a complex issue.",
        f"The results were quite {lemma} compared to expectations.",
        f"A {lemma} approach might yield better outcomes.",
        f"The {lemma} characteristics were immediately noticeable.",
        f"Their strategy was {lemma} but effective.",
        f"We need something more {lemma} for this situation.",
        f"The {lemma} quality makes it stand out.",
        f"A remarkably {lemma} example was presented.",
        f"It's the most {lemma} case I've ever encountered."
    ]
    
    # Try to determine if the lemma is a verb or adjective
    # This is a simple heuristic - in a real implementation you would use POS tagging
    if lemma.endswith(('e', 'ate', 'ize', 'ise', 'fy', 'en')):
        # Likely a verb
        templates = verb_templates
    elif lemma.endswith(('al', 'ful', 'ic', 'ive', 'less', 'ous', 'able', 'ible')):
        # Likely an adjective
        templates = adj_templates
    
    # Select a random template and return it without adding the sense ID
    return random.choice(templates)


def main():
    # Input SemCor XML and output CSV file paths
    xml_file = 'semcor.data.xml'
    csv_file = 'semcor_triplets.csv'
    
    start_time = time.time()
    
    print(f"[INFO] Starting SemCor XML processing for triplets...")
    print(f"[INFO] Input file: {xml_file}")
    print(f"[INFO] Output will be saved to: {csv_file}")
    
    # Dictionary to store words and their occurrences across sentences
    # Structure: {lemma: {sense_id: [(sentence, position_in_sentence, sentence_idx), ...], ...}, ...}
    lemma_sense_groups = {}
    
    # Parse the SemCor XML file
    try:
        print(f"[INFO] Parsing XML file: {xml_file}")
        parse_start = time.time()
        tree = ET.parse(xml_file)
        root = tree.getroot()
        parse_end = time.time()
        print(f"[SUCCESS] XML parsing complete in {parse_end - parse_start:.2f} seconds")
        sys.stdout.flush()
        
        # Process sentences and collect word occurrences
        # First check if we're dealing with semcor.data.xml standard format
        sentences = root.findall('.//sentence')
        
        # If no sentences found with that tag, try the other format (SemCor extracted files)
        if not sentences:
            sentences = root.findall('.//s')
            
        total_sentences = len(sentences)
        print(f"[INFO] Found {total_sentences} sentences to process")
        sys.stdout.flush()
        
        processed_instances = 0
        unique_lemmas = set()
        
        # Track senses per lemma to identify ambiguous lemmas
        lemma_senses = {}
        
        # Process sentences and collect word occurrences
        for sentence_idx, sentence in enumerate(sentences):
            words = []
            instances_in_sentence = 0
            
            # Extract words and their information
            for elem in sentence:
                # Handle both formats: instance tags directly in sentence or wf/instance tags
                if elem.tag == 'instance':
                    words.append(elem.text)
                    
                    lemma = elem.get('lemma')
                    # Try different attribute names for the sense ID
                    sense_id = elem.get('id') or elem.get('sense') or elem.get('wnsn') or elem.get('answer')
                    
                    if lemma and sense_id:
                        instances_in_sentence += 1
                        processed_instances += 1
                        position = len(words)-1
                        
                        # Use get_text to extract the full sentence text
                        sentence_text = ' '.join(word for word in sentence.itertext() if word.strip())
                        
                        # Group by lemma and sense
                        if lemma not in lemma_sense_groups:
                            lemma_sense_groups[lemma] = {}
                            unique_lemmas.add(lemma)
                            lemma_senses[lemma] = set()
                        
                        if sense_id not in lemma_sense_groups[lemma]:
                            lemma_sense_groups[lemma][sense_id] = []
                        
                        lemma_sense_groups[lemma][sense_id].append((sentence_text, position, sentence_idx))
                        lemma_senses[lemma].add(sense_id)
                
                elif elem.tag == 'wf':
                    words.append(elem.text)
            
            # Print progress
            if sentence_idx % 1000 == 0 or sentence_idx == total_sentences - 1:
                elapsed = time.time() - start_time
                progress = (sentence_idx + 1) / total_sentences * 100
                print(f"[PROGRESS] Processed {sentence_idx + 1}/{total_sentences} sentences ({progress:.1f}%) in {elapsed:.2f}s")
                sys.stdout.flush()
        
        print(f"[INFO] Processing complete!")
        print(f"[STATS] Processed {processed_instances} tagged instances")
        print(f"[STATS] Found {len(unique_lemmas)} unique lemmas")
        
        # Filter for ambiguous lemmas (those with multiple distinct senses)
        ambiguous_lemmas = [lemma for lemma, senses in lemma_senses.items() if len(senses) >= 2]
        print(f"[STATS] Found {len(ambiguous_lemmas)} lemmas with multiple senses")
        
        # Shuffle ambiguous lemmas for random selection
        random.shuffle(ambiguous_lemmas)
        
        # Debug: Check some sample ambiguous lemmas
        if ambiguous_lemmas:
            sample_lemmas = random.sample(ambiguous_lemmas, min(5, len(ambiguous_lemmas)))
            print(f"[DEBUG] Sample ambiguous lemmas: {sample_lemmas}")
            for lemma in sample_lemmas:
                print(f"[DEBUG] Lemma '{lemma}' has {len(lemma_senses[lemma])} senses")
                for sense_id, examples in lemma_sense_groups[lemma].items():
                    print(f"[DEBUG]   - Sense '{sense_id}': {len(examples)} examples")
        
        # Find lemmas with multiple examples per sense
        lemmas_with_multiple_examples_per_sense = []
        for lemma in ambiguous_lemmas:
            senses = lemma_sense_groups[lemma]
            # Check if at least one sense has at least 2 examples (for anchor and positive)
            if any(len(examples) >= 2 for examples in senses.values()):
                lemmas_with_multiple_examples_per_sense.append(lemma)
        
        print(f"[STATS] Found {len(lemmas_with_multiple_examples_per_sense)} lemmas with at least one sense having multiple examples")
        
        # Shuffle lemmas with multiple examples per sense for random selection
        random.shuffle(lemmas_with_multiple_examples_per_sense)
        
        # Build triplets: (lemma, positive, negative, anchor)
        triplets = []
        natural_variation_count = 0  # Track how many triplets have natural variations from the dataset
        synthetic_variation_count = 0  # Track how many triplets have generated variations
        generated_anchor_count = 0   # Track how many triplets have generated anchor sentences
        
        # First, process lemmas that have senses with multiple examples (if any)
        print(f"[INFO] Creating triplets from natural variations in the dataset...")
        for lemma in lemmas_with_multiple_examples_per_sense:
            senses = lemma_sense_groups[lemma]
            
            # Find senses with multiple examples
            multi_example_senses = [sense_id for sense_id, examples in senses.items() if len(examples) >= 2]
            
            for anchor_sense in multi_example_senses:
                anchor_examples = senses[anchor_sense]
                
                # Check if there are other senses with examples to use as negative samples
                other_senses = [s for s in senses.keys() if s != anchor_sense and senses[s]]
                if not other_senses:
                    continue
                    
                # If we have multiple examples, try to create natural variations
                if len(anchor_examples) >= 2:
                    # Get distinct examples for this sense
                    unique_examples = []
                    seen_texts = set()
                    
                    for ex in anchor_examples:
                        sentence_text = ex[0]
                        if sentence_text not in seen_texts:
                            unique_examples.append(ex)
                            seen_texts.add(sentence_text)
                    
                    # If we still have at least 2 unique examples after deduplication
                    if len(unique_examples) >= 2:
                        # Limit to a reasonable number of triplets per sense
                        num_triplets = min(5, len(unique_examples))
                        
                        # Create pairs of distinct examples (positive, anchor)
                        selected_examples = random.sample(unique_examples, min(num_triplets+1, len(unique_examples)))
                        
                        # If we have enough examples, use them directly
                        for i in range(min(num_triplets, len(selected_examples)-1)):
                            positive_tuple = selected_examples[i]
                            positive = positive_tuple[0]  # Get the sentence text
                            
                            # Use a different example for anchor
                            anchor_options = [selected_examples[j] for j in range(len(selected_examples)) if j != i]
                            if anchor_options:
                                anchor_tuple = random.choice(anchor_options)
                                anchor = anchor_tuple[0]  # Get the sentence text
                                
                                # Extra verification to ensure they're different
                                if positive == anchor:
                                    # Generate a completely new anchor sentence
                                    anchor = create_new_sentence_for_lemma(lemma, anchor_sense)
                                    generated_anchor_count += 1
                                
                                # Get a negative example from a different sense
                                negative_sense = random.choice(other_senses)
                                negative_examples = senses[negative_sense]
                                if negative_examples:
                                    negative_tuple = random.choice(negative_examples)
                                    negative = negative_tuple[0]  # Get the sentence text
                                    
                                    # Add this triplet
                                    triplets.append((lemma, positive, negative, anchor))
                                    natural_variation_count += 1
                    else:
                        # We have multiple entries but with the same text (duplicates)
                        # Use one example as positive and create a new anchor
                        positive_tuple = unique_examples[0]
                        positive = positive_tuple[0]
                        
                        # Generate a completely new anchor sentence
                        anchor = create_new_sentence_for_lemma(lemma, anchor_sense)
                        generated_anchor_count += 1
                        
                        # Get a negative example from a different sense
                        negative_sense = random.choice(other_senses)
                        negative_examples = senses[negative_sense]
                        if negative_examples:
                            negative_tuple = random.choice(negative_examples)
                            negative = negative_tuple[0]  # Get the sentence text
                            
                            # Add this triplet
                            triplets.append((lemma, positive, negative, anchor))
                            natural_variation_count += 1
        
        print(f"[INFO] Created {natural_variation_count} triplets with natural variations from the dataset")
        print(f"[INFO] Generated {generated_anchor_count} new anchor sentences when natural variations weren't available")
        
        # Determine how many additional triplets to create with synthetic variations
        target_triplets = 10000
        remaining_triplets = max(0, target_triplets - natural_variation_count)
        
        if remaining_triplets > 0:
            print(f"[INFO] Creating {remaining_triplets} additional triplets with synthetic variations...")
            
            # Use all ambiguous lemmas for synthetic variations
            # First use lemmas that don't have multiple examples of the same sense
            remaining_lemmas = [lemma for lemma in ambiguous_lemmas if lemma not in lemmas_with_multiple_examples_per_sense]
            random.shuffle(remaining_lemmas)
            
            # Add lemmas with multiple examples per sense at the end of the list
            # to prioritize using the other lemmas first
            for lemma in lemmas_with_multiple_examples_per_sense:
                if lemma not in remaining_lemmas:
                    remaining_lemmas.append(lemma)
            
            # Create synthetic triplets
            synthetic_count = 0
            for lemma in remaining_lemmas:
                if synthetic_count >= remaining_triplets:
                    break
                    
                senses = lemma_sense_groups[lemma]
                
                # Need at least 2 different senses with examples
                valid_senses = [s for s, examples in senses.items() if examples]
                if len(valid_senses) < 2:
                    continue
                
                # Randomly select two different senses
                random.shuffle(valid_senses)
                anchor_sense = valid_senses[0]
                anchor_examples = senses[anchor_sense]
                
                # Get negative examples from different senses
                negative_senses = [s for s in valid_senses if s != anchor_sense]
                if not negative_senses:
                    continue
                    
                negative_sense = random.choice(negative_senses)
                negative_examples = senses[negative_sense]
                
                if not anchor_examples or not negative_examples:
                    continue
                
                # Select a random example for positive sentence
                positive_tuple = random.choice(anchor_examples)
                positive = positive_tuple[0]  # Get the sentence text
                
                # Generate a completely new anchor sentence
                anchor = create_new_sentence_for_lemma(lemma, anchor_sense)
                generated_anchor_count += 1
                
                # Select a random negative example
                negative_tuple = random.choice(negative_examples)
                negative = negative_tuple[0]  # Get the sentence text
                
                # Add this triplet
                triplets.append((lemma, positive, negative, anchor))
                synthetic_count += 1
                synthetic_variation_count += 1
                
                # Print progress periodically
                if synthetic_count % 500 == 0:
                    print(f"[PROGRESS] Created {synthetic_count}/{remaining_triplets} synthetic triplets")
        
        total_triplets = len(triplets)
        print(f"[INFO] Created {total_triplets} total triplets")
        print(f"[INFO] {natural_variation_count} triplets have natural variations from the dataset ({natural_variation_count/total_triplets*100:.1f}% of total)")
        print(f"[INFO] {synthetic_variation_count} triplets have synthetic variations ({synthetic_variation_count/total_triplets*100:.1f}% of total)")
        print(f"[INFO] {generated_anchor_count} triplets have generated anchor sentences ({generated_anchor_count/total_triplets*100:.1f}% of total)")
        
        # Limit the number of triplets if there are too many
        max_triplets = 10000
        if len(triplets) > max_triplets:
            print(f"[INFO] Limiting to {max_triplets} randomly selected triplets")
            # Shuffle before sampling to ensure randomness
            random.shuffle(triplets)
            triplets = triplets[:max_triplets]

        # Write out to CSV
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Lemma', 'PositiveSense', 'NegativeSense', 'Anchor'])
            for lemma, positive, negative, anchor in triplets:
                writer.writerow([lemma, positive, negative, anchor])
        
        print(f"[SUCCESS] Output saved to {csv_file}")
        sys.stdout.flush()
    
    except ET.ParseError as e:
        print(f"[ERROR] Error parsing XML file: {e}")
    except FileNotFoundError:
        print(f"[ERROR] File not found: {xml_file}")
    except Exception as e:
        print(f"[ERROR] Unexpected error processing data: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()