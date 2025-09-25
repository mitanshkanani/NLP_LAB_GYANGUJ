import re

def is_valid_gujarati_word(word):
    """
    Checks if a word is a valid token. A valid token must contain
    ONLY Gujarati characters. This is a powerful filter for OCR noise.
    """
    pattern = re.compile(r'^[\u0A80-\u0AFE]+$')
    return pattern.match(word) is not None

def final_preprocess_and_save(input_file_path, output_file_path):
    """
    A robust, final preprocessing pipeline for OCR'd Gujarati text.
    It extracts core content first, then performs intelligent word-level cleaning,
    and saves the result to a file.
    """
    print(f"Reading from: {input_file_path}")
    with open(input_file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # --- Stage 1: Precise Content Extraction ---
    try:
        start_marker = "જીવવિજ્ઞાન એ જૈવ સ્વરૂપો અને સજીવોની ક્રિયાવિધિનું વિજ્ઞાન છે"
        end_marker = "સ્વાધ્યાય"
        
        start_index = raw_text.index(start_marker)
        end_index = raw_text.rindex(end_marker)
        
        core_content = raw_text[start_index:end_index]
        print("Successfully extracted core content.")
    except ValueError:
        print("Could not find precise start/end markers. Processing the whole file.")
        core_content = raw_text

    # --- Stage 2: Clean and Normalize the Core Content ---
    print("Starting cleaning and normalization...")
    text = re.sub(r'\s+', ' ', core_content).strip()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[^\u0A80-\u0AFE\s.।]', '', text)
    
    initial_words = text.split(' ')
    cleaned_words = [word for word in initial_words if is_valid_gujarati_word(word)]
    cleaned_text = ' '.join(cleaned_words)
    print("Cleaning complete.")

    # --- Stage 3: Segmentation and Stop Word Removal ---
    print("Segmenting into sentences and removing stop words...")
    sentences = re.split(r'[.।]', cleaned_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    stop_words = set([
        'અને', 'આ', 'એ', 'છે', 'માં', 'થી', 'તો', 'જ', 'ને', 'માટે', 'હોય', 'પણ', 'તે', 'કે'
    ])
    
    # --- Stage 4: Save the Processed Data to a File ---
    print(f"Saving processed data to: {output_file_path}")
    count = 0
    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        for sentence in sentences:
            tokens = sentence.split()
            processed_tokens = [word for word in tokens if word not in stop_words]
            if processed_tokens:
                processed_line = ' '.join(processed_tokens)
                f_out.write(processed_line + '\n')
                count += 1
    
    print("-" * 50)
    print(f"Processing complete!")
    print(f"Successfully saved {count} processed sentences to '{output_file_path}'.")
    print("-" * 50)


# --- RUN THE FINAL SCRIPT ---
input_file = '../../data/processed/class11_biology.txt'
output_file = '../../data/next/class11_biology.txt'

final_preprocess_and_save(input_file, output_file)