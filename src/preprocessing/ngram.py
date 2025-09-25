import os
import glob
import nltk
from nltk.util import ngrams
from collections import Counter

# It's good practice to ensure the necessary NLTK data is downloaded.
# The 'punkt' package is used for tokenization.
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading NLTK's 'punkt' model...")
    nltk.download('punkt')

def load_tokens_from_folder(folder_path):
    """
    Reads all .txt files from a folder and returns a single list of all words.
    """
    all_tokens = []
    # Use glob to find all files ending with .txt
    file_paths = glob.glob(os.path.join(folder_path, '*.txt'))
    
    if not file_paths:
        print(f"Error: No .txt files found in the folder '{folder_path}'.")
        return None

    print(f"Found {len(file_paths)} files to analyze.")
    
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Split each line into words and add them to the master list
                all_tokens.extend(line.strip().split())
                
    return all_tokens

def analyze_unigrams(tokens, top_n=50):
    """
    Analyzes unigram frequencies and probabilities.
    """
    report = ["="*20 + " UNIGRAM ANALYSIS " + "="*20]
    report.append(f"Total words (tokens): {len(tokens)}")
    report.append(f"Unique words (vocabulary): {len(set(tokens))}\n")
    
    fdist = nltk.FreqDist(tokens)
    
    report.append(f"--- Top {top_n} Most Common Words ---\n")
    report.append(f"{'Rank':<5} {'Word':<20} {'Count':<10} {'Probability (%)':<15}")
    report.append("-" * 60)
    
    for i, (word, count) in enumerate(fdist.most_common(top_n), 1):
        probability = (count / len(tokens)) * 100
        report.append(f"{i:<5} {word:<20} {count:<10} {probability:.4f}%")
        
    return "\n".join(report)

def analyze_bigrams(tokens, top_n=50):
    """
    Analyzes bigram frequencies and conditional probabilities.
    """
    report = ["\n" + "="*20 + " BIGRAM ANALYSIS " + "="*20]
    bigram_list = list(ngrams(tokens, 2))
    report.append(f"Total bigrams: {len(bigram_list)}\n")
    
    fdist = nltk.FreqDist(bigram_list)
    
    report.append(f"--- Top {top_n} Most Common Word Pairs ---\n")
    report.append(f"{'Rank':<5} {'Bigram':<30} {'Count':<10} {'Probability (%)':<15}")
    report.append("-" * 70)
    
    for i, (bigram, count) in enumerate(fdist.most_common(top_n), 1):
        probability = (count / len(bigram_list)) * 100
        bigram_str = ' '.join(bigram)
        report.append(f"{i:<5} {bigram_str:<30} {count:<10} {probability:.4f}%")
        
    return "\n".join(report)

def analyze_trigrams(tokens, top_n=50):
    """
    Analyzes trigram frequencies and conditional probabilities.
    """
    report = ["\n" + "="*20 + " TRIGRAM ANALYSIS " + "="*20]
    trigram_list = list(ngrams(tokens, 3))
    report.append(f"Total trigrams: {len(trigram_list)}\n")
    
    fdist = nltk.FreqDist(trigram_list)
    
    report.append(f"--- Top {top_n} Most Common Word Sequences ---\n")
    report.append(f"{'Rank':<5} {'Trigram':<40} {'Count':<10} {'Probability (%)':<15}")
    report.append("-" * 80)
    
    for i, (trigram, count) in enumerate(fdist.most_common(top_n), 1):
        probability = (count / len(trigram_list)) * 100
        trigram_str = ' '.join(trigram)
        report.append(f"{i:<5} {trigram_str:<40} {count:<10} {probability:.4f}%")
        
    return "\n".join(report)


# --- Main Execution ---
if __name__ == "__main__":
    # Assuming your preprocessed files are in a subfolder named 'processed_data'
    # If they are in the same folder as the script, use '.'
    input_folder = '../../data/processed' 
    output_report_file = '../../data/ngram_analysis_report.txt'
    output_file = output_report_file
    # 1. Load all tokens from the preprocessed files
    all_tokens = load_tokens_from_folder(input_folder)
    
    if all_tokens:
        print(f"Successfully loaded {len(all_tokens)} tokens.")
        
        # 2. Perform analysis for each N-gram type
        unigram_report = analyze_unigrams(all_tokens)
        bigram_report = analyze_bigrams(all_tokens)
        trigram_report = analyze_trigrams(all_tokens)
        
        # 3. Combine reports and save to a file
        final_report = f"""
N-GRAM ANALYSIS REPORT
=======================
Dataset Folder: {os.path.abspath(input_folder)}

{unigram_report}

{bigram_report}

{trigram_report}
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_report)
            
        print(f"\n✅ Analysis complete. Report saved to '{output_file}'.")