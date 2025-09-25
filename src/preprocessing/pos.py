import nltk
from nltk.tag import hmm, RegexpTagger

# --- Step 1: Create a Small, Manually Tagged Corpus ---
# In a real project, this would be a much larger file.
# We are using a simple tagset: NN (Noun), VM (Verb), JJ (Adjective), PRP (Pronoun), PSP (Postposition)

# Training data: A list of sentences, where each sentence is a list of (word, tag) tuples.
train_data = [
    [('જીવવિજ્ઞાન', 'NN'), ('એક', 'JJ'), ('વિજ્ઞાન', 'NN'), ('છે', 'VM')],
    [('સજીવ', 'NN'), ('વિશ્વ', 'NN'), ('વિવિધતા', 'NN'), ('ધરાવે', 'VM')],
    [('કોષ', 'NN'), ('જીવનનો', 'NN'), ('મૂળભૂત', 'JJ'), ('એકમ', 'NN'), ('છે', 'VM')],
    [('આ', 'PRP'), ('પ્રક્રિયા', 'NN'), ('ખૂબ', 'JJ'), ('જટિલ', 'JJ'), ('છે', 'VM')],
    [('વનસ્પતિઓ', 'NN'), ('ખોરાક', 'NN'), ('બનાવે', 'VM')],
    [('તે', 'PRP'), ('પાણીમાં', 'NN'), ('રહે', 'VM')],
    [('રુધિર', 'NN'), ('શરીરમાં', 'NN'), ('ફરે', 'VM')]
]

# Testing data: This data will NOT be used for training. We will use it to evaluate the taggers.
test_data = [
    [('કોષ', 'NN'), ('એક', 'JJ'), ('એકમ', 'NN'), ('છે', 'VM')],
    [('આ', 'PRP'), ('વનસ્પતિઓ', 'NN'), ('જટિલ', 'JJ'), ('છે', 'VM')],
    [('જીવવિજ્ઞાન', 'NN'), ('એક', 'JJ'), ('વિજ્ઞાન', 'NN'), ('છે', 'VM')],
    [('તે', 'PRP'), ('શરીરમાં', 'NN'), ('રહે', 'VM')]
]

# --- Step 2: Train the HMM Tagger ---

print("Training HMM Tagger...")
hmm_trainer = hmm.HiddenMarkovModelTrainer()
# The train_supervised method calculates the transition and emission probabilities from the data.
hmm_tagger = hmm_trainer.train_supervised(train_data)
print("HMM Tagger training complete.")

# --- Step 3: Create the Rule-Based Tagger ---

print("\nCreating Rule-Based Tagger...")
# Define a few simple rules based on Gujarati suffixes.
# This is a very basic example. A real tagger would have many more rules.
patterns = [
    (r'.*માં$', 'PSP'),   # Words ending in 'માં' are postpositions
    (r'.*નો$', 'NN'),    # Words ending in 'નો' are often nouns
    (r'.*ની$', 'NN'),    # Words ending in 'ની' are often nouns
    (r'.*નું$', 'NN'),    # Words ending in 'નું' are often nouns
    (r'.*ે$', 'VM'),     # Words ending in 'ે' can be verbs
    (r'.*ો$', 'NN'),     # Words ending in 'ો' can be nouns
    (r'.*', 'NN')        # Default to Noun for anything that doesn't match
]

rule_based_tagger = RegexpTagger(patterns)
print("Rule-Based Tagger created.")


# --- Step 4: Evaluate Both Taggers ---

print("\n--- EVALUATION REPORT ---")

# Evaluate the HMM Tagger
hmm_accuracy = hmm_tagger.evaluate(test_data)
print(f"\n1. HMM Tagger Accuracy: {hmm_accuracy * 100:.2f}%")

# Evaluate the Rule-Based Tagger
rule_based_accuracy = rule_based_tagger.evaluate(test_data)
print(f"2. Rule-Based Tagger Accuracy: {rule_based_accuracy * 100:.2f}%")

# --- Step 5: Show Tagging in Action and Compare ---

print("\n--- TAGGING EXAMPLES ---")
sentence_to_tag = ['આ', 'કોષ', 'જટિલ', 'છે']
print(f"\nTest Sentence: {sentence_to_tag}")

# HMM Tagger Prediction
hmm_tagged_sentence = hmm_tagger.tag(sentence_to_tag)
print(f"HMM Prediction:      {hmm_tagged_sentence}")

# Rule-Based Tagger Prediction
rule_based_tagged_sentence = rule_based_tagger.tag(sentence_to_tag)
print(f"Rule-Based Prediction: {rule_based_tagged_sentence}")

# Gold Standard (Correct Tags)
gold_standard = [('આ', 'PRP'), ('કોષ', 'NN'), ('જટિલ', 'JJ'), ('છે', 'VM')]
print(f"Correct Tags:        {gold_standard}")


# --- Step 6: Discussion and Analysis ---

print("\n--- ANALYSIS OF EFFECTIVENESS ---")
print("""
**HMM Tagger Effectiveness:**
- **Strengths:** The HMM tagger learns from context (transition probabilities). It correctly tagged 'જટિલ' as an Adjective (JJ) because it likely saw adjectives before the verb 'છે' in the training data. It also correctly identified 'કોષ' as a Noun (NN) and 'આ' as a Pronoun (PRP) based on the emission probabilities learned from the training set.
- **Weaknesses:** Its performance is entirely dependent on the training data. If it encounters a new word like 'સુંદર' (beautiful), it would have no information and would likely fail to tag it correctly. With our tiny dataset, it's already surprisingly effective on known words.

**Rule-Based Tagger Effectiveness:**
- **Strengths:** It doesn't need any training data. It correctly tagged 'છે' as a Verb (VM) because it matched the `r'.*ે$'` rule.
- **Weaknesses:** It is very rigid and lacks context. It incorrectly tagged 'જટિલ' (complex) as a Noun (NN) because it didn't match any specific suffix rule and fell back to the default 'NN' rule. It cannot distinguish between words that share the same suffix but have different POS tags. For example, it would tag both a noun and a verb ending in 'ે' as a verb.

**Conclusion for Gujarati:**
For a local language like Gujarati, neither method is perfect on its own.
- The **HMM approach is more powerful and scalable**, but it requires a significant investment in creating a large, manually tagged corpus. Its accuracy will grow as the corpus grows.
- The **Rule-based approach provides a good baseline** and can be useful for bootstrapping a larger corpus. You can use it to automatically tag a large amount of text and then have human annotators correct the mistakes, which is often faster than tagging from scratch.
- A **hybrid approach** is often the most effective strategy. NLTK allows you to chain taggers together, for example, using a rule-based tagger as a backoff for an HMM tagger to handle unknown words.
""")