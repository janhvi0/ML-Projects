import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from keybert import KeyBERT
import os

# Load required models
nlp = spacy.load('en_core_web_md')
keybert_model = KeyBERT()
tfidf = TfidfVectorizer(stop_words='english')

def find_duplicates(questions):
    """
    Identify duplicate questions using TF-IDF and cosine similarity.
    Returns a boolean array indicating duplicate status.
    """
    # Convert questions to TF-IDF vectors
    tfidf_matrix = tfidf.fit_transform(questions)
    
    # Calculate cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix)
    
    # Mask the diagonal to ignore self-similarity
    np.fill_diagonal(cosine_sim, 0)
    
    # Consider questions with similarity > 0.85 as duplicates
    is_duplicate = (cosine_sim > 0.85).any(axis=1)
    
    return is_duplicate

def generate_tags(question):
    """
    Generate relevant tags for a question using KeyBERT and spaCy.
    Returns a list of tags.
    """
    # Extract keywords using KeyBERT
    keywords = keybert_model.extract_keywords(
        question, 
        keyphrase_ngram_range=(1, 2),
        stop_words='english',
        top_n=3
    )
    
    # Process with spaCy for additional entity recognition
    doc = nlp(question)
    
    # Extract named entities and technical terms
    entities = [ent.text.lower() for ent in doc.ents]
    
    # Combine and clean tags
    all_tags = [kw[0] for kw in keywords] + entities
    
    # Remove duplicates and clean tags
    clean_tags = list(set([
        tag.strip() for tag in all_tags 
        if len(tag.strip()) > 2  # Remove very short tags
    ]))
    
    return clean_tags[:5]  # Limit to top 5 tags

def process_dataset(input_file):
    """
    Process the dataset and save enhanced version in the same directory.
    """
    # Get the directory of the input file
    input_dir = os.path.dirname(os.path.abspath(input_file))
    
    # Read the dataset
    df = pd.read_csv(input_file)
    print(f"Processing {len(df)} questions...")
    
    # Assuming the questions are in a column named 'question'
    questions = df['Questions'].tolist()
    
    # Find duplicates
    print("Finding duplicates...")
    df['duplicate'] = find_duplicates(questions)
    
    # Generate tags
    print("Generating tags...")
    tags = []
    
    for i, question in enumerate(questions, 1):
        if i % 100 == 0:
            print(f"Processed {i}/{len(questions)} questions")
            
        # Generate tags
        question_tags = generate_tags(question)
        tags.append(','.join(question_tags))
    
    df['tags'] = tags
    
    # Create output filename
    input_filename = os.path.basename(input_file)
    output_filename = f"enhanced1_{input_filename}"
    output_path = os.path.join(input_dir, output_filename)
    
    # Save the enhanced dataset
    df.to_csv(output_path, index=False)
    print(f"\nProcessing complete! Enhanced dataset saved as: {output_filename}")
    
    # Print summary statistics
    print("\nSummary:")
    print(f"Total questions processed: {len(df)}")
    print(f"Duplicate questions found: {df['duplicate'].sum()}")

# Example usage
if __name__ == "__main__":
    # You can either hardcode the path or use input()
    input_file = input("Enter the path to your CSV file: ")
    process_dataset(input_file)