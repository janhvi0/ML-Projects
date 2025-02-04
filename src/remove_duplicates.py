import pandas as pd
import os

def clean_duplicates_from_file(input_file, output_dir):

    try:
       
        df = pd.read_csv(input_file)
        
        # Verify required columns exist
        required_columns = ['Questions', 'duplicate', 'tags']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("Input file must contain 'Questions', 'duplicate', and 'tags' columns")
        
        
        clean_df = df[~df['duplicate']][['Questions', 'tags']]
        
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, f"clean_{base_name}.csv")
        
        # Save clean dataset
        clean_df.to_csv(output_file, index=False)
        
        print(f"Successfully cleaned duplicates!")
        print(f"Original questions: {len(df)}")
        print(f"Questions after cleaning: {len(clean_df)}")
        print(f"File saved at: {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")


INPUT_FILE = r"C:\GenAI Complete\small_projects\dataset_cleaner\enhanced1_questions.csv"  
OUTPUT_DIR = r"C:\GenAI Complete\small_projects\dataset_cleaner" 

if __name__ == "__main__":
    clean_duplicates_from_file(INPUT_FILE, OUTPUT_DIR)