import pandas as pd
import os

def load_data(file_path):
    """Loads the dataset from a CSV file."""
    return pd.read_csv(file_path)

def save_data(df, input_file):
    """Saves the processed dataset in the same directory as input."""
    input_dir = os.path.dirname(os.path.abspath(input_file))
    input_filename = os.path.basename(input_file)
    output_filename = f"updated_{input_filename}"
    output_path = os.path.join(input_dir, output_filename)
    
    df.to_csv(output_path, index=False)
    print(f"\nProcessing complete! Enhanced dataset saved as: {output_filename}")

    return output_path
