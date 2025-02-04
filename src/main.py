from src.data_loader import load_data, save_data
from src.duplicate_checker import detect_duplicates
from src.tag_generator import assign_tags

def main():
    input_file = input("Enter the path to your CSV file: ")

    # Load dataset
    df = load_data(input_file)
    print(f"Processing {len(df)} questions...")

    # Process data
    df = detect_duplicates(df)
    df = assign_tags(df)

    # Save processed data
    output_path = save_data(df, input_file)

    # Print summary
    print("\nSummary:")
    print(f"Total questions processed: {len(df)}")
    print(f"Duplicate questions found: {df['duplicate'].sum()}")

if __name__ == "__main__":
    main()
