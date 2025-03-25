import os
import json

def load_dataset(base_path):
    """
    Load the dataset from the directory structure.
    Returns a dictionary with data for each level and split.
    Also prints the number of files processed and any errors encountered.
    """
    levels = ['easy', 'medium', 'hard']
    splits = ['train', 'validation']
    dataset = {}

    for level in levels:
        dataset[level] = {}
        txt_count = 0
        json_count = 0
        
        for split in splits:
            split_path = os.path.join(base_path, level, split)
            documents = []
            
            # Iterate over files in the split directory
            for filename in sorted(os.listdir(split_path)):
                if filename.startswith('problem-') and filename.endswith('.txt'):
                    problem_id = filename.split('.')[0]  # e.g., 'problem-1'
                    txt_path = os.path.join(split_path, filename)
                    json_path = os.path.join(split_path, f'truth-{problem_id}.json')
                    
                    try:
                        # Load text file
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            sentences = [line.strip() for line in f.readlines() if line.strip()]
                        txt_count += 1  # Increment .txt file count
                        
                        # Load ground truth JSON
                        with open(json_path, 'r', encoding='utf-8') as f:
                            truth = json.load(f)
                            changes = truth['changes']
                        json_count += 1  # Increment .json file count
                        
                        # Store as tuple: (sentences, changes)
                        documents.append((sentences, changes))
                    
                    except Exception as e:
                        # Print error message and skip the file
                        print(f"Error processing file: {filename}, Level: {level}. Details: {e}")
            
            dataset[level][split] = documents
        
        print(f"Level '{level}': Processed {txt_count} .txt files and {json_count} .json files.")
    
    return dataset
