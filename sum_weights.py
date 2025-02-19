import os
import json
import glob
from collections import defaultdict

def process_json_files(input_directory, output_file):
    """
    Processes all JSON files in the input_directory, summarizes and normalizes weights,
    and writes the result to output_file.

    Parameters:
    - input_directory (str): Path to the directory containing JSON files.
    - output_file (str): Path to the output JSON file.
    """

    # Initialize a default dictionary to accumulate weights per stock ID
    total_weights = defaultdict(float)



    # Use glob to find all JSON files in the directory
    json_pattern = os.path.join(input_directory, '*.json')
    json_files = glob.glob(json_pattern)

    if not json_files:
        print(f"No JSON files found in directory: {input_directory}")
        return

    # Process each JSON file
    for file_path in json_files:
        print(f"Processing file: {file_path}")
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)

            # Ensure the JSON data is a list
            if not isinstance(data, list):
                print(f"Skipping {file_path}: JSON data is not a list.")
                continue

            # Iterate through each entry in the JSON file
            for entry in data:
                stock_id = entry.get('id')
                weight = entry.get('weight')
                operation = entry.get('operation')

                # Validate the entry
                if stock_id is None or weight is None or operation is None:
                    print(f"Skipping invalid entry in {file_path}: {entry}")
                    continue

                if operation.lower() == 'short':
                    print(f'id={stock_id} weight={weight} operation={operation}')
                    weight = -weight  # Treat short as negative weight


                total_weights[stock_id] += weight
                print(f"Stock ID: {stock_id}, weight={weight}")

        except json.JSONDecodeError:
            print(f"Error decoding JSON in file: {file_path}")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    if not total_weights:
        print("No valid weight data found across JSON files.")
        return

    # Calculate the total sum of absolute weights for normalization
    sum_abs_weights = sum(abs(w) for w in total_weights.values())
    print(f"Total sum of absolute weights: {sum_abs_weights}")
    if sum_abs_weights == 0:
        print("Total sum of absolute weights is zero. Cannot normalize.")
        return

    # Prepare the normalized data
    normalized_data = []
    for stock_id, weight in total_weights.items():
        normalized_weight = weight / sum_abs_weights
        print(f"Stock ID: {stock_id}, weight={weight} Normalized Weight: {normalized_weight}")
        if normalized_weight < 0:
            normalized_entry = {
                "id": stock_id,
                "weight": abs(normalized_weight),
                "operation": "short"
            }
        else:
            normalized_entry = {
                "id": stock_id,
                "weight": normalized_weight,
                "operation": "long"
            }

        normalized_data.append(normalized_entry)

    # Write the normalized data to the output JSON file
    try:
        with open(output_file, 'w') as outfile:
            json.dump(normalized_data, outfile, indent=2)
        print(f"Processed data has been written to {output_file}")
    except Exception as e:
        print(f"Error writing to output file {output_file}: {e}")

if __name__ == "__main__":
    # Example usage:
    # Define the input directory containing JSON files
    input_dir = "tmp/"  # Replace with your directory path

    # Define the output JSON file path
    output_json = "output.json"  # Replace with your desired output file path

    # Call the processing function
    process_json_files(input_dir, output_json)
