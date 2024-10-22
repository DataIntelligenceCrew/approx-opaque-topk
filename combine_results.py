import json
import sys

"""
This program is used to combine multiple JSON files worth of results into a single JSON file. 
Assumes that each config has a unique name, as we use a dictionary to store the results. 

USAGE: python combine_results.py target_file.json file1.json file2.json file3.json ...
"""

def combine_json_files(target_filename, *args):
    combined_dict = {}

    # Process the JSON files and identifiers
    for json_filename in args:
        # Read the JSON file
        try:
            with open(json_filename, 'r') as json_file:
                data = json.load(json_file)
                # Ensure data is a dictionary
                if not isinstance(data, dict):
                    print(f"Error: {json_filename} does not contain a dictionary.")
                    sys.exit(1)
                # Prefix the keys with the identifier and merge into combined_dict
                for key, value in data.items():
                    combined_dict[key] = value
        except Exception as e:
            print(f"Error reading file {json_filename}: {e}")
            sys.exit(1)

    # Write the combined dictionary to the target file
    try:
        with open(target_filename, 'w') as target_file:
            json.dump(combined_dict, target_file, indent=4)
        print(f"Combined JSON written to {target_filename}")
    except Exception as e:
        print(f"Error writing to target file {target_filename}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    target_file = sys.argv[1]
    json_files_and_ids = sys.argv[2:]
    combine_json_files(target_file, *json_files_and_ids)
