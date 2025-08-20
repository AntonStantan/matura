import json
import os
import sys
# Get the absolute path of the current script's directory
current_dir = os.path.dirname(os.path.abspath("gemini2.5.ipynb"))

# Get the absolute path of the parent directory (project_folder)
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to the Python path
sys.path.append(parent_dir)

# Now you can import from GetXY.py
from GetXY import x_string, y

# ... rest of your code
print("Successfully imported variables!")

dataset = [
    {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": str(x_item)}
                ]
            },
            {
                "role": "model",
                "parts": [
                    {"text": str(y_item)}
                ]
            }
        ]
    }
    for x_item, y_item in zip(x_string, y)
]

print(dataset[0])

file_name = 'my_dataset.jsonl'

#JSONL file
with open(file_name, 'w') as f:
    for entry in dataset:
        json.dump(entry, f)
        f.write('\n')
