import json

# Path to the JSON file
json_file_path = '/home/gaoqi/sam3/outputs/sam3_scribble/progress_2/evaluation_results_acdc.json'

# Load the JSON data
with open(json_file_path, 'r') as f:
    data = json.load(f)

# Extract per_patient data
per_patient = data.get('per_patient', {})

# Create a list of (patient, dice) tuples
patients_dice = [(patient, info['overall']['dice']) for patient, info in per_patient.items()]

# Sort by dice in ascending order (lowest dice first, meaning worst)
patients_dice.sort(key=lambda x: x[1])

# Get the worst 10 patients
worst_10 = patients_dice[:10]

# Print the results
print("最差的10个patient（overall_dice最低）：")
for patient, dice in worst_10:
    print(f"{patient}: {dice}")