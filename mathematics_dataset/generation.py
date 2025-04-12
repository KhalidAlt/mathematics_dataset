import os
import subprocess
import time
import random
import json

# The base directory of the mathematics_dataset repository
REPO_DIR = "mathematics_dataset"  # Adjust if needed

# Set the total number of examples we want
TOTAL_EXAMPLES = 11000000

# Create output directory
BASE_OUTPUT_DIR = "math_dataset_11M_ar_v2"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# Use smaller batches with smaller per_module counts to avoid hitting the error
NUM_BATCHES = 50
EXAMPLES_PER_BATCH = 220000  # 11M / 50 batches

# Calculate per module counts (reduce these values to avoid errors)
PER_TRAIN_MODULE = 2000  # Smaller than before
PER_TEST_MODULE = 400

# Track progress in a state file
STATE_FILE = os.path.join(BASE_OUTPUT_DIR, "generation_state.json")

# Track total generated examples and completed batches
if os.path.exists(STATE_FILE):
    with open(STATE_FILE, 'r') as f:
        state = json.load(f)
    total_generated = state.get("total_generated", 0)
    completed_batches = set(state.get("completed_batches", []))
    attempts = state.get("attempts", 0)
    print(f"Resuming from previous state: {total_generated} examples generated")
else:
    total_generated = 0
    completed_batches = set()
    attempts = 0

max_attempts = 10000  # Limit number of attempts

# Generate different modules separately to reduce failures
modules = [
    "algebra", "arithmetic", "calculus", "comparison", 
    "measurement", "numbers", "polynomials", "probability"
]

# Helper function to update and save state
def update_state():
    state = {
        "total_generated": total_generated,
        "completed_batches": list(completed_batches),
        "attempts": attempts
    }
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)

while total_generated < TOTAL_EXAMPLES and attempts < max_attempts:
    attempts += 1
    batch_num = attempts
    
    # Skip batches that are already completed
    if batch_num in completed_batches:
        continue
    
    output_dir = os.path.join(BASE_OUTPUT_DIR, f"batch_{batch_num}")
    
    # Skip if the directory already exists and has content
    if os.path.exists(output_dir) and os.listdir(output_dir):
        # Check if the batch was fully completed or partial
        if any(filename.endswith(".txt") for _, _, files in os.walk(output_dir) for filename in files):
            # This batch has generated some files, consider it completed
            num_files = sum(len(files) for _, _, files in os.walk(output_dir))
            estimated_examples = num_files * ((PER_TRAIN_MODULE + PER_TEST_MODULE) // 4)
            total_generated += estimated_examples
            completed_batches.add(batch_num)
            update_state()
            print(f"Batch {batch_num} already exists, counted {estimated_examples} examples")
            continue
    
    # Choose a random module to focus on
    module_filter = random.choice(modules)
    
    # Call the generate_to_file.py script with a module filter
    cmd = [
        "python", "-m", "mathematics_dataset.generate_to_file",
        f"--output_dir={output_dir}",
        f"--per_train_module={PER_TRAIN_MODULE}",
        f"--per_test_module={PER_TEST_MODULE}",
        f"--filter={module_filter}",  # Focus on one module at a time
    ]
    
    print(f"Running batch {batch_num} for module {module_filter}: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True, timeout=7200)  # 2 hour timeout
        
        # Count files and approximate number of examples
        num_files = sum(len(files) for _, _, files in os.walk(output_dir))
        estimated_examples = num_files * ((PER_TRAIN_MODULE + PER_TEST_MODULE) // 4)
        total_generated += estimated_examples
        completed_batches.add(batch_num)
        
        # Update state file after each successful batch
        update_state()
        
        print(f"Batch {batch_num} completed successfully")
        print(f"Progress: Generated approximately {total_generated}/{TOTAL_EXAMPLES} examples")
        
    except subprocess.SubprocessError as e:
        print(f"Error in batch {batch_num}: {e}")
        # Wait a bit before retrying
        time.sleep(5)
    
    # Add some randomness between runs
    time.sleep(2)

print(f"Generation completed: Approximately {total_generated} examples in {BASE_OUTPUT_DIR}")
if total_generated < TOTAL_EXAMPLES:
    print(f"Warning: Only generated {total_generated}/{TOTAL_EXAMPLES} examples due to errors")