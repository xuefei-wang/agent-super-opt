import json
from datetime import datetime

first_seed_run_info_path = "/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011141/run_info.json"
final_seed_run_info_path = "/home/afarhang/git/sci-agent/cellpose_segmentation/20250513-011318/run_info.json"

# laod jsons   
with open(first_seed_run_info_path, "r") as f:
    first_seed_run_info = json.load(f)

first_start_time = first_seed_run_info["timestamp"]

with open(final_seed_run_info_path, "r") as f:
    final_seed_run_info = json.load(f)
final_finish_time = final_seed_run_info["timestamp_finish"]

# Calculate the time difference in minutes
# they arein the form of : datetime.now().strftime("%Y%m%d-%H%M%S")
time_diff = datetime.strptime(final_finish_time, "%Y%m%d-%H%M%S") - datetime.strptime(first_start_time, "%Y%m%d-%H%M%S")

print(f"Time difference: {time_diff} minutes")





