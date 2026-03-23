import os
import subprocess
import itertools
import csv
import json

def get_rank1(best_result):
    # Depending on evaluate implementation, might be 'Rank-1'
    if 'Rank-1' in best_result:
        return best_result['Rank-1']
    if 'rank-1' in best_result:
        return best_result['rank-1']
    return 0.0

def main():
    epochs_list = [5, 10]
    lr_list = [3e-4, 1e-4]
    P_list = [8, 16]
    K_list = [4]

    combinations = list(itertools.product(epochs_list, lr_list, P_list, K_list))
    total_exp = len(combinations)
    print(f"Total number of experiments to run: {total_exp}")

    summary_file = "experiment_summary.csv"
    if not os.path.exists(summary_file):
        with open(summary_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["run_name", "epochs", "lr", "P", "K", "mAP", "rank1"])

    for i, (epochs, lr, P, K) in enumerate(combinations, 1):
        run_name = f"exp{i}_e{epochs}_lr{lr}_P{P}_K{K}"
        print(f"\n--- Running experiment {i}/{total_exp}: {run_name} ---")

        final_txt_path = os.path.join("runs", run_name, "final.txt")
        if os.path.exists(final_txt_path):
            print(f"Skipping {run_name}, already completed.")
            continue

        cmd = [
            "python", "scripts/train.py",
            "--run_name", run_name,
            "--epochs", str(epochs),
            "--lr", str(lr),
            "--P", str(P),
            "--K", str(K)
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Experiment {run_name} failed with error: {e}")
            continue

        if os.path.exists(final_txt_path):
            with open(final_txt_path, "r") as f:
                data = json.load(f)
            
            best_map = data.get("best_mAP", 0)
            best_result = data.get("best_result", {})
            rank1 = get_rank1(best_result)

            with open(summary_file, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([run_name, epochs, lr, P, K, best_map, rank1])
            print(f"Logged result for {run_name}: mAP={best_map:.4f}, Rank-1={rank1:.4f}")

if __name__ == "__main__":
    main()
