import csv
import os
import time

input_path = "./data/raw/23-24_season.csv"
output_dir = "./data/streaming"
delay_seconds =  10
matches_per_week = 10

os.makedirs(output_dir, exist_ok=True)

with open(input_path, mode="r", newline="", encoding="utf-8") as in_file:
    reader = csv.DictReader(in_file)

    for i, row in enumerate(reader, start=1):
        matchweek = ((i - 1) // matches_per_week) + 1
        matchweek_dir = os.path.join(output_dir, f"matchweek{matchweek}")
        os.makedirs(matchweek_dir, exist_ok=True)

        output_path = os.path.join(matchweek_dir, f"match{i}.csv")

        with open(output_path, mode="w", newline="", encoding="utf-8") as out_file:
            writer = csv.writer(out_file)
            writer.writerow([
                row["HomeTeam"],
                row["AwayTeam"],
                row["HomeGoals"],
                row["AwayGoals"],
                row["Result"]
            ])

        print(f"Wrote {output_path}")
        time.sleep(delay_seconds)

print(f"Done — wrote {i} match files across {matchweek} matchweeks to {output_dir}")
