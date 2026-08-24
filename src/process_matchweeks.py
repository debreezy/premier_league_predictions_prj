from pyspark.sql import functions as F
import os

from src.streaming import spark

output_dir = "./data/output/standings"
os.makedirs(output_dir, exist_ok=True)

matches_per_week = 10
match_counter = {"total": 0, "prev_week": 0}

def process_batch(df, id):
    count = df.count()
    if count == 0:
        return

    match_counter["total"] += count
    current_week = match_counter["total"]// matches_per_week

    if current_week > match_counter["prev_week"]:
        match_counter["prev_week"] = current_week

        standings_df = spark.sql(
            "SELECT * FROM league_table ORDER BY Points DESC, GoalDifference DESC"
        )

        output_path = os.path.join(output_dir, f"matchweek{current_week}_standings.csv")
        standings_df.toPandas().to_csv(output_path, index=False)
