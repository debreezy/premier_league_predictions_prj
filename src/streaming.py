import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType, IntegerType, DateType
from pyspark.sql import functions as F

from predict import load_model_and_schedule, normalize_team_names, remaining_fixtures, predict_title_probabilities


spark = (
    SparkSession.builder
    .appName("PremStreamingSession")
    .getOrCreate()
)

schema = StructType([
    StructField("HomeTeam", StringType(), True),
    StructField("AwayTeam", StringType(), True),
    StructField("HomeGoals", IntegerType(), True),
    StructField("AwayGoals", IntegerType(), True),
    StructField("Result", StringType(), True),
    StructField("MatchDate", DateType(), True),
])

stream_df = spark.readStream.schema(schema).option("recursiveFileLookup","true").csv("./data/streaming/")

standings_csv_dir = "./data/output/standings"
title_race_csv_dir = "./data/output/title_race"
checkpoint_dir = "./checkpoints/streaming"
os.makedirs(standings_csv_dir, exist_ok=True)
os.makedirs(title_race_csv_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

train_df, model, season_fixtures = load_model_and_schedule()

matches_per_week = 10
match_counter = {"total": 0, "last_printed_week": 0}
accumulated_matches_df = None

def compute_standings(matches_df):
    home = matches_df.select(
        F.col("HomeTeam").alias("Team"),
        F.col("HomeGoals").alias("GoalsFor"),
        F.col("AwayGoals").alias("GoalsAgainst"),
        F.when(F.col("Result") == "H", 3).when(F.col("Result") == "D", 1).otherwise(0).alias("Points"),
        F.when(F.col("Result") == "H", 1).otherwise(0).alias("Wins"),
        F.when(F.col("Result") == "D", 1).otherwise(0).alias("Draws"),
        F.when(F.col("Result") == "A", 1).otherwise(0).alias("Losses"),
        F.lit(1).alias("MatchCount")
    )

    away = matches_df.select(
        F.col("AwayTeam").alias("Team"),
        F.col("AwayGoals").alias("GoalsFor"),
        F.col("HomeGoals").alias("GoalsAgainst"),
        F.when(F.col("Result") == "A", 3).when(F.col("Result") == "D", 1).otherwise(0).alias("Points"),
        F.when(F.col("Result") == "A", 1).otherwise(0).alias("Wins"),
        F.when(F.col("Result") == "D", 1).otherwise(0).alias("Draws"),
        F.when(F.col("Result") == "H", 1).otherwise(0).alias("Losses"),
        F.lit(1).alias("MatchCount")
    )

    return (home.unionByName(away)
            .groupBy("Team")
            .agg(
                F.sum("GoalsFor").alias("GoalsFor"),
                F.sum("GoalsAgainst").alias("GoalsAgainst"),
                F.sum("Points").alias("Points"),
                F.sum("Wins").alias("Wins"),
                F.sum("Draws").alias("Draws"),
                F.sum("Losses").alias("Losses"),
                F.sum("MatchCount").alias("MatchesPlayed")
            )
            .withColumn("GoalDifference", F.col("GoalsFor") - F.col("GoalsAgainst"))
            .select("Team", "MatchesPlayed", "Wins", "Draws", "Losses",
                    "GoalsFor", "GoalsAgainst", "GoalDifference", "Points")
            .orderBy(F.desc("Points"), F.desc("GoalDifference")))

def process_batch(batch_df, batch_id):
    global accumulated_matches_df

    batch_count = batch_df.count()
    if batch_count == 0:
        return

    batch_df = batch_df.cache()

    if accumulated_matches_df is None:
        accumulated_matches_df = batch_df
    else:
        accumulated_matches_df = accumulated_matches_df.unionByName(batch_df)
    accumulated_matches_df = accumulated_matches_df.cache()

    match_counter["total"] += batch_count
    new_total_weeks = match_counter["total"] // matches_per_week

    for current_week in range(match_counter["last_printed_week"] + 1, new_total_weeks + 1):
        match_counter["last_printed_week"] = current_week

        matches_through_week = (
            accumulated_matches_df
            .orderBy("MatchDate")
            .limit(current_week * matches_per_week)
            .cache()
        )

        standings_df = compute_standings(matches_through_week)
        standings_csv_path = os.path.join(standings_csv_dir, f"matchweek{current_week}_standings.csv")
        standings_df.toPandas().to_csv(standings_csv_path, index=False)

        latest_match_date = matches_through_week.agg(F.max("MatchDate")).first()[0]
        print(f"=== Matchweek {current_week} complete ({current_week * matches_per_week} matches processed, through {latest_match_date}) ===")
        print(f"Standings written to {standings_csv_path}")

        played_matches = matches_through_week.select(
            "HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals", "Result", "MatchDate"
        ).toPandas()
        played_matches["MatchDate"] = pd.to_datetime(played_matches["MatchDate"])
        played_matches = normalize_team_names(played_matches)

        title_probabilities = predict_title_probabilities(
            played_matches, remaining_fixtures(season_fixtures, played_matches), train_df, model
        )
        title_probabilities["Matchweek"] = current_week

        title_race_csv_path = os.path.join(title_race_csv_dir, f"matchweek{current_week}_title_probabilities.csv")
        title_probabilities.to_csv(title_race_csv_path, index=False)

        print(f"Title probabilities written to {title_race_csv_path}")
        print(title_probabilities.to_string(index=False))

        matches_through_week.unpersist()

counter_query = (stream_df.writeStream
                 .foreachBatch(process_batch)
                 .outputMode("update")
                 .option("checkpointLocation", checkpoint_dir)
                 .start())

counter_query.awaitTermination()
