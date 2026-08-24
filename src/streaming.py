import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType, IntegerType
from pyspark.sql import functions as F



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
])

stream_df = spark.readStream.schema(schema).option("recursiveFileLookup","true").csv("./data/streaming/")

output_dir = "./data/output/standings"
os.makedirs(output_dir, exist_ok=True)

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
        accumulated_matches_df = accumulated_matches_df.unionByName(batch_df).cache()

    match_counter["total"] += batch_count
    current_week = match_counter["total"] // matches_per_week

    if current_week > match_counter["last_printed_week"]:
        match_counter["last_printed_week"] = current_week

        standings_df = compute_standings(accumulated_matches_df)

        output_path = os.path.join(output_dir, f"matchweek{current_week}_standings.csv")
        standings_df.toPandas().to_csv(output_path, index=False)

        batch_df.write.mode("overwrite").parquet(f"./data/batches/batch_{batch_id}.parquet")

        print(f"=== Matchweek {current_week} complete ({match_counter['total']} matches processed) ===")
        print(f"Standings written to {output_path}")

counter_query = (stream_df.writeStream
                 .foreachBatch(process_batch)
                 .outputMode("update")
                 .start())

counter_query.awaitTermination()