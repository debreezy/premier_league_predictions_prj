from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("standings").getOrCreate()

df = spark.read \
      .option("header","true") \
      .option("inferSchema","true") \
      .csv("./data/raw/23-24_season.csv")

hometeams_df = (df.select(
                    F.col("HomeTeam").alias("Team"),
                    F.col("MatchDate"),
                    F.col("HomeGoals").alias("GoalsFor"),
                    F.col("AwayGoals").alias("GoalsAgainst"),
                    F.when(F.col("Result") == "H", 3)
                     .when(F.col("Result") == "D", 1)
                     .otherwise(0).alias("Points"),
                    F.when(F.col("Result") == "H", 1).otherwise(0).alias("Wins"),
                    F.when(F.col("Result") == "D", 1).otherwise(0).alias("Draws"),
                    F.when(F.col("Result") == "A", 1).otherwise(0).alias("Losses"),
                    F.lit(1).alias("MatchCount")
                ))

awayteams_df = (df.select(
                    F.col("AwayTeam").alias("Team"),
                    F.col("MatchDate"),
                    F.col("AwayGoals").alias("GoalsFor"),
                    F.col("HomeGoals").alias("GoalsAgainst"),
                    F.when(F.col("Result") == "A", 3)
                     .when(F.col("Result") == "D", 1)
                     .otherwise(0).alias("Points"),
                    F.when(F.col("Result") == "A", 1).otherwise(0).alias("Wins"),
                    F.when(F.col("Result") == "D", 1).otherwise(0).alias("Draws"),
                    F.when(F.col("Result") == "H", 1).otherwise(0).alias("Losses"),
                    F.lit(1).alias("MatchCount")
                ))

hometeams_df.show(5)
awayteams_df.show(5)

league_table_df = (hometeams_df.unionByName(awayteams_df)
                   .groupby("Team")
                   .agg(
                       F.sum("GoalsFor").alias("GoalsFor"),
                       F.sum("GoalsAgainst").alias("GoalsAgainst"),
                       F.sum("Points").alias("Points"),
                       F.sum("Wins").alias("Wins"),
                       F.sum("Draws").alias("Draws"),
                       F.sum("Losses").alias("Losses"),
                       F.sum("MatchCount").alias("MatchCount"),
                       F.max("MatchDate").alias("LastMatchDate")
                   )
                   .withColumn("GoalDifference", F.col("GoalsFor") - F.col("GoalsAgainst"))
                   .select("Team","MatchCount","Wins","Draws", "Losses", "GoalsFor", "GoalsAgainst", "GoalDifference", "Points", "LastMatchDate")
                   .orderBy(F.desc("Points"), F.desc("GoalDifference")))

league_table_df.show()


league_table_df.collect()
spark.stop()