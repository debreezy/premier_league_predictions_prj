
import os
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

HISTORICAL_DATA_PATH = "./data/raw/epl_final.csv"
CURRENT_SEASON_PATH = "./data/raw/current_pl_fixtures.csv"
OUTPUT_DIR = "./data/output/predictions"
TEST_SEASON = "2023/24"
ROLLING_WINDOW = 5
HOME_ADVANTAGE = 60
ELO_K = 20

TEAM_NAME_ALIASES = {
    "Arsenal": "Arsenal FC", "Aston Villa": "Aston Villa FC",
    "Bournemouth": "AFC Bournemouth", "Brighton": "Brighton & Hove Albion FC",
    "Brentford": "Brentford FC", "Burnley": "Burnley FC",
    "Chelsea": "Chelsea FC", "Crystal Palace": "Crystal Palace FC",
    "Everton": "Everton FC", "Fulham": "Fulham FC", "Ipswich": "Ipswich Town FC",
    "Leeds": "Leeds United FC", "Liverpool": "Liverpool FC",
    "Luton": "Luton Town FC", "Man City": "Manchester City FC",
    "Man United": "Manchester United FC", "Newcastle": "Newcastle United FC",
    "Nott'm Forest": "Nottingham Forest FC", "Sheffield United": "Sheffield United FC",
    "Tottenham": "Tottenham Hotspur FC", "West Ham": "West Ham United FC",
    "Wolves": "Wolverhampton Wanderers FC",
}

FEATURE_COLUMNS = [
    "HomeRollingPts",
    "HomeRollingGoalsFor",
    "HomeRollingGoalsAgainst",
    "AwayRollingPts",
    "AwayRollingGoalsFor",
    "AwayRollingGoalsAgainst",
    "HomeElo",
    "AwayElo",
    "EloDifference",
]


def _load_engineered_matches(historical_path=HISTORICAL_DATA_PATH, test_season=TEST_SEASON):
    raw = pd.read_csv(historical_path)
    raw = raw.rename(columns={
        "FullTimeHomeGoals": "HomeGoals",
        "FullTimeAwayGoals": "AwayGoals",
        "FullTimeResult": "Result",
    })
    df = raw[["Season", "MatchDate", "HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals", "Result"]].copy()
    df["MatchDate"] = pd.to_datetime(df["MatchDate"])

    home_mappings = {"H": 3, "D": 1, "A": 0}
    away_mappings = {"A": 3, "D": 1, "H": 0}
    df["HomeResult"] = df["Result"].map(home_mappings)
    df["AwayResult"] = df["Result"].map(away_mappings)

    df["MatchID"] = df.index
    home_records = df[["MatchID", "Season", "MatchDate", "HomeTeam", "HomeGoals", "AwayGoals", "HomeResult"]].copy()
    home_records.columns = ["MatchID", "Season", "MatchDate", "Team", "GoalsFor", "GoalsAgainst", "Points"]
    away_records = df[["MatchID", "Season", "MatchDate", "AwayTeam", "AwayGoals", "HomeGoals", "AwayResult"]].copy()
    away_records.columns = ["MatchID", "Season", "MatchDate", "Team", "GoalsFor", "GoalsAgainst", "Points"]
    team_history = pd.concat([home_records, away_records]).sort_values(["Team", "MatchDate"])

    for column, source in [
        ("RollingPts", "Points"),
        ("RollingGoalsFor", "GoalsFor"),
        ("RollingGoalsAgainst", "GoalsAgainst"),
    ]:
        team_history[column] = (
            team_history.groupby("Team")[source]
            .transform(lambda values: values.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean())
        )

    df = df.merge(
        team_history[["MatchID", "Team", "RollingPts", "RollingGoalsFor", "RollingGoalsAgainst"]],
        left_on=["MatchID", "HomeTeam"], right_on=["MatchID", "Team"], how="left",
    ).rename(columns={
        "RollingPts": "HomeRollingPts",
        "RollingGoalsFor": "HomeRollingGoalsFor",
        "RollingGoalsAgainst": "HomeRollingGoalsAgainst",
    }).drop(columns=["Team"])

    df = df.merge(
        team_history[["MatchID", "Team", "RollingPts", "RollingGoalsFor", "RollingGoalsAgainst"]],
        left_on=["MatchID", "AwayTeam"], right_on=["MatchID", "Team"], how="left",
    ).rename(columns={
        "RollingPts": "AwayRollingPts",
        "RollingGoalsFor": "AwayRollingGoalsFor",
        "RollingGoalsAgainst": "AwayRollingGoalsAgainst",
    }).drop(columns=["Team"])

    ratings = defaultdict(lambda: 1500.0)
    elo_rows = []
    for _, matches_on_date in df.sort_values("MatchDate").groupby("MatchDate", sort=False):
        for match_index, match in matches_on_date.iterrows():
            home_rating = ratings[match["HomeTeam"]]
            away_rating = ratings[match["AwayTeam"]]
            elo_rows.append((match_index, home_rating, away_rating, home_rating + HOME_ADVANTAGE - away_rating))
        for _, match in matches_on_date.iterrows():
            home_rating = ratings[match["HomeTeam"]]
            away_rating = ratings[match["AwayTeam"]]
            expected_home = 1 / (1 + 10 ** (-(home_rating + HOME_ADVANTAGE - away_rating) / 400))
            actual_home = {"H": 1.0, "D": 0.5, "A": 0.0}[match["Result"]]
            rating_change = ELO_K * (actual_home - expected_home)
            ratings[match["HomeTeam"]] += rating_change
            ratings[match["AwayTeam"]] -= rating_change

    elo_df = pd.DataFrame(
        elo_rows, columns=["MatchID", "HomeElo", "AwayElo", "EloDifference"]
    ).set_index("MatchID")
    df = df.join(elo_df).fillna(0).drop(columns=["MatchID"])

    return df[df["Season"] != test_season].reset_index(drop=True)


def train_model(train_df):
    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(train_df[FEATURE_COLUMNS], train_df["Result"])
    return model


def _form_features(form_history, team):
    recent_matches = form_history[team][-ROLLING_WINDOW:]
    if not recent_matches:
        return 0.0, 0.0, 0.0
    values = np.asarray(recent_matches, dtype=float)
    return tuple(values.mean(axis=0))


def initialize_team_state(completed_matches):
    ratings = defaultdict(lambda: 1500.0)
    form_history = defaultdict(list)
    for _, completed_matchweek in completed_matches.sort_values("MatchDate").groupby("MatchDate", sort=False):
        update_team_state(completed_matchweek, ratings, form_history)
    return ratings, form_history


def predict_matchweek(fixtures, ratings, form_history, model):
    required_columns = {"HomeTeam", "AwayTeam"}
    missing_columns = required_columns.difference(fixtures.columns)
    if missing_columns:
        raise ValueError(f"Fixtures are missing columns: {sorted(missing_columns)}")

    feature_rows = []
    for _, fixture in fixtures.iterrows():
        home_points, home_goals_for, home_goals_against = _form_features(form_history, fixture["HomeTeam"])
        away_points, away_goals_for, away_goals_against = _form_features(form_history, fixture["AwayTeam"])
        home_elo = ratings[fixture["HomeTeam"]]
        away_elo = ratings[fixture["AwayTeam"]]
        feature_rows.append({
            "HomeRollingPts": home_points,
            "HomeRollingGoalsFor": home_goals_for,
            "HomeRollingGoalsAgainst": home_goals_against,
            "AwayRollingPts": away_points,
            "AwayRollingGoalsFor": away_goals_for,
            "AwayRollingGoalsAgainst": away_goals_against,
            "HomeElo": home_elo,
            "AwayElo": away_elo,
            "EloDifference": home_elo + HOME_ADVANTAGE - away_elo,
        })

    match_features = pd.DataFrame(feature_rows)[FEATURE_COLUMNS]
    probabilities = pd.DataFrame(model.predict_proba(match_features), columns=model.classes_)
    predictions = fixtures.reset_index(drop=True).copy()
    predictions[["Prob_A", "Prob_D", "Prob_H"]] = probabilities[["A", "D", "H"]]
    predictions["Prediction"] = model.predict(match_features)
    return predictions


def update_team_state(completed_matchweek, ratings, form_history):

    required_columns = {"HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals", "Result"}
    missing_columns = required_columns.difference(completed_matchweek.columns)
    if missing_columns:
        raise ValueError(f"Results are missing columns: {sorted(missing_columns)}")

    rating_changes = defaultdict(float)
    for _, match in completed_matchweek.iterrows():
        home_rating = ratings[match["HomeTeam"]]
        away_rating = ratings[match["AwayTeam"]]
        expected_home = 1 / (1 + 10 ** (-(home_rating + HOME_ADVANTAGE - away_rating) / 400))
        actual_home = {"H": 1.0, "D": 0.5, "A": 0.0}[match["Result"]]
        rating_change = ELO_K * (actual_home - expected_home)
        rating_changes[match["HomeTeam"]] += rating_change
        rating_changes[match["AwayTeam"]] -= rating_change

    for team, rating_change in rating_changes.items():
        ratings[team] += rating_change

    for _, match in completed_matchweek.iterrows():
        home_points = {"H": 3, "D": 1, "A": 0}[match["Result"]]
        away_points = {"A": 3, "D": 1, "H": 0}[match["Result"]]
        form_history[match["HomeTeam"]].append((home_points, match["HomeGoals"], match["AwayGoals"]))
        form_history[match["AwayTeam"]].append((away_points, match["AwayGoals"], match["HomeGoals"]))


def split_matchweeks(fixtures):
    matchweeks = []
    current_matchweek = []
    teams_in_matchweek = set()

    for _, fixture in fixtures.sort_values("MatchDate").iterrows():
        fixture_teams = {fixture["HomeTeam"], fixture["AwayTeam"]}
        if fixture_teams.intersection(teams_in_matchweek):
            matchweeks.append(pd.DataFrame(current_matchweek))
            current_matchweek = []
            teams_in_matchweek = set()
        current_matchweek.append(fixture)
        teams_in_matchweek.update(fixture_teams)

    if current_matchweek:
        matchweeks.append(pd.DataFrame(current_matchweek))
    return matchweeks


def load_model_and_schedule(current_season_path=CURRENT_SEASON_PATH):
    train_df = _load_engineered_matches()
    model = train_model(train_df)
    if not os.path.exists(current_season_path):
        raise FileNotFoundError(
            f"Current Premier League fixtures not found at {current_season_path}. "
            "Run get_data.py once before starting streaming."
        )
    season_fixtures = pd.read_csv(current_season_path, parse_dates=["MatchDate"])
    season_fixtures = normalize_team_names(season_fixtures)
    return train_df, model, season_fixtures


def normalize_team_names(matches):
    """Use Football-Data club names everywhere fixtures and results meet."""
    normalized = matches.copy()
    for column in ("HomeTeam", "AwayTeam"):
        if column in normalized:
            normalized[column] = normalized[column].replace(TEAM_NAME_ALIASES)
    return normalized


def remaining_fixtures(season_fixtures, played_matches):
    if played_matches.empty:
        return season_fixtures.copy()
    merged = season_fixtures.merge(
        played_matches[["HomeTeam", "AwayTeam", "MatchDate"]],
        on=["HomeTeam", "AwayTeam", "MatchDate"],
        how="left",
        indicator=True,
    )
    return merged[merged["_merge"] == "left_only"].drop(columns=["_merge"]).reset_index(drop=True)


def current_table(played_matches, teams):
    table = {team: 0 for team in teams}
    for _, match in played_matches.iterrows():
        if match["Result"] == "H":
            table[match["HomeTeam"]] += 3
        elif match["Result"] == "A":
            table[match["AwayTeam"]] += 3
        else:
            table[match["HomeTeam"]] += 1
            table[match["AwayTeam"]] += 1
    return table



_SIMULATED_GOALS_BY_OUTCOME = {"H": (2, 1), "D": (1, 1), "A": (1, 2)}


def predict_title_probabilities(played_matches, remaining_season_fixtures, train_df, model, n_simulations=100):

    played_matches = normalize_team_names(played_matches)
    remaining_season_fixtures = normalize_team_names(remaining_season_fixtures)

    base_ratings, base_form_history = initialize_team_state(train_df)
    for _, day_matches in played_matches.sort_values("MatchDate").groupby("MatchDate", sort=False):
        update_team_state(day_matches, base_ratings, base_form_history)

    teams = sorted(set(remaining_season_fixtures["HomeTeam"]).union(remaining_season_fixtures["AwayTeam"]))
    played_teams = set(played_matches["HomeTeam"]).union(played_matches["AwayTeam"])
    unexpected_teams = played_teams.difference(teams)
    if unexpected_teams:
        raise ValueError(
            "Played matches and fixture schedule are from different Premier League seasons or use "
            f"unmapped team names: {sorted(unexpected_teams)}. Refresh current_pl_fixtures.csv."
        )
    base_table = current_table(played_matches, teams)
    remaining_matchweeks = split_matchweeks(remaining_season_fixtures)

    outcomes = ["H", "D", "A"]
    title_counts = {team: 0 for team in teams}

    for _ in range(n_simulations):

        ratings = defaultdict(lambda: 1500.0, base_ratings)
        form_history = defaultdict(list, {team: list(matches) for team, matches in base_form_history.items()})
        table = base_table.copy()

        for matchweek in remaining_matchweeks:
            predictions = predict_matchweek(matchweek[["HomeTeam", "AwayTeam"]], ratings, form_history, model)

            simulated_results = []
            for _, fixture in predictions.iterrows():
                outcome = np.random.choice(outcomes, p=[fixture["Prob_H"], fixture["Prob_D"], fixture["Prob_A"]])
                home_goals, away_goals = _SIMULATED_GOALS_BY_OUTCOME[outcome]
                simulated_results.append({
                    "HomeTeam": fixture["HomeTeam"],
                    "AwayTeam": fixture["AwayTeam"],
                    "HomeGoals": home_goals,
                    "AwayGoals": away_goals,
                    "Result": outcome,
                })
                if outcome == "H":
                    table[fixture["HomeTeam"]] += 3
                elif outcome == "A":
                    table[fixture["AwayTeam"]] += 3
                else:
                    table[fixture["HomeTeam"]] += 1
                    table[fixture["AwayTeam"]] += 1

            update_team_state(pd.DataFrame(simulated_results), ratings, form_history)

        highest_points = max(table.values())
        tied_teams = [team for team, points in table.items() if points == highest_points]
        title_counts[np.random.choice(tied_teams)] += 1

    return pd.DataFrame({
        "Team": teams,
        "TitleProbability": [round(100 * title_counts[team] / n_simulations, 2) for team in teams],
    }).sort_values("TitleProbability", ascending=False).reset_index(drop=True)


def predict_season(current_season_path=CURRENT_SEASON_PATH, output_dir=OUTPUT_DIR):

    train_df = _load_engineered_matches()
    model = train_model(train_df)

    season_fixtures = (
        pd.read_csv(current_season_path, parse_dates=["MatchDate"])
        .sort_values("MatchDate")
        .reset_index(drop=True)
    )
    matchweeks = split_matchweeks(season_fixtures)
    ratings, form_history = initialize_team_state(train_df)

    os.makedirs(output_dir, exist_ok=True)
    all_predictions = []

    for matchweek_number, matchweek in enumerate(matchweeks, start=1):
        predictions = predict_matchweek(matchweek[["HomeTeam", "AwayTeam"]], ratings, form_history, model)
        predictions["Matchweek"] = matchweek_number
        predictions["ActualResult"] = matchweek["Result"].values

        output_path = os.path.join(output_dir, f"matchweek{matchweek_number}_predictions.csv")
        predictions.to_csv(output_path, index=False)

        print(f"=== Matchweek {matchweek_number} predictions (written to {output_path}) ===")
        for _, row in predictions.iterrows():
            print(f"{row['HomeTeam']} vs {row['AwayTeam']}: predicted {row['Prediction']} (actual {row['ActualResult']})")

        update_team_state(matchweek, ratings, form_history)
        all_predictions.append(predictions)

    return pd.concat(all_predictions, ignore_index=True)


