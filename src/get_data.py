import argparse
import csv
import json
import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()


API_URL = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY")
STREAM_DESTINATION = "./data/streaming"
STATE_FILE = "./data/processed_match_ids.json"
FIXTURES_FILE = "./data/raw/current_pl_fixtures.csv"


def result_from_score(home_goals, away_goals):
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def format_completed_matches(matches):
    formatted_matches = []
    for match in matches:
        full_time = match["score"]["fullTime"]
        home_goals = full_time["home"]
        away_goals = full_time["away"]

        if match["status"] != "FINISHED" or home_goals is None or away_goals is None:
            continue

        formatted_matches.append({
            "id": match["id"],
            "matchweek": match["matchday"],
            "row": [
                match["homeTeam"]["name"],
                match["awayTeam"]["name"],
                home_goals,
                away_goals,
                result_from_score(home_goals, away_goals),
                match["utcDate"][:10],
            ],
        })

    return formatted_matches


def write_current_fixtures(matches, fixtures_file=FIXTURES_FILE):
    fixtures = [
        {
            "HomeTeam": match["homeTeam"]["name"],
            "AwayTeam": match["awayTeam"]["name"],
            "MatchDate": match["utcDate"][:10],
        }
        for match in matches
    ]
    os.makedirs(os.path.dirname(fixtures_file), exist_ok=True)
    with open(fixtures_file, "w", newline="", encoding="utf-8") as fixture_file:
        writer = csv.DictWriter(fixture_file, fieldnames=["HomeTeam", "AwayTeam", "MatchDate"])
        writer.writeheader()
        writer.writerows(fixtures)


def load_processed_match_ids(state_file=STATE_FILE):
    if not os.path.exists(state_file):
        return set()
    with open(state_file, encoding="utf-8") as state_file_handle:
        return set(json.load(state_file_handle))


def save_processed_match_ids(match_ids, state_file=STATE_FILE):
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    with open(state_file, "w", encoding="utf-8") as state_file_handle:
        json.dump(sorted(match_ids), state_file_handle)


def write_stream_files(matches, processed_match_ids, destination=STREAM_DESTINATION):
    os.makedirs(destination, exist_ok=True)
    new_matches = [match for match in matches if match["id"] not in processed_match_ids]

    for match in new_matches:
        matchweek_directory = os.path.join(destination, f"matchweek_{match['matchweek']}")
        os.makedirs(matchweek_directory, exist_ok=True)
        file_path = os.path.join(matchweek_directory, f"match_{match['id']}.csv")
        temporary_path = f"{file_path}.tmp"
        with open(temporary_path, "w", newline="", encoding="utf-8") as stream_file:
            csv.writer(stream_file).writerow(match["row"])
        os.replace(temporary_path, file_path)

    return {match["id"] for match in new_matches}


def get_prem_matches(processed_match_ids):
    response = requests.get(
        f"{API_URL}competitions/PL/matches?season=2026",
        headers={"X-Auth-Token": API_KEY},
        timeout=30,
    )
    response.raise_for_status()

    matches = response.json()["matches"]
    write_current_fixtures(matches)
    completed_matches = format_completed_matches(matches)
    new_match_ids = write_stream_files(completed_matches, processed_match_ids)
    processed_match_ids.update(new_match_ids)
    save_processed_match_ids(processed_match_ids)
    print(f"Published {len(new_match_ids)} new completed matches.")


def poll_matches(poll_seconds):
    processed_match_ids = load_processed_match_ids()
    while True:
        try:
            get_prem_matches(processed_match_ids)
        except requests.RequestException as error:
            print(f"Could not retrieve Premier League matches: {error}")
        time.sleep(poll_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true", help="Fetch the current PL fixtures and exit.")
    arguments = parser.parse_args()
    if arguments.poll_seconds <= 0:
        parser.error("--poll-seconds must be greater than zero")
    if arguments.once:
        get_prem_matches(load_processed_match_ids())
    else:
        poll_matches(arguments.poll_seconds)



