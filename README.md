

This project streams completed Premier League matches from the Football-Data
API, produces a league table after each completed matchweek, and estimates the
chance of each club winning the title.

Copy `.env.example` to `.env` and add your API token:
```dotenv
API_URL="https://api.football-data.org/v4/"
API_KEY=your_api_key_here
```
## Run the live pipeline

First retrieve the current Premier League fixture list and any completed
matches. This also creates `data/raw/current_pl_fixtures.csv`, which is the
fixture source used for title-race simulations.

```powershell
python src/get_data.py --once
```

Start the streaming processor in a separate terminal:

```powershell
python src/streaming.py
```

To continuously poll Football-Data for new completed matches instead, run this
in another terminal (the default polling period is 60 seconds):

```powershell
python src/get_data.py --poll-seconds 60
```

Stop either long-running command with `Ctrl+C`.

## Outputs

After every 10 processed matches, the streaming job writes:

- `data/output/standings/matchweek<N>_standings.csv` — league table.
- `data/output/title_race/matchweek<N>_title_probabilities.csv` — simulated
  title probabilities.

The title simulation uses 100 season simulations by default. A probability is
therefore an estimate, not a guarantee.

## Team and season consistency

The title calculation must combine results and fixtures from the same Premier
League season. `get_data.py` now saves the API's full current fixture list and
the prediction code normalizes names such as `Arsenal` and `Arsenal FC`.

If you see an error saying the results and fixture schedule are from different
seasons, refresh the fixture list with `python src/get_data.py --once`, then
restart `src/streaming.py`. Do not point the live simulation at
`data/raw/23-24_season.csv`; it is a historical file used for local simulation
and examples.

For a completely fresh season run, stop both processes and remove the previous
stream state before running the commands above:

```powershell
Remove-Item -Recurse -Force data/streaming, data/output, checkpoints/streaming
Remove-Item -Force data/processed_match_ids.json
```

Only do this when you intentionally want to discard the locally generated
stream, outputs, and checkpoint state.

## Historical replay

To stream the included 2023/24 results locally, run:

```powershell
python src/simulator.py
```

It writes one CSV per match to `data/streaming`. The live streaming process can
then consume those files. This replay uses the historical season data, so clear
the stream state before switching back to live data.

## New season

The Football-Data request in `src/get_data.py` currently specifies
`season=2026`. Update that value when beginning a new Premier League season,
then fetch fixtures again with `--once` before starting the pipeline.
