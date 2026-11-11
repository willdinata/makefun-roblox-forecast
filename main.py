"""Forecast chart places from history using recency-weighted rank occupancy,
mobility smoothing, an entrant floor, playing-count momentum, and an age-out
fade. The probabilities are tuned for log score: spread probability broadly
and never concentrate it on a few confident picks.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path


UTC = timezone.utc


def parse_timestamp(value):
    """Parse an ISO timestamp as UTC, treating naive timestamps as UTC."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def clamp(value, lower, upper):
    return min(max(value, lower), upper)


def read_history(path):
    reads = []
    with path.open(encoding="utf-8") as history_file:
        for line in history_file:
            if line.strip():
                reads.append(json.loads(line))
    return reads


def forecast(input_dir):
    with (input_dir / "task.json").open(encoding="utf-8") as task_file:
        task = json.load(task_file)

    places = int(task["places"])
    window_from = parse_timestamp(task["window"]["from"])
    window_until = parse_timestamp(task["window"]["until"])
    reads = read_history(input_dir / "history.ndjson")
    if not reads:
        raise ValueError("history.ndjson must contain at least one read")

    read_times = [parse_timestamp(read["at"]) for read in reads]
    last_read_at = read_times[-1]
    total_weight = 0.0
    occupancy = {}
    chart_weight = {}
    playing_samples = {}
    latest_created = {}

    for read, at in zip(reads, read_times):
        weight = 0.5 ** ((last_read_at - at).total_seconds() / 86400.0)
        total_weight += weight
        for game in read.get("up_and_coming") or []:
            game_id = int(game["universe_id"])
            chart_weight[game_id] = chart_weight.get(game_id, 0.0) + weight

            rank = game.get("rank")
            if rank is not None and 1 <= int(rank) <= places:
                by_place = occupancy.setdefault(game_id, [0.0] * places)
                by_place[int(rank) - 1] += weight

            playing = game.get("playing")
            if playing is not None:
                playing_samples.setdefault(game_id, []).append((at, float(playing)))

            created = game.get("created")
            if created is not None:
                created_at = parse_timestamp(created)
                previous = latest_created.get(game_id)
                if previous is None or created_at > previous:
                    latest_created[game_id] = created_at

    if total_weight <= 0.0:
        raise ValueError("history must have positive total weight")

    probabilities = {}
    for game_id, presence_weight in chart_weight.items():
        raw = [
            occupancy.get(game_id, [0.0] * places)[place] / total_weight
            for place in range(places)
        ]
        smoothed = []
        for place in range(places):
            value = 0.2 * (raw[place - 1] if place > 0 else 0.0)
            value += 0.6 * raw[place]
            value += 0.2 * (raw[place + 1] if place + 1 < places else 0.0)
            smoothed.append(value)

        presence_rate = presence_weight / total_weight
        floor = 0.12 * presence_rate / places
        profile = [0.97 * value + floor for value in smoothed]

        samples = sorted(playing_samples.get(game_id, []), key=lambda sample: sample[0])
        midpoint = len(samples) // 2
        older = samples[:midpoint]
        newer = samples[midpoint:]
        multiplier = 1.0
        if older and newer:
            older_mean = sum(sample[1] for sample in older) / len(older)
            newer_mean = sum(sample[1] for sample in newer) / len(newer)
            if older_mean > 0.0:
                multiplier = clamp(newer_mean / older_mean, 0.75, 1.35)

        created_at = latest_created.get(game_id)
        eligible_fraction = 1.0
        window_duration = (window_until - window_from).total_seconds()
        if created_at is not None and window_duration > 0:
            eligible_fraction = clamp(
                (created_at + timedelta(days=28) - window_from).total_seconds()
                / window_duration,
                0.05,
                1.0,
            )

        probabilities[game_id] = [
            min(0.45, value * multiplier * eligible_fraction)
            for value in profile
        ]

    for place in range(places):
        place_total = sum(profile[place] for profile in probabilities.values())
        if place_total > 0.98:
            scale = 0.98 / place_total
            for profile in probabilities.values():
                profile[place] *= scale

    rows = []
    for place in range(places):
        candidates = [
            (profile[place], game_id)
            for game_id, profile in probabilities.items()
            if profile[place] >= 0.0002
        ]
        candidates.sort(key=lambda candidate: (-candidate[0], candidate[1]))
        for probability, game_id in candidates[:100]:
            rows.append(
                {
                    "universe_id": game_id,
                    "place": place + 1,
                    "p": round(probability, 6),
                }
            )
    rows.sort(key=lambda row: (row["place"], row["universe_id"]))

    top10 = []
    chosen = set()
    for place in range(places):
        available = (
            game_id for game_id in probabilities if game_id not in chosen
        )
        try:
            game_id = min(
                available,
                key=lambda candidate: (-probabilities[candidate][place], candidate),
            )
        except ValueError as error:
            raise ValueError(
                f"history contains fewer than {places} distinct games"
            ) from error
        top10.append(game_id)
        chosen.add(game_id)

    return {"probabilities": rows, "top10": top10}


def main():
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    result = forecast(input_dir)
    output = json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    (output_dir / "forecast.json").write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
