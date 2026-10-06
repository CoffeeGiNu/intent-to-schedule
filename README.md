# intent-to-schedule

Turn scheduling requests into commands and compute the schedule with a mixed integer linear program. Designed mainly for use from LLM-based AI agents.

## Usage

Requires [uv](https://docs.astral.sh/uv/).

Install the dependencies:

```bash
uv sync
```

`init` creates the state from a calendar; `examples/` holds synthetic calendars with scheduling requests to try:

```bash
uv run intent-to-schedule init --calendar examples/people_8_weeks_2/calendar.json
```

`apply` runs a batch of commands that expresses a request:

```bash
uv run intent-to-schedule apply <<'EOF'
{"commands": [
  {"kind": "add_task", "task": {
    "name": "Design review", "duration": "PT1H",
    "participant_ids": ["takahashi", "ito"],
    "importance": "high", "required": true, "stability": "normal"
  }}
]}
EOF
```

`schedule` solves the problem and saves the schedule:

```bash
uv run intent-to-schedule schedule
```

`query` reads the problem and the saved schedule:

```bash
uv run intent-to-schedule query <<'EOF'
{"kind": "previous_schedule"}
EOF
```

`schema` prints the JSON Schema of the apply or query input:

```bash
uv run intent-to-schedule schema apply
```

`help` lists every command, and `help <command>` describes each one:

```bash
uv run intent-to-schedule help
```

`chat` runs a small (demonstration) agent using OpenAI; set `OPENAI_API_KEY`:

```bash
uv run intent-to-schedule chat "Schedule a one-hour review with Takahashi and Ito on Friday" --model gpt-6.1-sol
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.
