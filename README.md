# intent-to-schedule

Turn scheduling requests into commands and solve the schedule as a mixed integer linear program. Designed mainly for use from LLM-based AI agents.

## Usage

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run intent-to-schedule init --calendar examples/people_8_weeks_2/calendar.json
uv run intent-to-schedule help
```

`help <command>` and `schema` describe everything else. `examples/` holds synthetic calendars with scheduling requests to try.

`chat` is a small demonstration agent using OpenAI; set `OPENAI_API_KEY`:

```bash
uv run intent-to-schedule chat "Schedule a one-hour review with Takahashi and Ito on Friday" --model <MODEL>
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.
