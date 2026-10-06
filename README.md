# intent-to-schedule

Turn scheduling requests into JSON commands and solve them with a mixed integer program (OR-Tools MathOpt). People and fixed appointments come from a calendar; movable tasks and constraints are added through commands. External agents can drive the command line interface without reading the complete state.

## Install and initialize

```bash
uv sync
uv run intent-to-schedule --state example-state.json init \
  --calendar examples/people_8_weeks_2/calendar.json
```

The examples below use that calendar, whose offset is `+09:00` and slot length is 30 minutes. The default state path is `.state/state.json`; pass `--state` before the subcommand to use another file. `init` replaces the state, clearing the previous schedule and dialogue, and prints the same answer as the `summary` query.

## Commands

```bash
uv run intent-to-schedule --help
uv run intent-to-schedule help query
uv run intent-to-schedule schema
uv run intent-to-schedule schema query
uv run intent-to-schedule --state example-state.json show
uv run intent-to-schedule --state example-state.json query --file query.json
uv run intent-to-schedule --state example-state.json apply --file commands.json
uv run intent-to-schedule --state example-state.json solve
```

`help <command>` describes each command. `schema` (or `schema apply`) prints the command input schema; `schema query` prints the query input schema. `show` prints the complete problem, previous schedule, and dialogue. Results are one JSON document; help is plain text.

Both `query` and `apply` read standard input when `--file` is omitted:

```bash
echo '{"kind":"summary"}' | uv run intent-to-schedule --state example-state.json query
```

## Query

Save any of these nine inputs as `query.json`. Filters combine with **and**. Listings return `kind`, `items`, `total` (all matches before the limit), and `truncated`. `limit` defaults to 20 and must be between 1 and 100. Empty results are successful. People and tasks also accept `select: "one"`: exactly one match is required before limiting; zero or multiple matches exit with status 1. Ambiguous selections describe up to five candidates.

`summary` returns the horizon and slot in `grid`, element `counts`, and `has_previous`:

```json
{"kind":"summary"}
```

`people` returns identifiers and names. Filter by `person_ids` or exact `name_equals`:

```json
{"kind":"people","filter":{"name_equals":"Misaki Takahashi"},"select":"one"}
```

`tasks` returns movable tasks and fixed appointments with identifiers. Filters include `task_ids`, `type` (`task` or `fixed`), `name_equals`, `participant_ids_all`, and `start_range`. All specified participants must be present. The start range applies to fixed appointments, includes its start, and excludes its end:

```json
{
  "kind":"tasks",
  "filter":{
    "type":"fixed",
    "name_equals":"Team meeting",
    "participant_ids_all":["takahashi"],
    "start_range":{"start":"2026-10-13T00:00:00+09:00","end":"2026-10-14T00:00:00+09:00"}
  },
  "select":"one"
}
```

`constraints` returns each constraint's `id`, `label`, `requirement`, and `condition`. Conditions keep the entered values, including the original windows before rounding. Optional filters are `constraint_ids` and `task_ids` (references any listed task). Each matching constraint appears once. Only the list of constraints is limited; individual conditions are returned in full:

```json
{"kind":"constraints"}
```

`previous_schedule` returns `has_previous` and entries with `status: "scheduled"`, `task_id`, `name`, `start`, `end`, and `participant_ids`, or `status: "dropped"`, `task_id`, and `name`. Names, end times, and participants are recorded at solve time and remain unchanged after a task is edited or removed. Optional filters are `task_ids` and `start_range`; a start range excludes dropped entries. This is the last saved solution, which may predate current problem changes:

```json
{"kind":"previous_schedule"}
```

`agenda` returns one person's `person_id`, `has_previous`, and `days`. Each day has `date`, `working` intervals from the calendar with overlapping or adjacent intervals merged, `items`, and `free` intervals. Items are the person's fixed appointments (`type: "fixed"`) and the tasks the last saved solution placed for that person (`type: "scheduled"`), in start order, each with `type`, `task_id`, `name`, `start`, `end`, and `participant_ids`. `free` is working time minus the items, using real times without slot rounding. Dates without working time appear with empty `working`. An item crossing midnight appears on both dates. The optional `date_range` includes its start date and excludes its end date, counted in the horizon's starting offset; omit it for the whole horizon. An unknown `person_id` is rejected:

```json
{"kind":"agenda","person_id":"ito","date_range":{"start":"2026-10-19","end":"2026-10-24"}}
```

**Scheduled items come from the last saved solution, as in `previous_schedule`, and may predate current problem changes.** They keep the names, times, and participants recorded at solve time. Movable tasks added or edited since then appear only after the next `solve`, and `free` does not account for them.

`evaluation` measures the **current constraints against the last saved solution**. Each item has `constraint_id`, `label`, `requirement`, `violation` (`amount` and `unit`, either `hours` or `count`), `cost` (soft constraints only), and `breakdown`. Hard constraints have no cost: their items and breakdown parts omit `cost`, so tell them apart by `requirement.kind`. Time windows and bounds break down violations by `task_id`; daily limits break them down by `date`; task gaps have an empty breakdown. Parts include their violation, and for soft constraints their cost, including zero amounts. Daily breakdowns include every horizon date.

```json
{"kind":"evaluation"}
```

Filter by `violated_only` (default false), `constraint_ids`, or `task_ids`. A task filter selects any constraint referencing a listed task and retains its complete amount and breakdown. Only the constraint list is limited. Hard violations appear first, then soft constraints by highest cost, then satisfied hard constraints; ties use the constraint identifier. With no saved solution, the answer is `{"kind":"evaluation","has_previous":false,"items":[],"total":0,"truncated":false}`.

```json
{"kind":"evaluation","filter":{"violated_only":true,"task_ids":["review"]},"limit":10}
```

Movable tasks use their recorded start and end, even after their duration changes. Missing or dropped movable tasks contribute zero; removed tasks are ignored. Fixed tasks use their current intervals rounded outward to slots. Changes after a solve can therefore produce hard violations in this query. Evaluations are computed from placements, using the same slot, window, gap, and daily rules as the solver.

`objective_policy` returns the coefficients actually used by the default solver: `drop_costs` by importance, `weights` by strength, `per_count` for daily count limits, and `stability_drop_cost_ratio`. `hard_violation_weight` (per hour or count) and `required_drop_cost` (per task) apply only to the relaxed solve that explains an infeasible solve; both exceed every soft cost. Use it for current values when comparing trade-offs:

```json
{"kind":"objective_policy"}
```

`available_starts` returns start times where all participants are available and free of their fixed appointments. Duration must be positive and a multiple of the slot. Omit `windows` for the whole horizon; an empty array allows no times. The whole task must fit within the windows. An empty participant list checks only the horizon and windows. Candidates may overlap each other:

```json
{
  "kind":"available_starts",
  "participant_ids":["takahashi","ito"],
  "duration":"PT1H",
  "windows":[{"weekdays":["friday"],"time_range":{"start":"13:00","end":"18:00"}}]
}
```

**Available starts do not consider movable tasks, their previous placements, or constraints.** The answer includes a `note` explaining this; `solve` makes the final scheduling decision.

## Apply and solve

Query person and fixed appointment identifiers first, narrowing ambiguous names with dates and participants. Reference those identifiers in `apply`; command references use identifiers, not names. To create a movable task, save this as `commands.json` and run `apply`:

```json
{
  "commands":[{
    "kind":"add_task",
    "task":{"name":"Review","duration":"PT1H","participant_ids":["takahashi","ito"],"importance":"medium","required":true,"stability":"normal"}
  }]
}
```

`add_task` with `start` creates a fixed appointment; without `start` it creates a movable task. Register an absence missing from the calendar, such as a health check, half day off, external training, or dentist appointment, as one fixed task for that person:

```json
{
  "commands":[{
    "kind":"add_task",
    "task":{"name":"Health check","start":"2026-10-19T10:30:00+09:00","duration":"PT1H30M","participant_ids":["ito"]}
  }]
}
```

Fixed tasks occupy their participants' time for `available_starts` and `solve`, including for tasks added later. Their starts and durations may fall between slot boundaries; every touched slot is occupied. Omit `importance`, `required`, and `stability` for fixed tasks. Use `remove_task` with the appointment's identifier to restore that time.

Prefer a fixed task over `avoid` constraints for such absences. An `avoid` constraint covers only the tasks listed when it is added, so tasks added later ignore it, and `available_starts` still reports the absent time as free. A fixed task covers every task, now and later, and shows up in the `tasks` query under its name. Recurring absences, such as every Tuesday 9:30 to 10:00, become one fixed task per occurrence. A half day off is strictly a change of working hours, which commands cannot edit; register it as a fixed task too.

Use `avoid` instead when the time is not taken but only some tasks must stay out of it, such as "no new meetings on Tuesday afternoon, focused work is fine": list just the meeting tasks.

The `executed` response supplies `task_id` and `name` for either shape. `add_task` and `add_constraint` accept an optional `id`, such as `"id":"review"`. It must be a non-empty string unique among tasks or among constraints, respectively. Later commands in the same batch can reference a given identifier, so a task and its constraints can be added together. Without `id`, an identifier is generated and is known only from the response, so constrain the task in a second batch. See the constraint examples below.

`apply` saves the entire accepted batch or rejects it without changing state. Other command kinds are `replace_task`, `remove_task`, `add_constraint`, `replace_constraint`, and `remove_constraint`; their complete input shapes are in `schema apply`. `replace_task` takes a task with its identifier, so a queried task can be edited and sent back.

To fix an existing movable task at a given time, use `replace_task` with its identifier and the fixed shape:

```json
{
  "commands":[{
    "kind":"replace_task",
    "task":{"id":"<task identifier>","name":"Review","start":"2026-10-19T10:30:00+09:00","duration":"PT1H","participant_ids":["ito"]}
  }]
}
```

The task moves into `fixed_tasks` and keeps constraints referencing its identifier. To make it movable again, replace the same identifier without `start`, supplying `importance`, `required`, and `stability`. Both commands reject input mixing fields from the fixed and movable shapes.

Run `solve` after applying changes. It prints `{"summary":{...},"items":[...]}` with scheduled entries first in start order, then dropped entries, using the same entry shape as `previous_schedule`, and saves those entries as the previous schedule. The summary has `total_cost`, `costs` split into `dropped_tasks`, `soft_constraints`, and `stability`, and `counts` of `scheduled_tasks`, `dropped_tasks`, `violated_soft_constraints`, and `moved_tasks`. Counts concern movable tasks; fixed appointments contribute to constraint costs when referenced. For example, two scheduled tasks, one dropped task, and one normal soft constraint violated by half an hour can produce:

```json
{
  "summary":{
    "total_cost":7.5,
    "costs":{"dropped_tasks":5.0,"soft_constraints":2.5,"stability":0.0},
    "counts":{"scheduled_tasks":2,"dropped_tasks":1,"violated_soft_constraints":1,"moved_tasks":0}
  },
  "items":[
    {"status":"scheduled","task_id":"review","name":"Review","start":"2026-10-19T09:00:00+09:00","end":"2026-10-19T10:00:00+09:00","participant_ids":["ito","takahashi"]},
    {"status":"scheduled","task_id":"planning","name":"Planning","start":"2026-10-19T10:00:00+09:00","end":"2026-10-19T11:00:00+09:00","participant_ids":["ito"]},
    {"status":"dropped","task_id":"optional","name":"Optional discussion"}
  ]
}
```

The objective adds three costs: the importance-based drop cost of every dropped optional task; each soft constraint's strength weight times its violation, with daily count violations also multiplied by `per_count`; and stability for each scheduled task with a previous start. For both optional and required tasks, stability costs `weight * hours / (1 + weight * hours / limit)`, where `weight` is the task's stability strength weight, `hours` is the distance moved in either direction, and `limit` is `stability_drop_cost_ratio` times its importance-based drop cost. Small moves cost about `weight * hours`; the cost grows with every further hour but stays below `limit`, so a nearer start is always cheaper and moving is always cheaper than dropping. With the default policy, moving a normal-stability high-importance task costs about 4.55 for one hour, 25 for ten hours, and 46.2 for five days. Hard constraints must have zero violation. Unscheduled tasks have no constraint or stability cost. A task previously dropped has no previous start.

For a small trade-off using the current default policy, dropping a low-importance optional task costs 5, and violating a normal soft constraint costs 5 per hour. Scheduling it with a 1.5-hour violation costs 7.5, so dropping it is cheaper if everything else stays equal. At one hour the costs tie and either result is possible. Query `objective_policy` for the current coefficients rather than assuming these example values.

`solve --no-stability` solves from scratch. Stability cost and moved count are zero with this option or without a previous schedule. Dropped tasks do not count as moved. An infeasible solve leaves the previous schedule intact. Use `evaluation` after solving to inspect the constraints behind the reported soft cost.

An infeasible solve exits with status 2 and solves again with relaxed rules: each hour or count of hard violation costs `hard_violation_weight`, and each dropped required task costs `required_drop_cost`. Both exceed every soft cost, and dropping every task always satisfies the relaxed rules, so this solve always has a solution; if a solver time limit stops it before finding one, `conflicts` is `null`. Its schedule is neither printed nor saved; the output lists what it gave up:

```json
{
  "infeasible":true,
  "conflicts":{
    "constraints":[{
      "constraint_id":"review-start",
      "label":"Start the review on Tuesday or later",
      "requirement":{"kind":"hard"},
      "violation":{"amount":8.0,"unit":"hours"},
      "breakdown":[{"task_id":"review","violation":{"amount":8.0,"unit":"hours"}}],
      "related_constraint_ids":["review-deadline"]
    }],
    "dropped_required_tasks":[{"task_id":"offsite","name":"Offsite","reason":"no_free_start"}]
  }
}
```

`constraints` lists the hard constraints broken by the relaxed schedule, in the `evaluation` item shape and measured the same way, plus `related_constraint_ids`: the other hard constraints referencing any of the same tasks. `dropped_required_tasks` lists the required tasks it dropped. `reason` is `no_free_start` when no start has every participant available and free of fixed appointments, as `available_starts` checks over the whole horizon, and `conflict` otherwise. The relaxed schedule is one way that breaks the least, not a list of everything involved in a conflict: of two contradictory deadlines, only one may appear, so check `related_constraint_ids`. Relaxing everything listed, by making constraints soft, making tasks optional, or removing them, makes the problem solvable.

## Constraints

Use `add_constraint` with a `constraint` containing `requirement` and `condition`. The optional `id` supplies your own identifier; otherwise one is generated. The response returns `constraint_id`. The optional `label` describes the request and has no effect on scheduling. Query `constraints` to read these fields before editing.

A requirement is `{"kind":"hard"}` or `{"kind":"soft","strength":"normal"}`. Hard requires zero violation when a task is scheduled. It does not require the task's placement; set `required: true` on a movable task to require scheduling it. Soft permits violations and adds their weighted cost to the solve objective. Strength is `weak`, `normal`, or `strong`, in increasing penalty weight. A strong preference can still be violated. Preferences compete with each other, dropping optional tasks, and moving previous placements.

Choose one of four conditions:

| Condition | Fields and meaning |
|---|---|
| `time_window` | `task_ids`, `relation`, `windows`. `within` keeps each whole task inside the combined windows. `avoid` forbids overlap with them. These apply to the whole task, not just its start. |
| `time_bound` | `task_ids`, `boundary`, `relation`, `at`. Boundary is `start` or `end`. `at_or_before` is an inclusive latest time; `at_or_after` is an inclusive earliest time; `at` is an exact time. |
| `task_gap` | `from_task_id`, `to_task_id`, `relation`, `gap`. The gap runs from the first task's end to the second task's start. `at_least` sets an inclusive minimum; `exactly` sets equality. Use a non-negative duration such as `PT30M`. `PT0S` means no gap. |
| `daily_limit` | `task_ids`, `quantity`, `maximum`. `count` takes a non-negative integer maximum. `total_duration` takes a non-negative duration string such as `PT4H`. Both maxima are inclusive; zero is allowed. |

Every task reference must be an existing movable or fixed task identifier, including tasks added earlier in the same batch. A `task_ids` list must be non-empty. Group tasks sharing a condition in one constraint. Unscheduled tasks have no condition violation and contribute zero to a daily limit. A task gap applies only when both tasks are scheduled.

Soft `time_window` violations sum hours outside the windows for `within`, or hours overlapping them for `avoid`. Soft `time_bound` violations sum hours late, early, or away from the target for the three relations, respectively. Soft `task_gap` violations are hours short of the minimum or hours away from the exact gap. Soft `daily_limit` violations sum excess counts or hours across days.

Daily limits group tasks by their start date in the calendar horizon's starting offset. The whole duration belongs to that date, even when a task crosses midnight. Only the listed tasks count; fixed appointments count only if listed. There is no person field on a constraint. Query that person's movable meeting tasks and list their identifiers to cap new meetings. Tasks added later need to be added to the list with `replace_constraint`.

Use a date and time with an offset for `time_bound.at`, such as `2026-10-19T17:00:00+09:00`. Bounds, gaps, and duration maxima are compared without rounding and need not align to slots. Movable tasks still start on slot boundaries. A hard exact start between slot boundaries is infeasible if the task is required. Conditions on fixed tasks use their occupied intervals rounded outward to slots, including boundaries, gaps, and daily durations and start dates.

Within each window, `date_range`, `weekdays`, and `time_range` combine with **and**; separate windows combine with **or**. Omitted or null fields mean the whole horizon, every weekday, or the whole day respectively. A date range uses `{"start":"2026-10-12","end":"2026-10-24"}` and excludes the end date. Weekday names are lowercase English names: `monday`, `tuesday`, `wednesday`, `thursday`, `friday`, `saturday`, `sunday`. An empty weekday array matches no day. Time ranges include the start and exclude the end; `end: null` means the end of that day. End must be later than start; split overnight ranges into separate windows.

Times of day must have no offset and are interpreted in the calendar horizon's starting offset. Resolve words such as “tomorrow” into dates yourself. Windows are clipped to the horizon and overlapping or adjacent intervals are merged before rounding: `within` rounds inward to complete slots, while `avoid` rounds outward to include touched slots. A changed boundary produces a rounding `note` beside the returned `constraint_id` on add or replace. An empty expansion is rejected. The state keeps the entered windows; rounding is applied when solving.

After the whole batch is applied, `add_constraint` and `replace_constraint` with a `time_window` condition check each listed movable task against the starts the `available_starts` query would return for its participants and duration, including fixed tasks and tasks from the same batch. If no such start keeps the whole task within the rounded windows, or out of them for `avoid`, that command's entry gets `warnings`, a list of messages naming the task. A constraint replaced or removed later in the same batch is not checked for that earlier command; only the version the batch keeps can warn. The batch is still saved and `apply` exits 0. A hard constraint then keeps the task unscheduled, or makes `solve` infeasible if the task is required; a soft constraint is violated whenever the task is scheduled.

Each example below is a separate `apply` input. Replace task placeholders with identifiers from `query` or `apply`.

Finish by 17:00 on October 19, including exactly 17:00:

```json
{
  "commands":[{
    "kind":"add_constraint",
    "constraint":{
      "id":"review-deadline",
      "label":"Finish the review by Monday at 17:00",
      "requirement":{"kind":"hard"},
      "condition":{"kind":"time_bound","task_ids":["<task identifier>"],"boundary":"end","relation":"at_or_before","at":"2026-10-19T17:00:00+09:00"}
    }
  }]
}
```

Start no earlier than 10:30 on October 19. For an exact start request, change the relation to `at`:

```json
{
  "commands":[{
    "kind":"add_constraint",
    "constraint":{
      "requirement":{"kind":"hard"},
      "condition":{"kind":"time_bound","task_ids":["<task identifier>"],"boundary":"start","relation":"at_or_after","at":"2026-10-19T10:30:00+09:00"}
    }
  }]
}
```

Put the second task after the first. It may start exactly when the first ends:

```json
{
  "commands":[{
    "kind":"add_constraint",
    "constraint":{
      "requirement":{"kind":"hard"},
      "condition":{"kind":"task_gap","from_task_id":"<first task identifier>","to_task_id":"<second task identifier>","relation":"at_least","gap":"PT0S"}
    }
  }]
}
```

Prefer starting right after a fixed appointment. Query the appointment's identifier first. This is a soft preference, so other start times are allowed:

```json
{
  "commands":[{
    "kind":"add_constraint",
    "constraint":{
      "requirement":{"kind":"soft","strength":"normal"},
      "condition":{"kind":"task_gap","from_task_id":"<fixed appointment identifier>","to_task_id":"<task identifier>","relation":"exactly","gap":"PT0S"}
    }
  }]
}
```

At most two new meetings per day for Takahashi. Query `tasks` with `{"type":"task","participant_ids_all":["takahashi"]}` as the filter, then identify the meetings by name. Include every intended meeting identifier; check `total` and `truncated` to avoid missing tasks:

```json
{
  "commands":[{
    "kind":"add_constraint",
    "constraint":{
      "requirement":{"kind":"hard"},
      "condition":{"kind":"daily_limit","task_ids":["<first meeting identifier>","<second meeting identifier>","<third meeting identifier>"],"quantity":"count","maximum":2}
    }
  }]
}
```

Prefer Friday between 13:00 and 18:00. The whole task should fit within that time:

```json
{
  "commands":[{
    "kind":"add_constraint",
    "constraint":{
      "requirement":{"kind":"soft","strength":"normal"},
      "condition":{"kind":"time_window","task_ids":["<task identifier>"],"relation":"within","windows":[{"weekdays":["friday"],"time_range":{"start":"13:00","end":"18:00"}}]}
    }
  }]
}
```

Keep these tasks out of October 16 between 09:00 and 12:00. For a person's absence, register a fixed task instead, as described above:

```json
{
  "commands":[{
    "kind":"add_constraint",
    "constraint":{
      "requirement":{"kind":"hard"},
      "condition":{"kind":"time_window","task_ids":["<first task identifier>","<second task identifier>"],"relation":"avoid","windows":[{"date_range":{"start":"2026-10-16","end":"2026-10-17"},"time_range":{"start":"09:00","end":"12:00"}}]}
    }
  }]
}
```

Edit the earlier deadline to October 20 at 17:00. `replace_constraint` needs the existing `id` and complete new content, not just the changed fields. Include `label` to retain or change it; omitting it clears the label:

```json
{
  "commands":[{
    "kind":"replace_constraint",
    "constraint":{
      "id":"review-deadline",
      "label":"Finish the review by Tuesday at 17:00",
      "requirement":{"kind":"hard"},
      "condition":{"kind":"time_bound","task_ids":["<task identifier>"],"boundary":"end","relation":"at_or_before","at":"2026-10-20T17:00:00+09:00"}
    }
  }]
}
```

Remove a constraint with its identifier:

```json
{"commands":[{"kind":"remove_constraint","constraint_id":"review-deadline"}]}
```

Removing a task drops it from conditions with `task_ids`; a condition disappears when its last task is removed. A task gap disappears when either task is removed.

## Chat demonstration

```bash
uv run intent-to-schedule --state example-state.json chat \
  "Schedule a review on Friday afternoon" --model <MODEL>
```

`chat` is a light demonstration loop using OpenAI. Set `OPENAI_API_KEY` and, for a different service endpoint, `OPENAI_BASE_URL`. `--model` is required; `--now` accepts a date and time for relative words, defaulting to the horizon start. Each turn allows up to 12 translator steps: query reads the working problem, apply attempts a batch and passes its result to the next step, solve ends with a scheduling result, and message ends with text. Query and apply results inform subsequent steps. A successful chat solve prints the same `{"summary":{...},"items":[...]}` result as `solve`, honoring the solve step's stability setting.

On a successful solve, chat saves the working problem and previous schedule. On an infeasible solve, it saves the working problem, retains the previous schedule, and prints the same `{"infeasible":true,"conflicts":{...}}` result as `solve`. A message or step limit saves no problem changes. These completed turns save the user request and final assistant text in dialogue; intermediate step records are not saved. A message prints `{"message":"..."}`; the step limit prints `{"exhausted":true}`.

## Exit status

| Status | Meaning |
|---|---|
| 0 | Success, including empty queries, messages, and help |
| 1 | Rejection (`rejected` contains messages), input or runtime error (`error` contains text), or chat step limit |
| 2 | Infeasible solve or chat solve (`{"infeasible":true,"conflicts":{...}}`) |
