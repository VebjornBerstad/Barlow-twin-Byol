# Todos

## Column descriptions

- `ID` - The ID of the task
- `Depends on` - The ID(s) of the task that this task depends on
- `Task` - The task to be completed
- `Status` - The current status of the task, one of:
  - `Done` - The task is completed
  - `Dropped` - The task is no longer being worked on
  - `Blocked` - The task is blocked by another task
  - `Waiting` - The task is waiting for another task to be completed
  - `Backlog` - The task is in the backlog
  - `On hold` - The task is on hold
  - `WiP` - The task is currently being worked on
- `Priority` - The priority of the task, one of:
  - `High` - The task is of high priority
  - `Medium` - The task is of medium priority
  - `Low` - The task is of low priority
- `Complexity` - The complexity of the task, one of:
  - `Easy` - The task is easy
  - `Medium` - The task is of medium complexity
  - `Hard` - The task is hard
- `Description` - A description of the task

## Current

| ID  | Depends on | Task                  | Status  | Priority | Description                                    | Complexity |
| --- | ---------- | --------------------- | ------- | -------- | ---------------------------------------------- | ---------- |
| 3   |            | New datasets          | Backlog | Medium   | Should evaluate on Audioset and other datasets | Medium     |
| 5   | 1,2,3      | Run hyperparam search | Backlog | High     | Need to generate results for the report        | High       |

## Backlog
| ID  | Depends on | Task               | Status  | Priority | Description                                | Complexity |
| --- | ---------- | ------------------ | ------- | -------- | ------------------------------------------ | ---------- |
| 6   |            | Split main         | Backlog | Medium   | Need to split into train.py, hyperparam.py | Medium     |
| 4   |            | Add LARS optimizer | On hold | Low      | Add the LARS optimizer                     | Easy       |


## Completed

| ID  | Depends on | Task                  | Status  | Priority | Description                                           | Complexity |
| --- | ---------- | --------------------- | ------- | -------- | ----------------------------------------------------- | ---------- |
| 0   |            | Make linear eval work | Done    | High     | Need to make the linear evaluation work.              | Medium     |
| 7   |            | Early stopping        | Done    | High     | Add slope-based early stopping                        | Low        |
| 1   |            | Replace encoder       | Dropped | Medium   | We should replace the encoder with Vebjørn's new one. | Medium     |
| 2   |            | Rework decoder        | Done    | Medium   | Need to rework number of layers, layer size etc.      | Medium     |