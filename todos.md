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
  - `WiP` - The task is currently being worked on
- `Priority` - The priority of the task, one of:
  - `High` - The task is of high priority
  - `Medium` - The task is of medium priority
  - `Low` - The task is of low priority
- `Complexity` - The complexity of the task, one of:
  - `Easy` - The task is easy
  - `Medium` - The task is of medium complexity
  - `Hard` - The task is hard
- `Risk` - The risks associated with the task
  - `Low` - The task has low risk
  - `Medium` - The task has medium risk
  - `High` - The task has high risk
- `Description` - A description of the task

## Current

| ID  | Depends on | Task               | Status  | Priority | Description                                    | Complexity | Risk |
| --- | ---------- | ------------------ | ------- | -------- | ---------------------------------------------- | ---------- | ---- |
| 4   |            | Add LARS optimizer | Backlog | Low      | Add the LARS optimizer                         | Easy       |      |
| 7   |            | Early stopping     | Backlog | High     | Add slope-based early stopping                 | Low        |      |
| 3   |            | New datasets       | Backlog | Medium   | Should evaluate on Audioset and other datasets | Medium     |      |

## Backlog
| ID  | Depends on | Task                  | Status  | Priority | Description                                           | Complexity | Risk |
| --- | ---------- | --------------------- | ------- | -------- | ----------------------------------------------------- | ---------- | ---- |
| 1   |            | Replace encoder       | Backlog | Medium   | We should replace the encoder with Vebjørn's new one. | Medium     |      |
| 2   |            | Rework decoder        | Backlog | Medium   | Need to rework number of layers, layer size etc.      | Medium     |      |
| 5   | 1,2,3,6,7  | Run hyperparam search | Backlog | High     | Need to generate results for the report               | High       |      |
| 6   |            | Split main            | Backlog | Medium   | Need to split into train.py, hyperparam.py            | Medium     |      |


## Completed

| ID  | Depends on | Task                  | Status | Priority | Description                              | Complexity | Risk |
| --- | ---------- | --------------------- | ------ | -------- | ---------------------------------------- | ---------- | ---- |
| 0   |            | Make linear eval work | Done   | High     | Need to make the linear evaluation work. | Medium     |      |