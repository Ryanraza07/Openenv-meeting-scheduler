---
title: Meeting Scheduler OpenEnv
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Meeting Scheduler OpenEnv

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> A deterministic, reproducible learning environment for training and evaluating AI agents on intelligent meeting slot selection.

## Overview

**Meeting-scheduler-openenv** is an OpenAI Gym-style environment where agents learn to select optimal meeting slots by balancing participant availability, priority scores, and required attendee constraints. The environment is fully deterministic, enabling reliable evaluation and reproducibility across all runs.

### ✨ Key Features

- 🎯 **Deterministic Evaluation**: Fixed task states and reward model ensure 100% reproducible results
- 📊 **Multi-level Difficulties**: Easy, medium, and hard task variants for progressive training
- 🤖 **Agent-Ready**: FastAPI REST API and OpenEnv-Core compatible interface
- 📈 **Comprehensive Grading**: Normalized scoring against mathematical optimal baseline
- 🔍 **Transparent Reasoning**: Detailed slot analysis and decision explanations
- 🚀 **Production Ready**: Docker support with requirements validation

---

## Quick Start

### 1. Installation

```bash
# Navigate to project directory
cd meeting-scheduler-openenv

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Set OpenAI API key for baseline agents
export OPENAI_API_KEY="sk-..."
```

### 2. Run Your First Evaluation

```bash
# Option A: Run inference with baseline agent
python inference.py

# Option B: Local evaluation without external API
python -m app.main --task easy

# Option C: Specify different task difficulty
python -m app.main --task medium  # or 'hard'
```

### 3. Start the API Server

```bash
# Run FastAPI server
python app/server.py

# Server will be available at http://localhost:8000
```

---

## Problem Statement

### Input (MeetingState)

Each task provides a fixed scheduling problem containing:

| Field | Description | Example |
|-------|-------------|---------|
| `participants` | List of attendees | `[Alice, Bob, Carol, Diego]` |
| `all_slots` | Valid time slots | `["2026-04-02T09:00", "2026-04-02T10:00", ...]` |
| `meeting_duration` | Duration in minutes | `45`, `60` |

### Participant Attributes

Each participant includes:

| Attribute | Type | Purpose |
|-----------|------|---------|
| `name` | string | Unique identifier |
| `available_slots` | list[string] | Times they can attend (ISO 8601) |
| `priority` | float | Importance weight (0.0 - 1.0) |
| `required` | bool | Must attend for valid slot selection |

### Task

Agent must select **exactly one slot** from `all_slots`.

### Evaluation Rules

Two core rules determine validity and quality:

1. **Hard Constraint**: If any **required** participant cannot attend → reward = `0.0` ❌
2. **Soft Objective**: Otherwise → reward = weighted priority sum of attendees ✓

---

## Reward Formula

The reward is mathematically defined and deterministic:

$$\text{reward}(s) = \begin{cases} 
0 & \text{if } \exists p \in \text{required} : s \notin p.\text{available\_slots} \\ 
\frac{\sum_{p \in \text{attend}(s)} \text{priority}(p)}{\sum_{p \in \text{all}} \text{priority}(p)} & \text{otherwise}
\end{cases}$$

Where:
- $\text{attend}(s)$ = set of participants available at slot $s$
- Priorities normalize to sum = 1.0

### Example Calculation

**Scenario:**
- Participants: Alice (priority=0.4), Bob (0.25), Carol (0.2), Diego (0.15)
- Slot 10:00: Alice ✓, Bob ✓, Carol ✓, Diego ✗

**Reward:** (0.4 + 0.25 + 0.2) / 1.0 = **0.85**

---

## Grading & Normalization

The grader evaluates agent performance against the mathematical optimum:

### Algorithm

1. **Find Optimal**: Search all slots for best reward
2. **Evaluate Agent**: Compute reward for agent's choice
3. **Normalize**: Apply normalization formula

### Normalization Formula

$$\text{score} = \begin{cases} 
1.0 & \text{if both best and agent reward = 0 (both failed)} \\ 
0.0 & \text{if best = 0 but agent > 0 (constraint violated)} \\ 
\frac{\text{agent\_reward}}{\text{best\_reward}} & \text{otherwise (both valid)}
\end{cases}$$

### Output Metrics

| Metric | Meaning | Range | Formula |
|--------|---------|-------|---------|
| `agent_score` | Agent's slot reward | [0.0, 1.0] | reward(agent_slot) |
| `optimal_score` | Best possible reward | [0.0, 1.0] | reward(best_slot) |
| `normalized_score` | Performance ratio | [0.0, 1.0] | agent / optimal |

### Score Interpretation

| Score | Meaning |
|-------|---------|
| `1.0` | 🏆 Agent found optimal slot |
| `0.5` | ⚡ Agent achieved 50% of optimal |
| `0.1` | ⚠️ Agent made poor choice |
| `0.0` | ❌ Agent violated constraints |

## Why This Approach is Correct

1. **Consistency**: Environment, grader, and inference all use identical reward logic
2. **Transparency**: All evaluation steps are deterministic and fully auditable
3. **Fairness**: Baseline agent uses same constraints and scoring as human judges
4. **Simplicity**: No hidden complexity; rules are explicit and mathematical

---

## API Reference

### Server Setup

```bash
# Start API server (default: http://localhost:8000)
python app/server.py

# Or run with custom port
python app/server.py --port 9000
```

### Endpoints

#### `POST /reset`
Initialize a new task and return initial state.

```bash
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_level": "easy"}'
```

#### `POST /step`
Execute action (choose slot) and receive reward.

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"chosen_slot": "2026-04-02T10:00"}'
```

#### `GET /state`
Retrieve current task state.

```bash
curl http://localhost:8000/state
```

---

## Inference Output

[inference.py](inference.py) generates comprehensive readable reports

```text
==== STATE ====
Meeting duration: 45 minutes
All slots:
   1. 2026-04-02T09:00
   2. 2026-04-02T10:00
   3. 2026-04-02T11:00
   4. 2026-04-02T14:00
Participants:
  Alice | priority=0.4000 | required=no  | available=2026-04-02T09:00, 2026-04-02T10:00, 2026-04-02T14:00
  Bob   | priority=0.2500 | required=no  | available=2026-04-02T10:00, 2026-04-02T11:00
  Carol | priority=0.2000 | required=no  | available=2026-04-02T10:00, 2026-04-02T14:00
  Diego | priority=0.1500 | required=no  | available=2026-04-02T09:00, 2026-04-02T11:00, 2026-04-02T14:00

==== SLOT ANALYSIS ====
Slot: 2026-04-02T09:00
  Attending: Alice(0.4000), Diego(0.1500)
  Missing: Bob(0.2500), Carol(0.2000)
  Required satisfied: Yes
  Score: 0.5500

Slot: 2026-04-02T10:00
  Attending: Alice(0.4000), Bob(0.2500), Carol(0.2000)
  Missing: Diego(0.1500)
  Required satisfied: Yes
  Score: 0.8500

Slot: 2026-04-02T11:00
  Attending: Bob(0.2500), Diego(0.1500)
  Missing: Alice(0.4000), Carol(0.2000)
  Required satisfied: Yes
  Score: 0.4000

Slot: 2026-04-02T14:00
  Attending: Alice(0.4000), Carol(0.2000), Diego(0.1500)
  Missing: Bob(0.2500)
  Required satisfied: Yes
  Score: 0.7500

==== GROUND TRUTH ====
Best slot: 2026-04-02T10:00
Best score: 0.8500

==== AGENT DECISION ====
Chosen slot: 2026-04-02T10:00
Slot validation: valid
Reward: 0.8500
Final score: 1.0000

==== EXPLANATION ====
Chosen slot maximizes weighted attendance while satisfying all constraints.
Attending participants: Alice(0.4000), Bob(0.2500), Carol(0.2000)
Missing participants: Diego(0.1500)
Total priority covered: 0.8500 of 1.0000
Required participants satisfied: Yes

==== FINAL EVALUATION ====
agent_score: 0.8500
optimal_score: 0.8500
normalized_score: 1.0000
```

### Debug Mode

Set `debug = True` in [inference.py](inference.py) to see full analysis for every candidate slot.

---

## Project Structure

```
meeting-scheduler-openenv/
├── app/
│   ├── agent/
│   │   └── baseline_agent.py          # OpenAI-based baseline
│   ├── env/
│   │   ├── action.py                  # Action definitions
│   │   ├── environment.py             # Core MeetingEnv class
│   │   ├── reward.py                  # Reward computation
│   │   └── state.py                   # MeetingState model
│   ├── tasks/
│   │   ├── easy.py                    # Easy difficulty task
│   │   ├── medium.py                  # Medium difficulty task
│   │   ├── hard.py                    # Hard difficulty task
│   │   └── grader.py                  # Evaluation & grading
│   ├── main.py                        # CLI entry point
│   └── server.py                      # FastAPI server
├── inference.py                       # Baseline agent inference
├── openenv.yaml                       # OpenEnv configuration
├── Dockerfile                         # Container definition
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## Usage Examples

### Local Evaluation (No API)

```bash
# Evaluate on easy task
python -m app.main --task easy

# Evaluate on hard task with specific slot
python -m app.main --task hard --slot "2026-04-02T10:00"
```

### Baseline Inference

```bash
# Run OpenAI baseline agent
python inference.py

# Requires: OPENAI_API_KEY environment variable
```

### Docker Deployment

```bash
# Build image
docker build -t meeting-scheduler .

# Run container
docker run --rm \
  --env OPENAI_API_KEY="sk-..." \
  meeting-scheduler

# Or run with interactive shell
docker run --rm -it meeting-scheduler /bin/bash
```

### Remote API Usage

```python
import requests

# Reset environment
state = requests.post("http://localhost:8000/reset").json()

# Choose slot
result = requests.post(
    "http://localhost:8000/step",
    json={"chosen_slot": state["all_slots"][0]}
).json()

print(f"Score: {result['info']['score']:.2%}")
```

---

## Task Difficulties

### Easy ⭐
- 2-4 participants
- 3-4 available slots
- Simple constraints
- **Typical optimal score**: 0.8-1.0

### Medium ⭐⭐
- 5-8 participants
- 6-10 available slots
- Mixed required/optional
- **Typical optimal score**: 0.6-0.9

### Hard ⭐⭐⭐
- 10+ participants
- 10+ available slots
- Complex dependencies
- **Typical optimal score**: 0.3-0.7

---

## Troubleshooting

### Issue: ImportError on startup

**Solution**: Ensure virtual environment is activated and all dependencies installed:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Issue: OPENAI_API_KEY not found

**Solution**: Set environment variable before running inference:
```bash
export OPENAI_API_KEY="sk-..."
python inference.py
```

### Issue: Port 8000 already in use

**Solution**: Specify alternative port:
```bash
python app/server.py --port 9000
```

### Issue: Inconsistent scores across runs

**Problem**: Results should be deterministic. If not, check for:
- Modified task definitions (easy.py, medium.py, hard.py)
- Non-zero temperature in agent sampling
---

## Project Structure

```
meeting-scheduler-openenv/
├── app/
│   ├── agent/
│   │   └── baseline_agent.py          # OpenAI-based baseline
│   ├── env/
│   │   ├── action.py                  # Action definitions
│   │   ├── environment.py             # Core MeetingEnv class
│   │   ├── reward.py                  # Reward computation
│   │   └── state.py                   # MeetingState model
│   ├── tasks/
│   │   ├── easy.py                    # Easy difficulty task
│   │   ├── medium.py                  # Medium difficulty task
│   │   ├── hard.py                    # Hard difficulty task
│   │   └── grader.py                  # Evaluation & grading
│   ├── main.py                        # CLI entry point
│   └── server.py                      # FastAPI server
├── inference.py                       # Baseline agent inference
├── openenv.yaml                       # OpenEnv configuration
├── Dockerfile                         # Container definition
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## Contributing

To extend this environment:

1. **Add new difficulty**: Create `app/tasks/custom.py`
2. **Custom agent**: Implement in `app/agent/`
3. **New actions**: Modify `app/env/action.py`
4. **Reward tweaks**: Update `app/env/reward.py` (⚠️ breaks reproducibility)

---

## References

- OpenAI Gym: https://gym.openai.com/
- Pydantic Models: https://docs.pydantic.dev/
- FastAPI: https://fastapi.tiangolo.com/

---

## License

MIT License - see LICENSE file for details.

---

**Last Updated**: April 2, 2026 | **Status**: ✅ Production Ready
