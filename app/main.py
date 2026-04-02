from __future__ import annotations

import argparse

try:
    from .env.action import Action
    from .env.environment import MeetingEnv
    from .tasks.easy import load_easy_task
    from .tasks.grader import find_best_slot, grade
    from .tasks.hard import load_hard_task
    from .tasks.medium import load_medium_task
except ImportError:
    from app.env.action import Action
    from app.env.environment import MeetingEnv
    from app.tasks.easy import load_easy_task
    from app.tasks.grader import find_best_slot, grade
    from app.tasks.hard import load_hard_task
    from app.tasks.medium import load_medium_task

TASKS = {
    "easy": load_easy_task,
    "medium": load_medium_task,
    "hard": load_hard_task,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the meeting scheduler environment.")
    parser.add_argument("--task", choices=TASKS.keys(), default="easy", help="Task level to evaluate.")
    parser.add_argument("--slot", help="Slot to evaluate. Defaults to the grader's best slot.")
    args = parser.parse_args()

    env = MeetingEnv(TASKS[args.task])
    current_state = env.reset()
    chosen_slot = args.slot or find_best_slot(current_state)
    step_result = env.step(Action(chosen_slot=chosen_slot))
    final_score = grade(current_state, chosen_slot)

    print("state:")
    print(current_state.model_dump_json(indent=2))
    print(f"chosen slot: {chosen_slot}")
    print(f"reward: {step_result['reward']:.4f}")
    print(f"final score: {final_score:.4f}")


if __name__ == "__main__":
    main()
