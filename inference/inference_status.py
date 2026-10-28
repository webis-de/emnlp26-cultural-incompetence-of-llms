import json
import os
import sqlite3


def count_completed_tasks(config_path, db_dir="."):
    with open(config_path, "r") as f:
        cfg = json.load(f)

    models = cfg.get("models", {})
    total_completed = 0
    total_tasks = 0
    print(f"{'Database Name':<30} | {'Progress':<15}")
    print("-" * 50)

    for model_name in models.keys():
        safe_name = model_name.replace("/", "_").replace(".", "_")
        db_path = os.path.join(db_dir, f"{safe_name}.db")

        if not os.path.exists(db_path):
            raise FileNotFoundError(f"{db_path} doesnt exist!")

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM tasks WHERE status = 'completed'")
            count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM tasks")
            total = cursor.fetchone()[0]

            total_completed += count
            total_tasks += total

            print(f"{safe_name:<30} | {count} / {total}")

            conn.close()
        except sqlite3.Error as e:
            print(f"{safe_name:<30} | Error: {e}")

    print("-" * 50)
    if total_tasks > 0:
        progress = (total_completed / total_tasks) * 100
        print(f"OVERALL PROGRESS: {total_completed} / {total_tasks} ({progress:.2f}%)")
    else:
        print("No tasks found in any database.")


if __name__ == "__main__":
    count_completed_tasks(
        "./inference/inference_config.json",
        "./inference_workspace/inference/databases/",
    )
