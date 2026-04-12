#!/usr/bin/env python3
import os
import sys
import json
import threading
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
import gradio as gr
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TaskStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Task(BaseModel):
    id: str
    model_type: str
    status: str
    created_at: str
    inputs: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class TaskManager:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.lock = threading.Lock()
        self._save_file = "tasks.json"
        self._load_tasks()

    def _load_tasks(self):
        if os.path.exists(self._save_file):
            try:
                with open(self._save_file, "r") as f:
                    tasks_data = json.load(f)
                    for task_data in tasks_data:
                        self.tasks[task_data["id"]] = Task(**task_data)
            except Exception:
                pass

    def _save_tasks(self):
        with self.lock:
            try:
                with open(self._save_file, "w") as f:
                    json.dump(
                        [task.model_dump() for task in self.tasks.values()], f, indent=2
                    )
            except Exception:
                pass

    def add_task(self, model_type: str, inputs: Dict[str, Any]) -> str:
        task_id = str(uuid.uuid4())[:8]
        with self.lock:
            self.tasks[task_id] = Task(
                id=task_id,
                model_type=model_type,
                status=TaskStatus.PENDING,
                created_at=datetime.now().isoformat(),
                inputs=inputs,
            )
        self._save_tasks()
        return task_id

    def update_task(
        self,
        task_id: str,
        status: str,
        result: Optional[Dict] = None,
        error: Optional[str] = None,
    ):
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].status = status
                if result is not None:
                    self.tasks[task_id].result = result
                if error is not None:
                    self.tasks[task_id].error = error
        self._save_tasks()

    def get_tasks(self) -> List[Task]:
        with self.lock:
            return list(self.tasks.values())

    def get_task(self, task_id: str) -> Optional[Task]:
        with self.lock:
            return self.tasks.get(task_id)

    def clear_completed(self):
        with self.lock:
            completed_ids = [
                tid
                for tid, task in self.tasks.items()
                if task.status == TaskStatus.COMPLETED
            ]
            for tid in completed_ids:
                del self.tasks[tid]
        self._save_tasks()


task_manager = TaskManager()


def run_task_async(task_id: str, model_type: str):
    task_manager.update_task(task_id, TaskStatus.RUNNING)
    task = task_manager.get_task(task_id)
    if not task:
        task_manager.update_task(task_id, TaskStatus.FAILED, error="Task not found")
        return

    try:
        import time

        time.sleep(2)

        result_data = {
            "id": task.inputs.get("siRNA_id", task_id),
            "sense_seq": task.inputs.get("sense_seq", ""),
            "anti_seq": task.inputs.get("anti_seq", ""),
            "result": 0.85,
            "note": f"Mock result for {model_type} - model not loaded in demo mode",
            "timestamp": datetime.now().isoformat(),
        }
        task_manager.update_task(task_id, TaskStatus.COMPLETED, result=result_data)
    except Exception as e:
        task_manager.update_task(task_id, TaskStatus.FAILED, error=str(e))


def submit_task(
    model_type: str, siRNA_id: str, sense_seq: str, anti_seq: str, **kwargs
) -> tuple:
    if not siRNA_id:
        return "Error: Please enter siRNA ID", ""
    if not sense_seq:
        return "Error: Please enter sense sequence", ""
    if not anti_seq:
        return "Error: Please enter anti-sense sequence", ""

    inputs = {
        "siRNA_id": siRNA_id,
        "sense_seq": sense_seq,
        "anti_seq": anti_seq,
        **kwargs,
    }
    task_id = task_manager.add_task(model_type, inputs)
    thread = threading.Thread(target=run_task_async, args=(task_id, model_type))
    thread.start()

    return f"Task submitted! Task ID: {task_id}", task_id


def get_tasks_table() -> str:
    tasks = task_manager.get_tasks()
    if not tasks:
        return "No tasks yet."

    rows = []
    for task in sorted(tasks, key=lambda x: x.created_at, reverse=True):
        status_icons = {
            TaskStatus.PENDING: "⏳",
            TaskStatus.RUNNING: "🔄",
            TaskStatus.COMPLETED: "✅",
            TaskStatus.FAILED: "❌",
        }
        status_icon = status_icons.get(task.status, "❓")

        result_str = ""
        if task.status == TaskStatus.COMPLETED and task.result:
            result_val = task.result.get("result")
            if result_val is not None:
                result_str = f"{result_val}"
        elif task.status == TaskStatus.FAILED:
            result_str = task.error or "Unknown error"

        model_short = "ENsiRNA" if task.model_type == "ENsiRNA" else "ENsiRNA-Mod"

        rows.append(
            f"| {task.id} | {model_short} | {status_icon} {task.status} | {task.created_at[:19]} | {result_str} |"
        )

    header = "| Task ID | Model | Status | Created At | Result |\n|---|---|---|---|---|"
    return header + "\n" + "\n".join(rows)


def refresh_tasks():
    return get_tasks_table()


def clear_completed_tasks():
    task_manager.clear_completed()
    return get_tasks_table()


def get_result_json(task_id: str) -> str:
    task = task_manager.get_task(task_id)
    if not task:
        return "Task not found"

    if task.status == TaskStatus.COMPLETED and task.result:
        return json.dumps(task.result, indent=2)
    elif task.status == TaskStatus.FAILED:
        return f"Error: {task.error}"
    elif task.status == TaskStatus.RUNNING:
        return "Task is still running..."
    else:
        return "Task status: " + task.status


def validate_seq(seq: str) -> bool:
    valid_bases = set("ACGUacgu")
    return all(c in valid_bases for c in seq.strip())


def create_ui():
    with gr.Blocks(title="ENsiRNA Web UI", theme=gr.themes.Soft()) as app:
        gr.Markdown("# ENsiRNA Prediction Web UI")
        gr.Markdown(
            "Submit siRNA prediction tasks for both ENsiRNA and ENsiRNA-Mod models."
        )
        gr.Markdown("**Note: This is a demo version with mock predictions.**")

        with gr.Tabs():
            with gr.Tab("ENsiRNA"):
                gr.Markdown("### ENsiRNA Prediction")
                gr.Markdown("Predict siRNA efficacy without modifications.")

                with gr.Row():
                    with gr.Column():
                        siRNA_id = gr.Textbox(
                            label="siRNA ID", placeholder="e.g., Test001"
                        )
                        sense_seq = gr.Textbox(
                            label="Sense Sequence (5'->3')",
                            placeholder="e.g., CAGAAAGAGUGUCUCAUCUUA",
                        )
                        anti_seq = gr.Textbox(
                            label="Anti-sense Sequence (5'->3')",
                            placeholder="e.g., UAAGAUGAGACACUCUUUCUGGU",
                        )

                    with gr.Column():
                        mRNA_seq = gr.Textbox(
                            label="mRNA Sequence (optional)",
                            placeholder="61nt mRNA sequence",
                        )
                        position = gr.Slider(
                            minimum=0, maximum=60, value=30, step=1, label="Position"
                        )

                ensiRNA_submit_btn = gr.Button("Submit Task", variant="primary")
                ensiRNA_result = gr.Textbox(
                    label="Submission Result", interactive=False
                )
                ensiRNA_task_id = gr.Textbox(label="Task ID", visible=False)

                def handle_submit_ensiRNA(
                    siRNA_id, sense_seq, anti_seq, mRNA_seq, position
                ):
                    if not validate_seq(sense_seq) or not validate_seq(anti_seq):
                        return (
                            "Error: Invalid sequence. Use only A, C, G, U.",
                            "",
                        )
                    return submit_task(
                        "ENsiRNA",
                        siRNA_id,
                        sense_seq,
                        anti_seq,
                        mRNA_seq=mRNA_seq,
                        position=position,
                    )

                ensiRNA_submit_btn.click(
                    handle_submit_ensiRNA,
                    inputs=[siRNA_id, sense_seq, anti_seq, mRNA_seq, position],
                    outputs=[ensiRNA_result, ensiRNA_task_id],
                )

            with gr.Tab("ENsiRNA-Mod"):
                gr.Markdown("### ENsiRNA-Mod Prediction")
                gr.Markdown("Predict siRNA efficacy with modifications.")

                with gr.Row():
                    with gr.Column():
                        siRNA_id_mod = gr.Textbox(
                            label="siRNA ID", placeholder="e.g., Test001_Mod"
                        )
                        sense_seq_mod = gr.Textbox(
                            label="Sense Sequence (5'->3')",
                            placeholder="e.g., CAGAAAGAGUGUCUCAUCUUA",
                        )
                        anti_seq_mod = gr.Textbox(
                            label="Anti-sense Sequence (5'->3')",
                            placeholder="e.g., UAAGAUGAGACACUCUUUCUGGU",
                        )

                    with gr.Column():
                        gr.Markdown("#### Sense Modifications")
                        sense_mod_1 = gr.Textbox(
                            label="Modification 1 (type:pos)",
                            placeholder="e.g., 2-Fluoro:2,3,4",
                        )
                        sense_mod_2 = gr.Textbox(
                            label="Modification 2 (type:pos)",
                            placeholder="e.g., 2-O-Methyl:1,5,7",
                        )
                        sense_mod_3 = gr.Textbox(
                            label="Modification 3 (type:pos)",
                            placeholder="e.g., Phosphorothioate:2,3",
                        )

                    with gr.Column():
                        gr.Markdown("#### Anti-sense Modifications")
                        anti_mod_1 = gr.Textbox(
                            label="Modification 1 (type:pos)",
                            placeholder="e.g., 2-Fluoro:2,3,4",
                        )
                        anti_mod_2 = gr.Textbox(
                            label="Modification 2 (type:pos)",
                            placeholder="e.g., 2-O-Methyl:1,5,7",
                        )
                        anti_mod_3 = gr.Textbox(
                            label="Modification 3 (type:pos)",
                            placeholder="e.g., Phosphorothioate:2,3",
                        )

                ensiRNA_mod_submit_btn = gr.Button("Submit Task", variant="primary")
                ensiRNA_mod_result = gr.Textbox(
                    label="Submission Result", interactive=False
                )
                ensiRNA_mod_task_id = gr.Textbox(label="Task ID", visible=False)

                def handle_submit_ensiRNA_mod(
                    siRNA_id,
                    sense_seq,
                    anti_seq,
                    sm1,
                    sm2,
                    sm3,
                    am1,
                    am2,
                    am3,
                ):
                    if not validate_seq(sense_seq) or not validate_seq(anti_seq):
                        return (
                            "Error: Invalid sequence. Use only A, C, G, U.",
                            "",
                        )
                    sense_mods = [sm1, sm2, sm3]
                    anti_mods = [am1, am2, am3]
                    return submit_task(
                        "ENsiRNA-Mod",
                        siRNA_id,
                        sense_seq,
                        anti_seq,
                        sense_mods=sense_mods,
                        anti_mods=anti_mods,
                    )

                ensiRNA_mod_submit_btn.click(
                    handle_submit_ensiRNA_mod,
                    inputs=[
                        siRNA_id_mod,
                        sense_seq_mod,
                        anti_seq_mod,
                        sense_mod_1,
                        sense_mod_2,
                        sense_mod_3,
                        anti_mod_1,
                        anti_mod_2,
                        anti_mod_3,
                    ],
                    outputs=[ensiRNA_mod_result, ensiRNA_mod_task_id],
                )

            with gr.Tab("Task Management"):
                gr.Markdown("### Task Management")
                gr.Markdown("View and manage your prediction tasks.")

                with gr.Row():
                    refresh_btn = gr.Button("Refresh", variant="secondary")
                    clear_btn = gr.Button("Clear Completed", variant="secondary")

                tasks_table = gr.Markdown(get_tasks_table())

                refresh_btn.click(refresh_tasks, outputs=tasks_table)
                clear_btn.click(clear_completed_tasks, outputs=tasks_table)

                with gr.Row():
                    gr.Markdown("### View Task Result")
                    check_task_id = gr.Textbox(
                        label="Task ID", placeholder="Enter task ID to check"
                    )
                    check_btn = gr.Button("Check Result")
                    result_output = gr.Textbox(
                        label="Result", interactive=False, lines=10
                    )

                check_btn.click(
                    get_result_json, inputs=check_task_id, outputs=result_output
                )

        gr.Markdown("---")
        gr.Markdown("© 2026 ENsiRNA Web UI | Powered by Gradio")

    return app


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ENsiRNA Web UI")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind")
    parser.add_argument("--share", action="store_true", help="Create share link")
    args = parser.parse_args()

    app = create_ui()
    try:
        app.launch(server_name="0.0.0.0", server_port=7860, share=False)
    except Exception:
        app.launch(server_name="0.0.0.0", server_port=7860, share=True)


if __name__ == "__main__":
    main()
