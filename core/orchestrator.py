"""
APEX Multi-Agent Orchestration
DAG-based task decomposition and parallel execution.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any
from enum import Enum

from apex.core.agent import Agent, AgentResult

logger = logging.getLogger("apex.orchestrator")


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    goal: str = ""
    context: str = ""
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    error: str = ""
    dependencies: list[str] = field(default_factory=list)  # task IDs
    subagent_role: str = "worker"  # worker, reviewer, researcher
    model_override: str | None = None
    timeout: int = 300


@dataclass
class DAG:
    """Directed Acyclic Graph of tasks."""
    tasks: dict[str, Task] = field(default_factory=dict)

    def add(self, task: Task) -> str:
        self.tasks[task.id] = task
        return task.id

    def get_ready_tasks(self) -> list[Task]:
        """Get tasks whose dependencies are all completed."""
        ready = []
        for task in self.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            deps_met = all(
                self.tasks.get(dep_id, Task()).status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
            )
            if deps_met:
                ready.append(task)
        return ready

    def is_complete(self) -> bool:
        return all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED)
            for t in self.tasks.values()
        )

    def has_failures(self) -> bool:
        return any(t.status == TaskStatus.FAILED for t in self.tasks.values())


class Orchestrator:
    """
    Manages multi-agent workflows using DAG scheduling.

    Supports:
    - Parallel task execution
    - Dependency management
    - Role-based subagent assignment
    - Result aggregation
    """

    def __init__(self, agent_factory, max_parallel: int = 3):
        """
        Args:
            agent_factory: Callable that returns a new Agent instance
            max_parallel: Max concurrent subagents
        """
        self.agent_factory = agent_factory
        self.max_parallel = max_parallel
        self._results: dict[str, AgentResult] = {}

    async def execute_dag(self, dag: DAG) -> dict[str, AgentResult]:
        """Execute a DAG of tasks, respecting dependencies and parallelism."""
        start = time.time()
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def run_task(task: Task):
            async with semaphore:
                task.status = TaskStatus.RUNNING
                logger.info(f"[Orchestrator] Starting task {task.id}: {task.goal[:60]}")

                agent = self.agent_factory()
                try:
                    result = await asyncio.wait_for(
                        agent.run(
                            user_message=task.goal,
                            system_prompt=self._build_subagent_prompt(task),
                        ),
                        timeout=task.timeout,
                    )
                    task.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
                    task.result = result.content
                    self._results[task.id] = result
                    logger.info(f"[Orchestrator] Task {task.id} completed in {result.duration:.1f}s")
                except asyncio.TimeoutError:
                    task.status = TaskStatus.FAILED
                    task.error = f"Timed out after {task.timeout}s"
                    logger.warning(f"[Orchestrator] Task {task.id} timed out")
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    logger.error(f"[Orchestrator] Task {task.id} failed: {e}")

        while not dag.is_complete():
            ready = dag.get_ready_tasks()
            if not ready:
                # Check if we're stuck (failed deps)
                if dag.has_failures():
                    # Cancel remaining pending tasks
                    for task in dag.tasks.values():
                        if task.status == TaskStatus.PENDING:
                            task.status = TaskStatus.CANCELLED
                    break
                await asyncio.sleep(0.5)
                continue

            # Run all ready tasks in parallel
            await asyncio.gather(*(run_task(t) for t in ready))

        duration = time.time() - start
        logger.info(f"[Orchestrator] DAG complete in {duration:.1f}s")
        return self._results

    def _build_subagent_prompt(self, task: Task) -> str:
        role_prompts = {
            "worker": "You are a focused worker agent. Complete the given task efficiently.",
            "researcher": "You are a research agent. Gather information, search the web, and synthesize findings.",
            "reviewer": "You are a code reviewer. Check for bugs, security issues, and quality problems.",
            "writer": "You are a writing agent. Produce clear, well-structured content.",
        }
        base = role_prompts.get(task.subagent_role, role_prompts["worker"])
        if task.context:
            base += f"\n\n## Context\n{task.context}"
        return base

    async def decompose_and_run(
        self,
        goal: str,
        agent: Agent,
        max_tasks: int = 5,
    ) -> dict[str, AgentResult]:
        """
        Automatically decompose a goal into a DAG and execute it.
        Uses the agent to plan the decomposition.
        """
        # Ask the agent to plan
        plan_prompt = f"""Decompose this goal into parallelizable tasks (max {max_tasks}).

Goal: {goal}

Respond with JSON:
{{
  "tasks": [
    {{
      "id": "task1",
      "goal": "specific task description",
      "role": "worker|researcher|reviewer|writer",
      "dependencies": []
    }}
  ]
}}

Rules:
- Tasks with no dependency can run in parallel
- Keep tasks focused and independent where possible
- Use "researcher" for information gathering
- Use "reviewer" for quality checks
- Use "writer" for content creation
- Use "worker" for general tasks
"""

        result = await agent.run(plan_prompt)
        try:
            # Extract JSON from response
            import json, re
            json_match = re.search(r"\{.*\}", result.content, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group())
            else:
                plan = {"tasks": [{"id": "task1", "goal": goal, "role": "worker", "dependencies": []}]}
        except Exception:
            plan = {"tasks": [{"id": "task1", "goal": goal, "role": "worker", "dependencies": []}]}

        # Build DAG
        dag = DAG()
        for t in plan.get("tasks", []):
            dag.add(Task(
                id=t.get("id", str(uuid.uuid4())[:8]),
                goal=t.get("goal", ""),
                subagent_role=t.get("role", "worker"),
                dependencies=t.get("dependencies", []),
            ))

        return await self.execute_dag(dag)
