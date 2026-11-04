"""Short-lived LangGraph control state for the Step 6D candidate loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from pmmc.compiler.ids import canonical_content_hash
from pmmc.schemas import DoubterAction


class DoubterGraphState(TypedDict):
    actions: list[str]
    max_round: int
    round_id: int
    trace: list[str]
    terminal: str | None


@dataclass(frozen=True)
class DoubterControlTrace:
    action_count: int
    last_round: int
    terminal: str
    trace_hash: str


def run_doubter_control_graph(
    actions: list[DoubterAction],
    *,
    max_round: int = 3,
    max_rounds_exhausted: bool = False,
) -> DoubterControlTrace:
    """Replay one candidate's persisted actions through the frozen control graph."""

    if not actions:
        raise ValueError("Doubter control graph requires at least one persisted action")
    if max_round != 3:
        raise ValueError("Step 6D control graph freezes max_round=3")
    if max_rounds_exhausted and (
        len(actions) != max_round + 1
        or actions[-1] is not DoubterAction.REJECT_CANDIDATE
    ):
        raise ValueError("exhausted control traces require a terminal round-3 rejection")
    try:
        from langgraph.graph import END, START, StateGraph
    except ModuleNotFoundError as exc:
        raise RuntimeError("LangGraph is required for the Step 6D control loop") from exc

    def append(state: DoubterGraphState, value: str) -> DoubterGraphState:
        return {**state, "trace": [*state["trace"], value]}

    def plan(state: DoubterGraphState) -> DoubterGraphState:
        return append(state, f"plan_{state['round_id']}")

    def dry_run(state: DoubterGraphState) -> DoubterGraphState:
        return append(state, f"dry_run_{state['round_id']}")

    def doubt(state: DoubterGraphState) -> DoubterGraphState:
        round_id = state["round_id"]
        if round_id >= len(state["actions"]):
            raise ValueError("Doubter action sequence has a round gap")
        action = state["actions"][round_id]
        return append(state, f"feedback_{round_id}:{action}")

    def route(state: DoubterGraphState) -> str:
        action = DoubterAction(state["actions"][state["round_id"]])
        if action is DoubterAction.PASS:
            return "pass"
        if action is DoubterAction.REJECT_CANDIDATE:
            if max_rounds_exhausted and state["round_id"] == state["max_round"]:
                return "exhausted"
            return "reject"
        if state["round_id"] >= state["max_round"]:
            return "exhausted"
        if state["round_id"] + 1 >= len(state["actions"]):
            return "revision_unavailable"
        return "revise"

    def revise(state: DoubterGraphState) -> DoubterGraphState:
        value = append(state, f"revise_to_{state['round_id'] + 1}")
        return {**value, "round_id": state["round_id"] + 1}

    def terminal(value: str):  # type: ignore[no-untyped-def]
        def node(state: DoubterGraphState) -> DoubterGraphState:
            updated = append(state, f"terminal:{value}")
            return {**updated, "terminal": value}

        return node

    graph = StateGraph(DoubterGraphState)
    graph.add_node("plan", plan)
    graph.add_node("dry_run", dry_run)
    graph.add_node("doubt", doubt)
    graph.add_node("revise", revise)
    for name in ("pass", "reject", "exhausted", "revision_unavailable"):
        graph.add_node(name, terminal(name))
        graph.add_edge(name, END)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "dry_run")
    graph.add_edge("dry_run", "doubt")
    graph.add_conditional_edges(
        "doubt",
        route,
        {
            "pass": "pass",
            "reject": "reject",
            "exhausted": "exhausted",
            "revision_unavailable": "revision_unavailable",
            "revise": "revise",
        },
    )
    graph.add_edge("revise", "plan")
    result = graph.compile().invoke(
        {
            "actions": [item.value for item in actions],
            "max_round": max_round,
            "round_id": 0,
            "trace": [],
            "terminal": None,
        }
    )
    terminal_value = result.get("terminal")
    if not isinstance(terminal_value, str):
        raise RuntimeError("Doubter control graph did not reach a terminal state")
    consumed = int(result["round_id"]) + 1
    if consumed != len(actions):
        raise ValueError("Doubter action sequence contains facts after a terminal action")
    trace = list(result["trace"])
    return DoubterControlTrace(
        action_count=len(actions),
        last_round=int(result["round_id"]),
        terminal=terminal_value,
        trace_hash=canonical_content_hash(trace),
    )


__all__ = [
    "DoubterControlTrace",
    "DoubterGraphState",
    "run_doubter_control_graph",
]
