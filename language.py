import enum
import json
from utils import is_keyboard_action


class NodeType(enum.Enum):
    ACTION = "action"
    SEQUENCE = "sequence"

# %% Action Node

class Time:
    def __init__(self, before: str, after: str, range: float = None, diff: float = None):
        """Initialize the time object associated with an action."""
        self.before = before
        self.after = after
        self.range = range
        # diff is ignored/unused

    @classmethod
    def from_json(cls, path: str = None, data: dict = None):
        if path is not None:
            data = json.load(open(path))
        elif data is None:
            return None
        data = dict(data)
        data.pop("diff", None)
        return cls(**data)
    
    def to_json(self, path: str = None):
        data = {
            "before": self.before,
            "after": self.after,
            "range": self.range,
        }
        if path is not None:
            json.dump(data, open(path, 'w'))
        return data

    def get_time(self, reverse: bool = False) -> str:
        if reverse: return self.after if (self.after is not None) else self.before
        else: return self.before if (self.before is not None) else self.after
    

class State:
    def __init__(self, before: str, after: str, diff_score: float = None):
        """Initialize the state object associated with an action.
        Args:
            before: The screenshot path of the state before the action.
            after: The screenshot path of the state after the action.
            diff_score: The MSE difference score between the before and after states.
        """
        self.before = before
        self.after = after
        self.diff_score = diff_score

    @classmethod
    def from_json(cls, path: str = None, data: dict = None):
        if path is not None:
            data = json.load(open(path))
        elif data is None:
            raise ValueError("Either path or data must be provided.")
        return cls(before=data["before"], after=data["after"], diff_score=data.get("diff_score", None))
    
    def to_json(self, path: str = None):
        data = {"before": self.before, "after": self.after, "diff_score": self.diff_score}
        if path is not None:
            json.dump(data, open(path, 'w'))
        return data
    
    def get_state(self, reverse: bool = False) -> str:
        if reverse: return self.after if (self.after is not None) else self.before
        else: return self.before if (self.before is not None) else self.after


class ActionNode:
    def __init__(self, action: str, state: dict | State, time: dict | Time = None, annotation: dict = None):
        self.node_type = NodeType.ACTION
        self.length = 1
        self.action = action
        self.state = state if isinstance(state, State) else State(**state)
        if time is None:
            self.time = None
        else:
            self.time = time if isinstance(time, Time) else Time(**time)
        self.annotation = annotation

    def __str__(self):
        return f"ActionNode(action={self.action}, state={self.state}, description={self.description})"
    
    def get_semantic_repr(self):
        return self.action
    
    def get_num_actions(self):
        return 1

    @classmethod
    def from_json(cls, path: str = None, data: dict = None):
        if path is not None:
            data = json.load(open(path))
        elif data is None:
            raise ValueError("Either path or data must be provided.")
        state = State.from_json(data=data["state"])
        time = Time.from_json(data=data.get("time", None))
        annotation = data.get("annotation", None)
        return cls(action=data["action"], state=state, time=time, annotation=annotation)

    def to_json(self, path: str = None):
        """Save the action node to a JSON file.
        Args:
            path: The path to save the action node.
        """
        data = {
            "node_type": self.node_type.value,
            "action": self.action,
            "state": self.state.to_json(),
            "time": self.time.to_json() if self.time is not None else None
        }
        if self.annotation is not None:
            data["annotation"] = self.annotation
        if path is not None:
            json.dump(data, open(path, 'w'))
        return data

# %% Sequence Node

class SequenceNode:
    def __init__(
        self,
        nodes: list,
        time: dict | Time = None,
        metadata: dict = None,
    ):
        self.node_type = NodeType.SEQUENCE
        self.nodes = nodes
        self.length = len(self.nodes)

        if time is None:
            self.time = None
        else:
            self.time = time if isinstance(time, Time) else Time(**time)
        self.metadata = metadata
    
    def __str__(self):
        return f"SequenceNode(nnodes={len(self.nodes)})"
    
    def get_semantic_repr(self):
        subgoals = [n.get_semantic_repr() for n in self.nodes]
        return '\n'.join([sg for sg in subgoals if sg is not None])
    
    def get_num_actions(self):
        num_actions = 0
        for n in self.nodes:
            num_actions += n.get_num_actions()
        return num_actions

    @classmethod
    def from_json(cls, path: str = None, data: dict = None):
        if path is not None:
            data = json.load(open(path, 'r'))
        elif data is None:
            raise ValueError("Either path or data must be provided.")

        nodes = []
        for node in data["nodes"]:
            if node["node_type"] == NodeType.ACTION.value:
                nodes.append(ActionNode.from_json(data=node))
            elif node["node_type"] == NodeType.SEQUENCE.value:
                nodes.append(cls.from_json(data=node))

        # Extract metadata (fields not in standard schema)
        metadata = {k: v for k, v in data.items() if k not in ['node_type', 'nodes', 'goal', 'time']}

        return cls(
            nodes=nodes,
            time=Time.from_json(data=data.get("time", None)),
            metadata=metadata if metadata else None,
        )

    def to_json(self, path: str = None):
        """Save the sequence node to a JSON file.
        Args:
            path: The path to save the sequence node.
        """
        data = {
            "node_type": self.node_type.value,
            "nodes": [node.to_json() for node in self.nodes],
        }
        if getattr(self, "time", None) is not None:
            data["time"] = self.time.to_json()
        # Merge metadata fields if present
        if self.metadata:
            data.update(self.metadata)
        if path is not None:
            json.dump(data, open(path, 'w'))
        return data

# % Node Utilities (for dict-based nodes)

import copy
from typing import Any, Sequence


def state_path(node: dict[str, Any], reverse: bool = False) -> str | None:
    """Get the state path (before or after) from a node dict."""
    state = node.get("state") or {}
    first = state.get("after") if reverse else state.get("before")
    second = state.get("before") if reverse else state.get("after")
    return first or second


def clone_node(node: dict[str, Any]) -> dict[str, Any]:
    """Deep copy a node dict."""
    return copy.deepcopy(node)


def wrap_sequence(nodes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Wrap a list of nodes in a sequence node."""
    return {"node_type": "sequence", "nodes": [clone_node(node) for node in nodes]}


def get_first_action(node: dict[str, Any]) -> dict[str, Any] | None:
    """Recursively find the first action node in a node tree."""
    if node.get("node_type") == "action":
        return node
    for child in node.get("nodes", []):
        found = get_first_action(child)
        if found is not None:
            return found
    return None


def get_last_action(node: dict[str, Any]) -> dict[str, Any] | None:
    """Recursively find the last action node in a node tree."""
    if node.get("node_type") == "action":
        return node
    for child in reversed(node.get("nodes", [])):
        found = get_last_action(child)
        if found is not None:
            return found
    return None


def node_length(node: dict[str, Any]) -> int:
    """Recursively compute the total number of action nodes in a node tree."""
    if node.get("node_type") == "action":
        return 1
    return sum(node_length(child) for child in node.get("nodes", []))


def merge_nodes(nodes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Merge nodes while preserving sequence structure and state."""
    merged: list[dict[str, Any]] = []
    for node in nodes:
        # Keep sequences intact instead of flattening them
        merged.append(clone_node(node))
    return wrap_sequence(merged)


