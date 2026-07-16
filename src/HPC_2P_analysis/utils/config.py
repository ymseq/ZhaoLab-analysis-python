from __future__ import annotations

import fnmatch
import itertools
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple

import pandas as pd


# =============================================================================
# Canonical definitions
# =============================================================================

TRACK_TYPES = {
    0: ["CAB", "CBA", "ACB", "BCA", "ABC", "BAC"],
    1: [
        "couple_ACB",
        "couple_BCA",
        "CAB",
        "CBA",
        "ACB",
        "BCA",
        "ABC",
        "BAC",
    ],
    2: ["couple_ACB", "couple_BCA"],
}


BEHAVIOR_TYPES = {
    0: ["Correct", "FalseAlarm", "Miss"],
    1: ["Correct", "FalseAlarm", "Miss", "NoReward"],
}


TASK_TYPES = (
    "first_pattern",
    "first_position",
    "pattern",
    "position",
    "couple",
)


REWARD_MODES = (
    "full",
    "75",
)


VALID_FILE_KINDS = {
    ".mat": "mat",
    ".pkl": "pickle",
    ".pickle": "pickle",
}


# =============================================================================
# Basic normalization
# =============================================================================

def normalize_name(name: str | Path) -> str:
    """
    Normalize filename / folder name.

    Examples
    --------
    first-pattern -> first_pattern
    cut-off-0     -> cut_off_0
    """
    return str(name).replace("-", "_")


def normalize_task_type(task_type: str) -> str:
    """
    Normalize and validate task_type.
    """
    task_type = normalize_name(task_type)

    if task_type not in TASK_TYPES:
        raise ValueError(
            f"Invalid task_type: {task_type!r}. "
            f"Valid values are: {TASK_TYPES}."
        )

    return task_type


def normalize_reward_mode(reward_mode: str) -> str:
    """
    Normalize and validate reward_mode.
    """
    reward_mode = str(reward_mode).lower()

    if reward_mode not in REWARD_MODES:
        raise ValueError(
            f"Invalid reward_mode: {reward_mode!r}. "
            f"Valid values are: {REWARD_MODES}."
        )

    return reward_mode


# =============================================================================
# ID <-> type conversion
# =============================================================================

def track_type_id_to_type(track_type_id: int) -> list[str]:
    """
    Convert track_type_id to track type list.
    """
    track_type_id = int(track_type_id)

    if track_type_id not in TRACK_TYPES:
        raise ValueError(
            f"Only support track_type_id in {sorted(TRACK_TYPES.keys())}, "
            f"got {track_type_id!r}."
        )

    return list(TRACK_TYPES[track_type_id])


def behavior_type_id_to_type(behavior_type_id: int) -> list[str]:
    """
    Convert behavior_type_id to behavior type list.
    """
    behavior_type_id = int(behavior_type_id)

    if behavior_type_id not in BEHAVIOR_TYPES:
        raise ValueError(
            f"Only support behavior_type_id in "
            f"{sorted(BEHAVIOR_TYPES.keys())}, "
            f"got {behavior_type_id!r}."
        )

    return list(BEHAVIOR_TYPES[behavior_type_id])


def task_type_to_track_type_id(task_type: str) -> int:
    """
    Infer track_type_id from task_type.
    """
    task_type = normalize_task_type(task_type)

    if task_type in {"pattern", "position"}:
        return 0

    if task_type in {"first_pattern", "first_position"}:
        return 1

    if task_type == "couple":
        return 2

    raise ValueError(f"Unknown task_type: {task_type!r}")


def reward_mode_to_behavior_type_id(reward_mode: str) -> int:
    """
    Infer behavior_type_id from reward_mode.
    """
    reward_mode = normalize_reward_mode(reward_mode)

    if reward_mode == "full":
        return 0

    if reward_mode == "75":
        return 1

    raise ValueError(f"Unknown reward_mode: {reward_mode!r}")


# Short aliases.
get_track_type = track_type_id_to_type
get_behavior_type = behavior_type_id_to_type
infer_track_type_id = task_type_to_track_type_id
infer_behavior_type_id = reward_mode_to_behavior_type_id


# =============================================================================
# Data-contained type access
# =============================================================================

def get_file_info(data: dict) -> dict:
    """
    Return data["file_info"] with validation.
    """
    if "file_info" not in data:
        raise KeyError(
            "data does not contain 'file_info'. "
            "Please load data using load_raw_data() or load_screen_data()."
        )

    return data["file_info"]


def get_data_track_type_id(data: dict) -> int:
    """
    Return track_type_id from data["file_info"].
    """
    file_info = get_file_info(data)

    if "track_type_id" not in file_info:
        raise KeyError("data['file_info'] does not contain 'track_type_id'.")

    return int(file_info["track_type_id"])


def get_data_behavior_type_id(data: dict) -> int:
    """
    Return behavior_type_id from data["file_info"].
    """
    file_info = get_file_info(data)

    if "behavior_type_id" not in file_info:
        raise KeyError("data['file_info'] does not contain 'behavior_type_id'.")

    return int(file_info["behavior_type_id"])


def get_data_track_type(data: dict) -> list[str]:
    """
    Return track type list from data["file_info"]["track_type_id"].
    """
    return track_type_id_to_type(get_data_track_type_id(data))


def get_data_behavior_type(data: dict) -> list[str]:
    """
    Return behavior type list from data["file_info"]["behavior_type_id"].
    """
    return behavior_type_id_to_type(get_data_behavior_type_id(data))


# =============================================================================
# Index helpers
# =============================================================================

def resolve_type_patterns(
    patterns: Iterable[str],
    universe: List[str],
) -> List[str]:
    """
    Resolve names / wildcard patterns against a universe list.

    Examples
    --------
    ["*"] against ["CAB", "CBA"]
        -> ["CAB", "CBA"]

    ["couple_*"] against ["couple_ACB", "CAB"]
        -> ["couple_ACB"]
    """
    resolved: List[str] = []

    for pattern in patterns:
        if pattern in universe:
            if pattern not in resolved:
                resolved.append(pattern)
            continue

        matches = [
            item for item in universe
            if fnmatch.fnmatch(item, pattern)
        ]

        if not matches:
            raise ValueError(
                f"No match for {pattern!r} in {universe}"
            )

        for match in matches:
            if match not in resolved:
                resolved.append(match)

    return resolved


def get_itr_index(
    data: dict,
    ana_tt: Iterable[str],
    ana_bt: Iterable[str],
) -> Iterator[Tuple[int, int]]:
    """
    Convert track / behavior names into index pairs.

    This reads type definitions from data["file_info"].
    """
    track_type = get_data_track_type(data)
    behavior_type = get_data_behavior_type(data)

    tt_names = resolve_type_patterns(ana_tt, track_type)
    bt_names = resolve_type_patterns(ana_bt, behavior_type)

    tt_idx = {
        name: i
        for i, name in enumerate(track_type)
    }

    bt_idx = {
        name: i
        for i, name in enumerate(behavior_type)
    }

    return itertools.product(
        (tt_idx[name] for name in tt_names),
        (bt_idx[name] for name in bt_names),
    )


def get_one_index(
    data: dict,
    ana_tt: str,
    ana_bt: str,
) -> Tuple[int, int]:
    """
    Convert one track / behavior name into one index pair.
    """
    track_type = get_data_track_type(data)
    behavior_type = get_data_behavior_type(data)

    tt_names = resolve_type_patterns([ana_tt], track_type)
    bt_names = resolve_type_patterns([ana_bt], behavior_type)

    if len(tt_names) != 1:
        raise ValueError(
            f"Expected exactly one match for track type "
            f"{ana_tt!r}, got {tt_names}."
        )

    if len(bt_names) != 1:
        raise ValueError(
            f"Expected exactly one match for behavior type "
            f"{ana_bt!r}, got {bt_names}."
        )

    tt_idx = {
        name: i
        for i, name in enumerate(track_type)
    }

    bt_idx = {
        name: i
        for i, name in enumerate(behavior_type)
    }

    return tt_idx[tt_names[0]], bt_idx[bt_names[0]]


def index_to_type(
    data: dict,
    tt_idx: int,
    bt_idx: int,
) -> Tuple[str, str]:
    """
    Convert index pair back to track / behavior names.
    """
    track_type = get_data_track_type(data)
    behavior_type = get_data_behavior_type(data)

    if not (0 <= tt_idx < len(track_type)):
        raise ValueError(
            f"Track type index {tt_idx} out of range."
        )

    if not (0 <= bt_idx < len(behavior_type)):
        raise ValueError(
            f"Behavior type index {bt_idx} out of range."
        )

    return track_type[tt_idx], behavior_type[bt_idx]


# =============================================================================
# File parsing
# =============================================================================

def parse_task_type_from_name(name: str | Path) -> str | None:
    """
    Parse task_type from filename stem.

    Supported examples
    ------------------
    neuro_type_xxx_pattern.mat
    neuro_type_xxx_75_pattern.mat
    neuro_type_xxx_position.mat
    neuro_type_xxx_75_position.mat
    neuro_type_xxx_first-pattern.mat
    neuro_type_xxx_first_position.mat
    neuro_type_xxx_couple.mat
    neuro_type_xxx_75_couple.mat
    """
    name = Path(name).stem if isinstance(name, Path) else str(name)
    name = normalize_name(name)

    for task_type in TASK_TYPES:
        if task_type in name:
            return task_type

    return None


def parse_reward_mode_from_name(name: str | Path) -> str:
    """
    Parse reward schedule from filename stem.

    Examples
    --------
    neuro_type_xxx_pattern       -> full
    neuro_type_xxx_75_pattern    -> 75
    neuro_type_xxx_reward75_pos  -> 75
    """
    name = Path(name).stem if isinstance(name, Path) else str(name)
    name = normalize_name(name).lower()

    tokens = name.split("_")

    if "75" in tokens:
        return "75"

    if "reward75" in name or "75reward" in name:
        return "75"

    return "full"


def parse_mouse_id_from_screen_name(
    name: str,
    task_type: str,
) -> str | None:
    """
    Parse mouse/session id from screen filename.

    Examples
    --------
    HP01_2025_01_01_pattern_screen
        -> HP01_2025_01_01

    HP01_2025_01_01_75_pattern_screen
        -> HP01_2025_01_01
    """
    name = normalize_name(name)
    task_type = normalize_task_type(task_type)

    idx = name.find(task_type)

    if idx <= 0:
        return None

    mouse_id = name[:idx].rstrip("_")

    if mouse_id.endswith("_75"):
        mouse_id = mouse_id[: -len("_75")].rstrip("_")

    return mouse_id if len(mouse_id) > 0 else None


def parse_file(file_path) -> dict | None:
    """
    Parse one original .mat file or one screening .pkl file.

    Returned file_info
    ------------------
    mouse_id
    file_path
    file_name
    suffix
    file_kind
    source_stem
    task_type
    reward_mode
    track_type_id
    behavior_type_id
    """
    file_path = Path(file_path)

    suffix = file_path.suffix.lower()

    if suffix not in VALID_FILE_KINDS:
        return None

    name = file_path.stem
    name_norm = normalize_name(name)

    is_raw_mat = (
        suffix == ".mat"
        and name_norm.startswith("neuro_type")
    )

    is_screen_pkl = (
        suffix in {".pkl", ".pickle"}
        and name_norm.endswith("_screen")
    )

    if not (is_raw_mat or is_screen_pkl):
        return None

    task_type = parse_task_type_from_name(name_norm)

    if task_type is None:
        return None

    reward_mode = parse_reward_mode_from_name(name_norm)
    track_type_id = task_type_to_track_type_id(task_type)
    behavior_type_id = reward_mode_to_behavior_type_id(reward_mode)

    if is_screen_pkl:
        source_stem = name_norm[: -len("_screen")]

        mouse_id = parse_mouse_id_from_screen_name(
            source_stem,
            task_type,
        )

        if mouse_id is None:
            mouse_id = file_path.parent.name

    else:
        source_stem = name_norm
        mouse_id = file_path.parent.name

    return {
        "mouse_id": mouse_id,
        "file_path": str(file_path),
        "file_name": file_path.name,
        "suffix": suffix,
        "file_kind": VALID_FILE_KINDS[suffix],
        "source_stem": source_stem,
        "task_type": task_type,
        "reward_mode": reward_mode,
        "track_type_id": track_type_id,
        "behavior_type_id": behavior_type_id,
    }


# =============================================================================
# File selection
# =============================================================================

def _to_tuple_or_none(x):
    """
    Normalize filter input.
    """
    if x is None:
        return None

    if isinstance(x, str):
        return (x,)

    return tuple(x)


def _normalize_task_type_filter(task_type):
    task_type = _to_tuple_or_none(task_type)

    if task_type is None:
        return None

    out = []

    for x in task_type:
        out.append(normalize_task_type(x))

    return tuple(dict.fromkeys(out))


def _normalize_reward_mode_filter(reward_mode):
    reward_mode = _to_tuple_or_none(reward_mode)

    if reward_mode is None:
        return None

    out = []

    for x in reward_mode:
        out.append(normalize_reward_mode(x))

    return tuple(dict.fromkeys(out))


def _normalize_file_kind_filter(file_kind):
    file_kind = _to_tuple_or_none(file_kind)

    if file_kind is None:
        return None

    out = []

    for x in file_kind:
        if x not in {"mat", "pickle"}:
            raise ValueError(
                f"Invalid file_kind: {x!r}. "
                "Valid values are: 'mat', 'pickle'."
            )

        out.append(x)

    return tuple(dict.fromkeys(out))


def metadata_dataframe(rows: list[dict]) -> pd.DataFrame:
    """
    Convert metadata rows into a DataFrame with stable column order.
    """
    columns = [
        "mouse_id",
        "file_path",
        "file_name",
        "suffix",
        "file_kind",
        "source_stem",
        "task_type",
        "reward_mode",
        "track_type_id",
        "behavior_type_id",
    ]

    if len(rows) == 0:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)

    existing_columns = [
        col for col in columns
        if col in df.columns
    ]

    other_columns = [
        col for col in df.columns
        if col not in existing_columns
    ]

    return df[existing_columns + other_columns]


def select_files(
    data_root,
    *,
    task_type=None,
    reward_mode=None,
    file_kind="mat",
    recursive: bool = False,
) -> pd.DataFrame:
    """
    Select original .mat files or screening .pkl files.

    Parameters
    ----------
    data_root : str or Path
        Folder to search.

    task_type : None, str, or iterable
        pattern / position / couple / first_pattern / first_position

    reward_mode : None, str, or iterable
        full / 75

    file_kind : None, str, or iterable
        mat / pickle

    recursive : bool
        If False:
            Search files directly under data_root and one-level child folders.

        If True:
            Recursively search all files under data_root.
    """
    data_root = Path(data_root)

    if not data_root.exists():
        raise FileNotFoundError(
            f"data_root does not exist: {data_root}"
        )

    task_type_set = _normalize_task_type_filter(task_type)
    reward_mode_set = _normalize_reward_mode_filter(reward_mode)
    file_kind_set = _normalize_file_kind_filter(file_kind)

    if recursive:
        candidate_files = [
            p for p in sorted(data_root.rglob("*"))
            if p.is_file()
        ]

    else:
        candidate_files = []

        candidate_files.extend(
            p for p in sorted(data_root.iterdir())
            if p.is_file()
        )

        for child_dir in sorted(data_root.iterdir()):
            if not child_dir.is_dir():
                continue

            candidate_files.extend(
                p for p in sorted(child_dir.iterdir())
                if p.is_file()
            )

    rows = []

    for file_path in candidate_files:
        info = parse_file(file_path)

        if info is None:
            continue

        if task_type_set is not None and info["task_type"] not in task_type_set:
            continue

        if reward_mode_set is not None and info["reward_mode"] not in reward_mode_set:
            continue

        if file_kind_set is not None and info["file_kind"] not in file_kind_set:
            continue

        rows.append(info)

    return metadata_dataframe(rows)


def select_screen_files(
    screen_root,
    *,
    task_type=None,
    reward_mode=None,
    recursive: bool = True,
) -> pd.DataFrame:
    """
    Convenience function for selecting screening pickle files.
    """
    return select_files(
        screen_root,
        task_type=task_type,
        reward_mode=reward_mode,
        file_kind="pickle",
        recursive=recursive,
    )

