from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio

from .config import (
    parse_file,
    track_type_id_to_type,
    behavior_type_id_to_type,
)


# =============================================================================
# MATLAB index field configuration
# =============================================================================

INDEX_FIELD_BY_BEHAVIOR = {
    # Temporary patch:
    # One legacy file mistakenly saved index_correct as correct.
    # Remove "correct" after the raw file is fixed.
    "Correct": ("index_correct", "correct"),

    "FalseAlarm": ("index_false",),
    "Miss": ("index_miss",),
    "NoReward": ("index_noreward",),
}


# =============================================================================
# MATLAB struct / array helpers
# =============================================================================

def mat_struct_to_dict(mat_struct) -> dict:
    """
    Convert one scipy-loaded MATLAB struct object to a Python dict.
    """
    if not hasattr(mat_struct, "_fieldnames"):
        raise TypeError(
            f"Expected MATLAB struct with _fieldnames, got {type(mat_struct)}."
        )

    return {
        name: getattr(mat_struct, name)
        for name in mat_struct._fieldnames
    }


def unwrap_single_object(value: Any) -> Any:
    """
    Unwrap scipy-loaded single-element object arrays.

    This is intentionally conservative to avoid infinite recursion on MATLAB
    struct objects.
    """
    while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        item = value.ravel()[0]

        if item is value:
            break

        value = item

        if hasattr(value, "_fieldnames"):
            break

    return value


def is_nan_like(value: Any) -> bool:
    """
    Return True only for numeric NaN-like values.

    Examples
    --------
    np.nan:
        True

    np.array([np.nan]):
        True

    MATLAB struct:
        False

    Empty array:
        False
    """
    if value is None:
        return False

    value = unwrap_single_object(value)

    if hasattr(value, "_fieldnames") or isinstance(value, (str, bytes)):
        return False

    try:
        arr = np.asarray(value)
    except Exception:
        return False

    if arr.size == 0:
        return False

    if np.issubdtype(arr.dtype, np.number):
        return bool(np.isnan(arr.astype(float)).all())

    if arr.dtype != object:
        return False

    results = []

    for item in arr.ravel():
        item = unwrap_single_object(item)

        if hasattr(item, "_fieldnames") or isinstance(item, (str, bytes)):
            return False

        try:
            item_arr = np.asarray(item, dtype=float)
        except Exception:
            return False

        if item_arr.size == 0:
            return False

        results.append(bool(np.isnan(item_arr).all()))

    return len(results) > 0 and all(results)


def empty_index() -> np.ndarray:
    """
    Standard empty trial index:
        shape = (0,)
    """
    return np.empty(0, dtype=int)


# =============================================================================
# Firing helpers
# =============================================================================

def get_behavior_array(
    temp_dict: dict,
    behavior_name: str,
) -> np.ndarray:
    """
    Strictly read one behavior array from one neuron's MATLAB struct.

    No alias is used here. The MATLAB field name must exactly match
    behavior_name.

    Therefore, if behavior_type contains "NoReward", the MATLAB struct must
    contain field "NoReward"; otherwise this function raises KeyError.
    """
    if behavior_name not in temp_dict:
        raise KeyError(
            f"Cannot find required behavior field {behavior_name!r}. "
            f"Available fields: {list(temp_dict.keys())}"
        )

    return temp_dict[behavior_name]


def ensure_3d(x: np.ndarray, len_position: int) -> np.ndarray:
    """
    Ensure firing matrix has shape:
        (n_neuron, n_trial, n_position)

    Accepted input examples
    -----------------------
    (n_neuron, len_position):
        one trial per neuron.

    (n_neuron, n_trial, len_position):
        already standard.

    (n_neuron, 0):
        zero trials.

    Fake-zero rule
    --------------
    If the firing matrix has exactly one trial and the whole matrix is all zero,
    treat it as zero trial:
        (n_neuron, 0, n_position)
    """
    x = np.asarray(x)

    if x.ndim == 2:
        if x.shape[1] == 0:
            x = x[..., None]

            return np.broadcast_to(
                x,
                (*x.shape[:2], len_position),
            ).copy()

        if x.shape[1] == len_position:
            x = x[:, None, :]

        else:
            raise ValueError(
                f"Expected 2D array with second dimension 0 or "
                f"{len_position}, got shape {x.shape}."
            )

    elif x.ndim != 3:
        raise ValueError(
            f"Expected 2D or 3D array, got shape {x.shape}."
        )

    if x.shape[1] == 1 and np.all(x == 0):
        return np.empty((x.shape[0], 0, x.shape[2]), dtype=x.dtype)

    return x


def build_firing_matrix(
    spk: np.ndarray,
    *,
    track_idx: int,
    track_name: str,
    behavior_name: str,
    len_position: int,
) -> np.ndarray:
    """
    Build firing matrix for one track type and one behavior type.

    Returns
    -------
    firing_matrix:
        np.ndarray with shape:
            (n_neuron, n_trial, n_position)
    """
    num_neuron = spk.shape[0]
    behavior_matrix = []

    for neuron_idx in range(num_neuron):
        temp_f = spk[neuron_idx, track_idx]
        temp_dict = mat_struct_to_dict(temp_f)

        arr = get_behavior_array(
            temp_dict,
            behavior_name,
        )

        behavior_matrix.append(arr)

    try:
        behavior_matrix = np.stack(
            behavior_matrix,
            axis=0,
        )

    except ValueError as exc:
        shapes = [
            np.asarray(x).shape
            for x in behavior_matrix
        ]

        raise ValueError(
            f"Cannot stack behavior={behavior_name!r}, "
            f"track={track_name!r}. "
            f"Neuron-wise array shapes are inconsistent: "
            f"{shapes[:10]}..."
        ) from exc

    return ensure_3d(
        behavior_matrix,
        len_position,
    )


# =============================================================================
# Index helpers
# =============================================================================

def get_index_field_name(
    mat_data: dict,
    behavior_name: str,
) -> str:
    """
    Resolve MATLAB index field name for one behavior.

    Strict rule
    -----------
    FalseAlarm:
        must use index_false

    Miss:
        must use index_miss

    NoReward:
        must use index_noreward

    Temporary patch
    ---------------
    Correct:
        first try index_correct;
        if missing, allow correct.

    This patch is only for one legacy file whose index_correct was
    accidentally saved as correct.
    """
    if behavior_name not in INDEX_FIELD_BY_BEHAVIOR:
        raise KeyError(
            f"Unknown behavior_name={behavior_name!r}. "
            f"Known behaviors: {list(INDEX_FIELD_BY_BEHAVIOR.keys())}"
        )

    candidate_keys = INDEX_FIELD_BY_BEHAVIOR[behavior_name]

    for key in candidate_keys:
        if key in mat_data:
            return key

    raise KeyError(
        f"Missing required index field for behavior={behavior_name!r}. "
        f"Expected one of {candidate_keys}. "
        f"Available top-level fields: {list(mat_data.keys())}"
    )


def split_index_by_track(
    raw_index: Any,
    *,
    num_track: int,
    index_key: str,
) -> list[Any]:
    """
    Convert one MATLAB index field to a Python list of length num_track.

    Rules
    -----
    NaN / empty:
        all tracks are empty.

    scalar:
        first track uses the scalar, remaining tracks are empty.

    length < num_track:
        pad missing tail elements with None.

    length == num_track:
        use as-is.

    length > num_track:
        raise ValueError.
    """
    raw_index = unwrap_single_object(raw_index)

    if is_nan_like(raw_index):
        return [None] * num_track

    arr = np.asarray(raw_index, dtype=object)
    arr = np.squeeze(arr)

    if arr.size == 0:
        return [None] * num_track

    if arr.ndim == 0:
        item = arr.item()

        if is_nan_like(item):
            return [None] * num_track

        return [item] + [None] * (num_track - 1)

    flat = list(arr.ravel())

    if len(flat) == 1 and is_nan_like(flat[0]):
        return [None] * num_track

    if len(flat) < num_track:
        return flat + [None] * (num_track - len(flat))

    if len(flat) > num_track:
        raise ValueError(
            f"{index_key} should contain at most {num_track} track elements, "
            f"but got {len(flat)}. Squeezed shape: {arr.shape}."
        )

    return flat


def normalize_index_value(value: Any) -> np.ndarray:
    """
    Convert one MATLAB index element to a 1D integer array.

    Rules
    -----
    None:
        empty index.

    NaN:
        empty index.

    []:
        empty index.

    scalar:
        shape = (1,)

    vector:
        flattened 1D index array.

    MATLAB struct:
        If it has field "index", use that field.
        If it has exactly one field, use that field.
    """
    if value is None:
        return empty_index()

    value = unwrap_single_object(value)

    if is_nan_like(value):
        return empty_index()

    if hasattr(value, "_fieldnames"):
        fields = list(value._fieldnames)

        if len(fields) == 0:
            return empty_index()

        if "index" in fields:
            value = getattr(value, "index")
        elif len(fields) == 1:
            value = getattr(value, fields[0])
        else:
            raise ValueError(
                f"Cannot infer index field from MATLAB struct fields: {fields}"
            )

    value = unwrap_single_object(value)

    if is_nan_like(value):
        return empty_index()

    arr = np.asarray(value)

    if arr.size == 0:
        return empty_index()

    if arr.dtype == object:
        parts = []

        for item in arr.ravel():
            item = normalize_index_value(item)

            if item.size > 0:
                parts.append(item)

        if len(parts) == 0:
            return empty_index()

        return np.concatenate(parts)

    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(
            f"Index value must be numeric, got dtype={arr.dtype}, value={arr}"
        )

    arr = np.asarray(arr, dtype=float).ravel()
    arr = arr[~np.isnan(arr)]

    if arr.size == 0:
        return empty_index()

    return arr.astype(int)


def build_index_matrix(
    mat_data: dict,
    *,
    behavior_type: list[str],
    num_track: int,
) -> np.ndarray:
    """
    Build trial index object array.

    Returns
    -------
    trial_index:
        object array with shape:
            (n_track_type, n_behavior_type)

    trial_index[tt_idx, bt_idx]:
        np.ndarray with shape:
            (n_trial,)
    """
    trial_index = np.empty(
        (num_track, len(behavior_type)),
        dtype=object,
    )

    for bt_idx, behavior_name in enumerate(behavior_type):
        index_key = get_index_field_name(
            mat_data,
            behavior_name,
        )

        index_by_track = split_index_by_track(
            mat_data[index_key],
            num_track=num_track,
            index_key=index_key,
        )

        for tt_idx, raw_index in enumerate(index_by_track):
            trial_index[tt_idx, bt_idx] = normalize_index_value(raw_index)

    return trial_index


# =============================================================================
# Validation helpers
# =============================================================================

def validate_file_info(file_info: dict) -> None:
    """
    Validate the minimal metadata needed by loaded data.
    """
    required = [
        "mouse_id",
        "file_path",
        "file_name",
        "task_type",
        "reward_mode",
        "track_type_id",
        "behavior_type_id",
    ]

    missing = [
        key for key in required
        if key not in file_info
    ]

    if len(missing) > 0:
        raise KeyError(
            f"file_info is missing required keys: {missing}"
        )


def check_firing_index_consistency(
    firing: np.ndarray,
    trial_index: np.ndarray,
    track_type: list[str],
    behavior_type: list[str],
    *,
    file_path: str | Path | None = None,
) -> None:
    """
    Check whether each firing matrix and its corresponding trial index have
    the same trial count.

    Specifically:
        firing[tt_idx, bt_idx].shape[1]
        ==
        trial_index[tt_idx, bt_idx].size
    """
    if firing.shape != trial_index.shape:
        raise ValueError(
            f"firing and index shape mismatch. "
            f"firing.shape={firing.shape}, index.shape={trial_index.shape}."
        )

    expected_shape = (
        len(track_type),
        len(behavior_type),
    )

    if firing.shape != expected_shape:
        raise ValueError(
            f"firing/index shape mismatch. "
            f"Expected {expected_shape}, got {firing.shape}. "
            f"track_type={track_type}, behavior_type={behavior_type}."
        )

    mismatches = []

    for tt_idx, track_name in enumerate(track_type):
        for bt_idx, behavior_name in enumerate(behavior_type):
            fr = np.asarray(firing[tt_idx, bt_idx])
            idx = np.asarray(trial_index[tt_idx, bt_idx])

            if fr.ndim != 3:
                raise ValueError(
                    f"firing[{tt_idx}, {bt_idx}] "
                    f"track={track_name!r}, behavior={behavior_name!r} "
                    f"should be 3D (n_neuron, n_trial, n_position), "
                    f"got shape {fr.shape}."
                )

            if idx.ndim > 1:
                raise ValueError(
                    f"index[{tt_idx}, {bt_idx}] "
                    f"track={track_name!r}, behavior={behavior_name!r} "
                    f"should be 1D, got shape {idx.shape}."
                )

            n_trial_firing = fr.shape[1]
            n_trial_index = idx.size

            if n_trial_firing != n_trial_index:
                mismatches.append(
                    {
                        "track_idx": tt_idx,
                        "track": track_name,
                        "behavior_idx": bt_idx,
                        "behavior": behavior_name,
                        "n_trial_firing": n_trial_firing,
                        "n_trial_index": n_trial_index,
                        "firing_shape": fr.shape,
                        "index_shape": idx.shape,
                        "index_values": idx,
                    }
                )

    if len(mismatches) > 0:
        lines = ["Firing-index trial number mismatch found."]

        if file_path is not None:
            lines.append(f"File: {file_path}")

        for item in mismatches:
            lines.append(
                f"  track={item['track']!r}, "
                f"behavior={item['behavior']!r}, "
                f"firing_trial={item['n_trial_firing']}, "
                f"index_trial={item['n_trial_index']}, "
                f"firing_shape={item['firing_shape']}, "
                f"index_shape={item['index_shape']}, "
                f"index_values={item['index_values']}"
            )

        raise ValueError("\n".join(lines))


def validate_standard_data(data: dict) -> None:
    """
    Validate simplified standard data structure.

    Required structure
    ------------------
    data = {
        "file_info": {...},
        "zones": ...,
        "cell_ids": ...,
        "firing": object array,
        "index": object array,
    }
    """
    required = [
        "file_info",
        "zones",
        "cell_ids",
        "firing",
        "index",
    ]

    missing = [
        key for key in required
        if key not in data
    ]

    if len(missing) > 0:
        raise KeyError(
            f"Loaded data is missing required keys: {missing}"
        )

    file_info = data["file_info"]
    validate_file_info(file_info)

    track_type = track_type_id_to_type(
        int(file_info["track_type_id"])
    )

    behavior_type = behavior_type_id_to_type(
        int(file_info["behavior_type_id"])
    )

    check_firing_index_consistency(
        firing=data["firing"],
        trial_index=data["index"],
        track_type=track_type,
        behavior_type=behavior_type,
        file_path=file_info.get("file_path", None),
    )


# =============================================================================
# Raw MAT loading
# =============================================================================

def load_raw_mat_data(
    file_path,
    *,
    file_info: dict,
    len_position: int = 160,
) -> dict:
    """
    Load original MATLAB neuro_type*.mat file and return simplified data.

    Output structure
    ----------------
    data = {
        "file_info": file_info,
        "zones": reward_bin,
        "cell_ids": response_cell,
        "firing": object array,
        "index": object array,
    }

    data["firing"][tt_idx, bt_idx]:
        np.ndarray with shape:
            (n_neuron, n_trial, n_position)

    data["index"][tt_idx, bt_idx]:
        np.ndarray with shape:
            (n_trial,)
    """
    file_path = Path(file_path)

    mat = sio.loadmat(
        file_path,
        squeeze_me=True,
        struct_as_record=False,
    )

    if "neuro_type_save" not in mat:
        raise KeyError(
            f"{file_path} does not contain MATLAB variable "
            f"'neuro_type_save'."
        )

    temp_s = mat["neuro_type_save"]
    mat_data = mat_struct_to_dict(temp_s)

    for field in ("reward_bin", "response_cell", "spk"):
        if field not in mat_data:
            raise KeyError(
                f"MAT file {file_path} is missing required field {field!r}."
            )

    spk = np.asarray(mat_data["spk"], dtype=object)

    if spk.ndim == 1:
        spk = spk[None, :]

    if spk.ndim != 2:
        raise ValueError(
            f"Expected spk to be 2D after loading, got shape {spk.shape}."
        )

    num_neuron, num_track_in_mat = spk.shape

    track_type = track_type_id_to_type(
        int(file_info["track_type_id"])
    )

    behavior_type = behavior_type_id_to_type(
        int(file_info["behavior_type_id"])
    )

    if num_track_in_mat < len(track_type):
        raise ValueError(
            f"MAT spk has fewer track columns than expected. "
            f"spk.shape={spk.shape}, expected track_type={track_type}."
        )

    firing = np.empty(
        (len(track_type), len(behavior_type)),
        dtype=object,
    )

    for tt_idx, track_name in enumerate(track_type):
        for bt_idx, behavior_name in enumerate(behavior_type):
            firing[tt_idx, bt_idx] = build_firing_matrix(
                spk,
                track_idx=tt_idx,
                track_name=track_name,
                behavior_name=behavior_name,
                len_position=len_position,
            )

    trial_index = build_index_matrix(
        mat_data,
        behavior_type=behavior_type,
        num_track=len(track_type),
    )

    data = {
        "file_info": copy.deepcopy(file_info),
        "zones": mat_data["reward_bin"],
        "cell_ids": np.atleast_1d(mat_data["response_cell"]),
        "firing": firing,
        "index": trial_index,
    }

    validate_standard_data(data)

    return data


def load_raw_pkl_data(
    file_path,
    *,
    file_info: dict,
    len_position: int = 160,
) -> dict:
    """
    Placeholder for future raw pkl support.

    This is intentionally different from load_screen_data().
    """
    raise NotImplementedError(
        "Raw pkl loading is not implemented yet. "
        "Please use .mat raw files for now, or implement "
        "load_raw_pkl_data() later."
    )


def load_raw_data(
    file_path,
    *,
    len_position: int = 160,
) -> dict:
    """
    Load raw data and return simplified standard data.

    The file name must be parseable by config.parse_file().
    No external task_type / reward_mode / id override is supported.
    Rename the file if parsing fails.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    file_info = parse_file(file_path)

    if file_info is None:
        raise ValueError(
            f"Cannot parse raw file name: {file_path.name}. "
            "Please rename the file according to the standard naming rule."
        )

    validate_file_info(file_info)

    if suffix == ".mat":
        return load_raw_mat_data(
            file_path,
            file_info=file_info,
            len_position=len_position,
        )

    if suffix in {".pkl", ".pickle"}:
        return load_raw_pkl_data(
            file_path,
            file_info=file_info,
            len_position=len_position,
        )

    raise ValueError(
        f"Unsupported raw file suffix: {suffix}. "
        "Only .mat, .pkl, and .pickle are supported."
    )


# =============================================================================
# Screen data loading
# =============================================================================

def load_screen_result(file_path):
    """
    Load full screen result dictionary.

    Expected structure
    ------------------
    result = {
        "screening_result": ...,
        "final_mask": ...,
        "attracted_data": ...,
        "masked_attracted_data": ...,
    }
    """
    file_path = Path(file_path)

    with open(file_path, "rb") as f:
        return pickle.load(f)


def load_screen_data(
    file_path,
    *,
    screen_mode: str = "masked",
) -> dict:
    """
    Load data from screen result pkl and return simplified standard data.

    This function does not reconstruct raw MATLAB fields.
    It only reads the saved attracted_data or masked_attracted_data.

    Parameters
    ----------
    screen_mode : {"masked", "screened", "raw", "original"}
        masked / screened:
            return result["masked_attracted_data"]

        raw / original:
            return result["attracted_data"]
    """
    file_path = Path(file_path)
    result = load_screen_result(file_path)

    if screen_mode in {"raw", "original"}:
        key = "attracted_data"

    elif screen_mode in {"masked", "screened"}:
        key = "masked_attracted_data"

    else:
        raise ValueError(
            "screen_mode must be one of: "
            "'masked', 'screened', 'raw', 'original'."
        )

    if key not in result:
        raise KeyError(
            f"This screen pickle does not contain {key!r}."
        )

    data = copy.deepcopy(result[key])

    validate_standard_data(data)

    return data