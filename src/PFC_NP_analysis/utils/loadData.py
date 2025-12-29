from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Union
import numpy as np

_PICKLE_EXTS = {".pkl", ".pickle"}

def _is_pickle_file(p: Path) -> bool:
    """Check if a path points to a valid pickle file (.pkl or .pickle)."""
    return p.is_file() and p.suffix.lower() in _PICKLE_EXTS


def load_pickle(path: Union[str, Path]) -> Any | Dict[str, Any]:
    """
    Load pickle data.

    - File path: return the deserialized object.
    - Directory path: recursively load all .pkl/.pickle files and return a dict
      mapping **filename without extension** to the loaded object.
      * If duplicate stems occur, append a counter (#1, #2...).
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")

    # Case 1: Single file
    if p.is_file():
        if not _is_pickle_file(p):
            raise ValueError(f"Unsupported file type: {p.name} (only .pkl/.pickle)")
        with open(p, "rb") as f:
            return pickle.load(f)

    # Case 2: Directory — recursively load pickle files
    if p.is_dir():
        result: Dict[str, Any] = {}
        for file in p.rglob("*"):
            if not _is_pickle_file(file):
                continue

            key = file.stem  # default key: file name (without extension)
            suffix = 1
            while key in result:
                key = f"{file.stem}#{suffix}"
                suffix += 1

            with open(file, "rb") as f:
                result[key] = pickle.load(f)
        return result

    # Neither a file nor a directory
    raise RuntimeError(f"Unsupported path type: {p}")



def extract_used_data(data: Dict[str, Any], is_pretrain: bool = False) -> Dict[str, Any]:
    """
    Strictly validate and preprocess the input dict.
    Required keys:
      - 'simple_firing': 2D object-like grid
      - 'zones': shape must be (5, 2)
      - 'pos_lick_type': same 2D shape as 'simple_firing'
      - 'pos_reward_type': same 2D shape as 'simple_firing'
      - 'type_index': same 2D shape as 'simple_firing'
    If any key is missing or shape does not match, an error is raised.
    """
    # ----- simple_firing -----
    if "simple_firing" not in data:
        raise KeyError("missing key: simple_firing")

    firing = np.array(data["simple_firing"], dtype=object)
    _dim_test(firing.shape, "simple_firing")
    extract_data: Dict[str, Any] = {}
    extract_data["shape"] = firing.shape[0:2]

    # apply element-wise preprocessing
    processed_firing = _apply_on_object_grid(
        firing,
        _to_array_and_drop_first_dim,
        is_object=False, # set false, because withn trials, the firing data get the same shape
    )
    extract_data["simple_firing"] = processed_firing
    assert processed_firing.shape == extract_data["shape"], f"simple_firing shape{processed_firing.shape} does not match shape{extract_data['shape']}"

    # ----- zones -----
    if "zones" not in data:
        raise KeyError("missing key: zones")

    zones = np.array(data["zones"])
    if zones.shape != (5, 2):
        raise ValueError(f"zones expected shape (5, 2); got {zones.shape}")
    extract_data["zones"] = zones

    # ----- pos_lick_type -----
    if "pos_lick_type" not in data:
        raise KeyError("missing key: pos_lick_type")

    lick_positions = np.array(data["pos_lick_type"], dtype=object)
    _dim_test(lick_positions.shape, "pos_lick_type")

    processed_lick_positions = _apply_on_object_grid(
        lick_positions,
        _to_array_and_drop_first_dim,
        is_object=True, # set true, because the lick data is not the same shape in different trials
    )
    extract_data["pos_lick_type"] = processed_lick_positions
    assert processed_lick_positions.shape == extract_data["shape"], f"pos_lick_type shape{processed_lick_positions.shape} does not match shape{extract_data['shape']}"

    # ----- pos_reward_type -----
    if "pos_reward_type" not in data:
        raise KeyError("missing key: pos_reward_type")

    reward_positions = np.array(data["pos_reward_type"], dtype=object)
    _dim_test(reward_positions.shape, "pos_reward_type")

    processed_reward_positions = _apply_on_object_grid(
        reward_positions,
        _to_array_and_drop_first_dim,
        is_object=True, # set true, because the reward data is not the same shape in different trials
    )
    extract_data["pos_reward_type"] = processed_reward_positions
    assert processed_reward_positions.shape == extract_data["shape"], f"pos_reward_type shape{processed_reward_positions.shape} does not match shape{extract_data['shape']}"
    
    # ----- type_index -----
    if "type_index" not in data:
        raise KeyError("missing key: type_index")
    
    type_index = np.array(data["type_index"], dtype=object)
    _dim_test(type_index.shape, "type_index")
    
    type_index = _change_type_index_data(type_index, is_pretrain)
    processed_type_index = _apply_on_object_grid(
        type_index,
        _to_array_and_drop_first_dim,
        is_object=False, # set true, because the type_index data is 1d
    )
    extract_data["type_index"] = processed_type_index
    assert type_index.shape == extract_data["shape"], f"type_index shape{type_index.shape} does not match shape{extract_data['shape']}"
    
    return extract_data


def _apply_on_object_grid(grid: np.ndarray, func, **kwargs) -> np.ndarray:
    """
    Apply `func` to every element in a at least 2D  array and return
    another 2D  array with the same shape.
    """
    out = np.empty(grid.shape[0:2], dtype=object)
    for idx in np.ndindex(grid.shape[0:2]):
        out[idx] = func(grid[idx], **kwargs)
    return out


def _to_array_and_drop_first_dim(x: Any, is_object: bool = False) -> np.ndarray | None:
    """
    Convert input to ndarray. If the array is empty, return empty array
    Otherwise:
      - squeeze the first dimension
    """
    if len(x) == 0:
        return None
    
    if is_object:
        arr = np.array(x, dtype=object)
    else:
        arr = np.array(x) # Make sure the data is array shape, otherwise throw warning
        
    if len(arr.shape) != 1:
        assert arr.shape[0] == 1, f"Expected 1D array, or multi dim array with first dim equal to 1, got {arr.shape}"
        arr = np.squeeze(arr, axis=0)
    
    if len(arr.shape) == 2:
        # For some lick and reward data, shape would be (*,0), (*,*), change that to (*,) shape
        assert arr.shape[0] != 0, "If the array is 2d and not empty, the first dim should not be 0"
        arr_new = np.empty((arr.shape[0],), dtype=object)
        for i in range(arr.shape[0]):
            arr_new[i] = arr[i].tolist()
        arr = arr_new
    
    assert len(arr.shape) == 1 or len(arr.shape) == 3, "The shape of data of task type should be 1 or 3"
    assert arr.size != 0, "The array should not be empty"
    return arr


def _dim_test(dim: int | tuple[int, ...], label):
    if isinstance(dim, tuple):
        if len(dim) == 2:
            return
        elif len(dim) == 3 and dim[2] == 1:
            return
    raise ValueError(f"{label} of task type dimensions wrong, got {dim}")


def _change_type_index_data(ti: np.ndarray, is_pretrain: bool = False):

    if ti.shape[0] == 3 and ti.shape[1] != 3:
        ti = np.transpose(ti, (1, 0, *range(2, ti.ndim)))

    n_rows, _ = ti.shape

    if is_pretrain:
        cols = np.r_[0:2, 14:n_rows]
        return ti[cols, :]
    else:
        return ti[:14, :]