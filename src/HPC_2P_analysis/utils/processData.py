from __future__ import annotations
from typing import Iterable, List, Tuple, Iterator
from scipy.ndimage import gaussian_filter1d
import fnmatch
import itertools
import numpy as np
import scipy.io as sio

_track_type = None
_behavior_type = ["Correct", "FalseAlarm", "Miss"]


def set_track_type(which: int = 0):
    global _track_type
    match which:
        case 0:
            _track_type = ["CAB", "CBA", "ACB", "BCA", "ABC", "BAC"]
        case 1:
            _track_type = ["couple_ACB", "couple_BCA", "CAB", "CBA", "ACB", "BCA", "ABC", "BAC"]
        case _:
            raise ValueError("Only support which == 0 / 1")

def validate_track_type():
    if _track_type is None:
        raise ValueError("Please set track type first by using function set_track_type()")
    return _track_type

def _resolve(patterns: Iterable[str], universe: List[str]) -> List[str]:
    """
    Resolve names/patterns (supports '*' wildcards) against a universe list.
    Keeps universe order, deduplicates, and errors if no match for a token.
    """
    resolved: List[str] = []
    for p in patterns:
        # literal match first
        if p in universe:
            if p not in resolved:
                resolved.append(p)
            continue

        matches = [u for u in universe if fnmatch.fnmatch(u, p)]
        if not matches:
            raise ValueError(f"No match for '{p}' in {universe}")

        for m in matches:
            if m not in resolved:
                resolved.append(m)

    return resolved

def get_itr_index(ana_tt: Iterable[str], ana_bt: Iterable[str]) -> Iterator[Tuple[int, int]]:
    """
    ana_tt / ana_bt: iterable of strings (supports wildcards like '*', '?')
    Returns an iterator of (tt_index, bt_index) pairs.
    """
    track_type = validate_track_type()
    tt_names = _resolve(ana_tt, track_type)
    bt_names = _resolve(ana_bt, _behavior_type)

    tt_idx = {name: i for i, name in enumerate(track_type)}
    bt_idx = {name: i for i, name in enumerate(_behavior_type)}

    return itertools.product(
        (tt_idx[name] for name in tt_names),
        (bt_idx[name] for name in bt_names),
    )
    

def get_one_index(ana_tt: str, ana_bt: str) -> Tuple[int, int]:
    """
    ana_tt / ana_bt: single string (supports wildcards like '*', '?')
    Returns a single (tt_index, bt_index) pair.
    """
    track_type = validate_track_type()
    tt_names = _resolve([ana_tt], track_type)
    bt_names = _resolve([ana_bt], _behavior_type)

    if len(tt_names) != 1:
        raise ValueError(f"Expected exactly one match for track type '{ana_tt}', got {tt_names}")
    if len(bt_names) != 1:
        raise ValueError(f"Expected exactly one match for behavior type '{ana_bt}', got {bt_names}")

    tt_idx = {name: i for i, name in enumerate(track_type)}
    bt_idx = {name: i for i, name in enumerate(_behavior_type)}

    return tt_idx[tt_names[0]], bt_idx[bt_names[0]]


def index_to_type(tt_idx: int, bt_idx: int) -> Tuple[str, str]:
    """
    Convert (tt_index, bt_index) back to (track_type, behavior_type) strings.
    """
    track_type = validate_track_type()
    if not (0 <= tt_idx < len(track_type)):
        raise ValueError(f"Track type index {tt_idx} out of range")
    if not (0 <= bt_idx < len(_behavior_type)):
        raise ValueError(f"Behavior type index {bt_idx} out of range")

    return track_type[tt_idx], _behavior_type[bt_idx]


def _ensure_3d(x: np.ndarray, len_position: int) -> np.ndarray:

    if x.ndim == 2:
        if x.shape[1] == 0:
            x = x[..., None]
            return np.broadcast_to(x, (*x.shape[:2], len_position)).copy()
        elif x.shape[1] == len_position:
            return x[:,None,:]
        else:
            raise ValueError(f"Expected 2D array with second dimension 0 or {len_position}, got shape {x.shape}")
    
    if x.ndim == 3:
        return x

    raise ValueError(f"Expected 2D or 3D array, got shape {x.shape}")


def load_data(file_path):
    attracted_data = {}
    
    mat = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    temp_s = mat["neuro_type_save"]
    data = {name: getattr(temp_s, name) for name in temp_s._fieldnames}
    
    attracted_data["task_rule"] = data["type"]
    attracted_data["zones"] = data["reward_bin"]
    attracted_data["cell_ids"] = data["response_cell"]
    
    
    spk = data["spk"]
    num_neuron, _ = spk.shape
    
    track_type = validate_track_type()
    firing = np.empty((len(track_type), len(_behavior_type)), dtype=object)
    
    for j in range(len(track_type)):
        
        c_matrix = []
        f_matrix = []
        m_matrix = []
        
        for i in range(num_neuron):
            temp_f = spk[i, j]
            temp_dict = {name: getattr(temp_f, name) for name in temp_f._fieldnames}
            c_matrix.append(temp_dict["Correct"])
            f_matrix.append(temp_dict["FalseAlarm"])
            m_matrix.append(temp_dict["Miss"])
        
        len_position = 160
        c_matrix = np.stack(c_matrix, axis=0)
        f_matrix = np.stack(f_matrix, axis=0)
        m_matrix = np.stack(m_matrix, axis=0)
        c_matrix = _ensure_3d(c_matrix, len_position)
        f_matrix = _ensure_3d(f_matrix, len_position)
        m_matrix = _ensure_3d(m_matrix, len_position) # shape (num_neuron, trials, len_position)
        
        firing[j] = [c_matrix, f_matrix, m_matrix]
    
    attracted_data["firing"] = firing
    return attracted_data


def align_track_hpc(
    data: dict,
    *,
    gaussian_sigma: int = 0,
):
    if "firing" not in data:
        raise KeyError("data must contain 'firing'")

    smooth_firing = np.empty_like(data["firing"], dtype=object)
    aligned_firing = np.empty_like(data["firing"], dtype=object)
    firing_std = np.empty_like(data["firing"], dtype=object)

    for tt_idx, bt_idx in get_itr_index(["*"], ["*"]):
        fr = data["firing"][tt_idx, bt_idx]

        if fr is None:
            aligned_firing[tt_idx, bt_idx] = None
            firing_std[tt_idx, bt_idx] = None
            continue
        
        if fr.ndim != 3:
            raise ValueError(f"Expected fr with shape (n_neuron, n_trial, n_pos), got {fr.shape}")

        if fr.shape[1] == 0:
            aligned_firing[tt_idx, bt_idx] = None
            firing_std[tt_idx, bt_idx] = None
            continue

        if gaussian_sigma > 0:
            fr = gaussian_filter1d(
                fr, sigma=gaussian_sigma, axis=2,
                mode="nearest", truncate=3.0
            )

        smooth_firing[tt_idx, bt_idx] = fr
        aligned_firing[tt_idx, bt_idx] = np.mean(fr, axis=1)
        firing_std[tt_idx, bt_idx] = np.std(fr, axis=1, ddof=0) / np.sqrt(fr.shape[1])

    data["smooth_firing"] = smooth_firing
    data["aligned_firing"] = aligned_firing
    data["firing_std"] = firing_std

    return data

set_track_type(0)
