from __future__ import annotations

from .config import Params
import numpy as np
from scipy.signal import convolve
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d
from typing import List, Dict, Tuple
from scipy.signal import resample_poly


def mergeAB(data: dict):
    
    len_tt = data["shape"][0]
    len_bt = data["shape"][1]
    
    assert len_tt % 2 == 0, " data should have even number of trial types"
    
    # merge data
    
    mergeShape = (len_tt // 2, len_bt)
    
    # merge simple_firing
    simple_firing = np.empty(mergeShape, dtype=object)
    # empty array or num_neuron * num_trials * num_positions 3D array
    # merge the adjacent trials
    for i in range(0, len_tt, 2):
        for j in range(len_bt):
            simple_firing[i // 2, j] = _concat_safe(data['simple_firing'][i, j], data['simple_firing'][i + 1, j], axis=1)
            
    # merge pos_lick_type
    pos_lick_type = np.empty(mergeShape, dtype=object)
    for i in range(0, len_tt, 2):
        for j in range(len_bt):
            pos_lick_type[i // 2, j] = _concat_safe(data['pos_lick_type'][i, j], data['pos_lick_type'][i + 1, j], axis=0)
            
    # merge pos_reward_type
    pos_reward_type= np.empty(mergeShape, dtype=object)
    for i in range(0, len_tt, 2):
        for j in range(len_bt):
            pos_reward_type[i // 2, j] = _concat_safe(data['pos_reward_type'][i, j], data['pos_reward_type'][i + 1, j], axis=0)
            
    # merge type_index
    type_index = np.empty(mergeShape, dtype=object)
    for i in range(0, len_tt, 2):
        for j in range(len_bt):
            type_index[i // 2, j] = _concat_safe(data['type_index'][i, j], data['type_index'][i + 1, j], axis=0)
            
    data['shape'] = mergeShape
    data['simple_firing'] = simple_firing
    data['pos_lick_type'] = pos_lick_type
    data['pos_reward_type'] = pos_reward_type
    data['type_index'] = type_index




def _concat_safe(a: np.ndarray | None, b: np.ndarray | None, axis: int = 0) -> np.ndarray | None:
    """
    Safely concatenate two arrays along given axis.
    Special cases:
      - if both are None -> return None array
      - if one is None   -> return a copy of the other
    Otherwise falls back to np.concatenate.
    """

    # both empty
    if a is None and b is None:
        return None

    # only a empty
    if a is None and b is not None:
        return b.copy()

    # only b empty
    if b is None and a is not None:
        return a.copy()

    if a is not None and b is not None:
        return np.concatenate((a, b), axis=axis)

def align_track(data: Dict, params: Params, is_clip: bool = True, is_gaussian: bool = True):
    
    # # Process the whole data
    # g1d = _gaussian_1d_kernel(params.gaussian_range, params.gaussian_sigma)
    # # 1 * k
    # g2d = np.reshape(g1d, (1, g1d.size))
    
    aligned_firing = np.empty_like(data["simple_firing"],dtype=object)
    
    for index in params.total_index_grid:
        
        # fr: [num_neurons, trials, num_positions]
        fr = data["simple_firing"][index]
        if fr is None:
            continue
        
        # average within trials
        # fr: [num_neurons, num_positions]
        fr = np.mean(fr, axis=1)

        # # convolve with Gaussian kernel
        # # fr: [num_neurons, num_positions]
        # fr = convolve(fr, g2d, mode='same')
        # fr = medfilt(fr, kernel_size=(1, 11))
        
        # clip, keep the track within the setted range
        # fr: [num_neurons, len_track]

        if is_gaussian:
            fr = gaussian_filter1d(fr, sigma=params.gaussian_sigma, axis=1, mode="nearest", truncate=3.0)

        if is_clip:
            fr = fr[:, params.track_range[0]:params.track_range[1]]
        
        k = params.len_pos_average
        if k != 1:
            fr = resample_poly(fr, up=1, down=k, axis=1)

        aligned_firing[index] = fr
    
    aligned_zones_id = np.empty_like(data["zones"],dtype=int)
    zones = data["zones"]
    for i in range(len(zones)):
        if is_clip:
            offset = params.track_range[0]
        else:
            offset = 0
        start = int((zones[i][0] / params.space_unit - offset) / params.len_pos_average) - 1
        end = int((zones[i][1] / params.space_unit - offset) / params.len_pos_average) + 1
        start = max(start, 0)
        end = min(end, int(params.len_track / params.len_pos_average))
        aligned_zones_id[i] = [start, end]
        
    data["aligned_firing"] = aligned_firing
    data["aligned_zones_id"] = aligned_zones_id
    
    
    
def z_score_on(data: Dict, params: Params, ana_tt: List[str], ana_bt: List[str]):
    
    blocks = []
    for index in params.ana_index_grid(ana_tt, ana_bt):
        if data["aligned_firing"][index] is not None:
            blocks.append(data["aligned_firing"][index])

    if len(blocks) == 0:
        return None, None

    # fr_sum: [num_neurons, (len_track * num_types)]
    fr_sum = np.hstack(blocks)

    mu = fr_sum.mean(axis=1, keepdims=True)
    sg = fr_sum.std(axis=1, ddof=0, keepdims=True)
    sg = np.maximum(sg, np.finfo(float).eps)
    
    if "z_scored_firing" not in data:
        data["z_scored_firing"] = np.empty_like(data["aligned_firing"],dtype=object)
        
    if "z_score_value" not in data:
        data["z_score_value"] = np.empty_like(data["aligned_firing"],dtype=object)
    
    for index in params.ana_index_grid(ana_tt, ana_bt):
        if data["aligned_firing"][index] is not None:
            data["z_scored_firing"][index] = (data["aligned_firing"][index] - mu) / sg
            data["z_score_value"][index] = np.array([mu, sg])



