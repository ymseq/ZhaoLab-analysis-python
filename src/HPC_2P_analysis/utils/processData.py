from __future__ import annotations

import copy

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .config import get_itr_index


def apply_neuron_mask_to_attracted_data(
    attracted_data: dict,
    mask: np.ndarray,
) -> dict:
    """
    Apply neuron mask to attracted_data.

    Expected input structure
    ------------------------
    attracted_data = {
        "file_info": {...},
        "zones": ...,
        "cell_ids": ...,
        "firing": object array,
    }
    """
    mask = np.asarray(mask, dtype=bool)

    masked = {}

    for key, value in attracted_data.items():
        if key == "cell_ids":
            masked[key] = np.asarray(value)[mask]

        elif key == "firing":
            firing = value
            masked_firing = np.empty_like(firing, dtype=object)

            for idx in np.ndindex(firing.shape):
                fr = firing[idx]

                if fr is None:
                    masked_firing[idx] = None
                    continue

                if fr.shape[0] != len(mask):
                    raise ValueError(
                        f"Neuron dimension mismatch in firing{idx}: "
                        f"fr.shape[0]={fr.shape[0]}, "
                        f"mask length={len(mask)}."
                    )

                masked_firing[idx] = fr[mask, :, :]

            masked[key] = masked_firing

        else:
            masked[key] = copy.deepcopy(value)

    return masked


def align_track_hpc(
    data: dict,
    *,
    gaussian_sigma: float = 0,
) -> dict:
    """
    Compute condition-averaged firing maps and SEM.

    Input
    -----
    data["firing"][tt_idx, bt_idx]:
        shape (n_neuron, n_trial, n_pos)

    Output
    ------
    data["smooth_firing"]:
        optionally smoothed trial-level firing.

    data["aligned_firing"]:
        mean over trials, shape (n_neuron, n_pos).

    data["firing_std"]:
        SEM over trials, shape (n_neuron, n_pos).
    """
    if "firing" not in data:
        raise KeyError("data must contain 'firing'.")

    smooth_firing = np.empty_like(data["firing"], dtype=object)
    aligned_firing = np.empty_like(data["firing"], dtype=object)
    firing_std = np.empty_like(data["firing"], dtype=object)

    for tt_idx, bt_idx in get_itr_index(data, ["*"], ["*"]):
        fr = data["firing"][tt_idx, bt_idx]

        if fr is None:
            smooth_firing[tt_idx, bt_idx] = None
            aligned_firing[tt_idx, bt_idx] = None
            firing_std[tt_idx, bt_idx] = None
            continue

        if fr.ndim != 3:
            raise ValueError(
                f"Expected fr with shape "
                f"(n_neuron, n_trial, n_pos), got {fr.shape}."
            )

        if fr.shape[1] == 0:
            smooth_firing[tt_idx, bt_idx] = None
            aligned_firing[tt_idx, bt_idx] = None
            firing_std[tt_idx, bt_idx] = None
            continue

        if gaussian_sigma > 0:
            fr = gaussian_filter1d(
                fr,
                sigma=gaussian_sigma,
                axis=2,
                mode="nearest",
                truncate=3.0,
            )

        smooth_firing[tt_idx, bt_idx] = fr
        aligned_firing[tt_idx, bt_idx] = np.nanmean(fr, axis=1)
        firing_std[tt_idx, bt_idx] = (
            np.nanstd(fr, axis=1, ddof=0) / np.sqrt(fr.shape[1])
        )

    data["smooth_firing"] = smooth_firing
    data["aligned_firing"] = aligned_firing
    data["firing_std"] = firing_std

    return data
