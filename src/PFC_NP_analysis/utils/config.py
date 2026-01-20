from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import List, Dict, Iterable, Tuple
import fnmatch, itertools

from numpy import average


@dataclass
class Params:
    # basic
    version: str = "v1"
    sub_directory: str = "flexible_shift"

    # paths
    data_path: str = "D:/Data/Lab/code/python/lwx_lyr_data_analysis/data"
    results_path: str = "D:/Data/Lab/code/python/lwx_lyr_data_analysis/results"

    # trial / track
    space_unit: float = 0.1
    track_range = [100, 3100]

    # lick alignment
    len_before_lick: int = 10 * 10
    len_after_lick: int = 10 * 10

    # gaussian smooth
    gaussian_sigma: int = 50

    # PCA
    pca_n_components: int = 20
    
    # UMAP
    umap_n_neighbors: int = 50
    umap_min_dist: float = 0.3
    umap_n_components: int = 3
    umap_metric: str = "correlation"
    
    # position average
    len_pos_average: int = 1
    
    tt: List[str] = field(default_factory=list)
    bt: List[str] = field(default_factory=list)

    # Presets
    TT_PRESETS: Dict[str, List[str]] = field(default_factory=lambda: {
        "basic": [
            "couple_ACB", "couple_BCA",
            "pattern_CAB", "pattern_CBA", "pattern_ACB", "pattern_BCA", "pattern_ABC", "pattern_BAC",
            "position_CAB", "position_CBA", "position_ACB", "position_BCA", "position_ABC", "position_BAC",
        ],
        "merge": [
            "couple",
            "pattern_1", "pattern_2", "pattern_3",
            "position_1", "position_2", "position_3",
        ],
    })

    BT_PRESETS: Dict[str, List[str]] = field(default_factory=lambda: {
        "basic": ["correct", "false", "miss"],
    })

    @classmethod
    def from_presets(
        cls,
        tt_preset: str,
        bt_preset: str,
        **overrides,
    ) -> "Params":
        """
        Build a Params from named tt/bt presets.
        """
        tmp = cls()  # just to access preset dicts (they are defaults on the class)
        if tt_preset not in tmp.TT_PRESETS:
            raise KeyError(f"Unknown tt_preset: {tt_preset}. Available: {list(tmp.TT_PRESETS)}")
        if bt_preset not in tmp.BT_PRESETS:
            raise KeyError(f"Unknown bt_preset: {bt_preset}. Available: {list(tmp.BT_PRESETS)}")

        base = cls(
            tt=tmp.TT_PRESETS[tt_preset].copy(),
            bt=tmp.BT_PRESETS[bt_preset].copy(),
            **overrides,
        )
        return base

    def with_presets(self, tt_preset: str | None = None, bt_preset: str | None = None, **overrides) -> "Params":
        """
        Clone current Params but swap to another tt/bt preset (and/or override other fields).
        """
        cur = self
        tmp = self
        if tt_preset is not None:
            if tt_preset not in tmp.TT_PRESETS:
                raise KeyError(f"Unknown tt_preset: {tt_preset}. Available: {list(tmp.TT_PRESETS)}")
            cur = replace(cur, tt=tmp.TT_PRESETS[tt_preset].copy())
        if bt_preset is not None:
            if bt_preset not in tmp.BT_PRESETS:
                raise KeyError(f"Unknown bt_preset: {bt_preset}. Available: {list(tmp.BT_PRESETS)}")
            cur = replace(cur, bt=tmp.BT_PRESETS[bt_preset].copy())
        if overrides:
            cur = replace(cur, **overrides)
        return cur

    # -------------------- derived values (dynamic) --------------------
    # @property
    # def gaussian_range(self) -> int:
    #     return 2 * int(3 * self.gaussian_sigma) + 1
    
    @property
    def len_track(self) -> int:
        return self.track_range[1] - self.track_range[0]

    @property
    def len_lick(self) -> int:
        return self.len_before_lick + self.len_after_lick + 1

    @property
    def tt_idx(self) -> Dict[str, int]:
        return {name: i-1 for i, name in enumerate(self.tt, start=1)}

    @property
    def bt_idx(self) -> Dict[str, int]:
        return {name: i-1 for i, name in enumerate(self.bt, start=1)}
    
    @property
    def total_index_grid(self) -> List[Tuple[int, int]]:
        return list(itertools.product(self.tt_idx.values(), self.bt_idx.values()))
    
    # -------------------- analysis selection (supports wildcards) --------------------
    def _resolve(self, patterns: Iterable[str], universe: List[str]) -> List[str]:
        """
        Resolve names/patterns (supports '*' wildcards) against a universe list.
        Keeps universe order, deduplicates, and errors if no match for a token.
        """
        resolved: List[str] = []
        for p in patterns:
            if p in universe:  # literal first
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

    def ana_tt_indices(self, ana_tt: List[str]) -> List[int]:
        ana_tt_names = self._resolve(ana_tt, self.tt)
        m = self.tt_idx
        return [m[name] for name in ana_tt_names]

    def ana_bt_indices(self, ana_bt: List[str]) -> List[int]:
        ana_bt_names = self._resolve(ana_bt, self.bt)
        m = self.bt_idx
        return [m[name] for name in ana_bt_names]

    def ana_index_grid(self, ana_tt: List[str], ana_bt: List[str]) -> List[Tuple[int, int]]:
        return list(itertools.product(self.ana_tt_indices(ana_tt), self.ana_bt_indices(ana_bt)))


def set_params(tt_preset: str = "basic", bt_preset: str = "basic", **overrides) -> Params:
    return Params.from_presets(tt_preset, bt_preset, **overrides)

