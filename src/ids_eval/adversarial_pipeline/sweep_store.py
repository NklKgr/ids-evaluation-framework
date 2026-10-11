from __future__ import annotations
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import numpy as np
import pickle

from ids_eval.dto.run_config import RunConfig
from ids_eval.run_config_pipeline.config_manager import ConfigManager

_VALID_RUN_ID = re.compile(r"[A-Za-z0-9_.-]+")
Status = Literal["ok", "not_applicable", "invalid", "failed"]


@dataclass(frozen=True, eq=False)
class AttackOutcome:
    name: str #name of plugin
    params: dict
    status: str # "ok" | "not_applicable" | "invalid" | "failed"
    error: str | None
    seconds: float
    pred: np.ndarray | None
    linf: np.ndarray | None
    l0: np.ndarray | None
    x_adv: np.ndarray | None

@dataclass(frozen=True, eq=False)
class SweepCellResult:
    run_id: str
    y_true: np.ndarray
    clean_pred: np.ndarray    # int8  (n,)
    positions: np.ndarray     # int64 (n,)
    attacked: np.ndarray      # bool  (n,)
    perturbable: np.ndarray   # bool  (d,)
    outcomes: tuple[AttackOutcome, ...]
    meta: dict


class SweepStore():
    """Here we handle the persisting and checking of all RI-sweep related Infos for the evaluation"""
    def __init__(self, root: Path)->None:
        self.root = Path(root)

    #we safe everything in the corresponding run directory
    @staticmethod
    def default_path(config: RunConfig)-> Path:
        return ConfigManager.get_report_directory(config) / "RI"

    #validate f path name
    def _path(self, run_id: str)->Path:
        if not _VALID_RUN_ID.fullmatch(run_id):
            raise ValueError(f"run_id {run_id!r} contains restricted symbols: abort writing.")
        file_path = self.root / f"{run_id}.pkl"
        return file_path

    #persisting the results serialized
    def write(self, cell: SweepCellResult) -> None:
        final_file = self._path(cell.run_id)#valid symbols?
        if final_file.exists():
            raise FileExistsError(f"Cell {cell.run_id!r} exists already.")
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = final_file.with_name(final_file.name + ".tmp")
        try:
            with open(tmp, "wb") as f:
                pickle.dump({"cell": cell}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, final_file) #atomic
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def has(self, run_id: str) -> bool:
        return self._path(run_id).exists()

    def read(self, run_id: str) -> SweepCellResult:
        with open(self._path(run_id), "rb") as f:
            payload = pickle.load(f)
        cell = payload["cell"]
        if not isinstance(cell, SweepCellResult):
            raise TypeError(f"Cell {run_id!r} is not a SweepCellResult.")
        return cell

    def clear(self)->None:
        for path in self.root.glob("*.pkl*"): #also tmp! -.-
            path.unlink()

    #run over sells to compute RI
    def iter_cells()->SweepCellResult:
        pass #TODO