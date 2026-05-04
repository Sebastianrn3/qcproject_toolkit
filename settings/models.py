from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class JobConfig:
    jobname: str
    charge: int | str
    unpaired_electrons: int = 0
    mopac_path: Path | None = None
    mopac_exe: Path | None = None
    inputs_folder: Path | None = None
    geometries_folder: Path | None = None
    opt_folder: Path | None = None
    neb_folder: Path | None = None
    folder: Path | None = None
    fixed_atoms: list[int] | None = None
    rigid_groups: list[list[int]] | None = None
    analysis_folder: Path | None = None