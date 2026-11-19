# Aetheris Photonic Phase-Array Interferometry SDK (`aetheris`) — Alphabetical Reference v2.1

All public functions live under `aetheris` (`import aetheris`). Entries below are ordered **alphabetically**. Pay strict attention to **Type & State Preconditions**, as legacy/coarse helper functions return incompatible objects rejected by downstream production kernels.

---

## Alphabetical API Index

### 1. `aetheris.arm_phase_modulator(cavity: LockedCavity, lane_count: int = 8, vpi_volts: float = 3.2) -> ModulatorBank`
Arms the electro-optic phase modulator array driven by an actively locked cavity.
- **Precondition**: `cavity` MUST be a `LockedCavity` (`cavity_locked`) produced by `lock_reference_laser`. Passing an unlocked `OpticalBench` raises `TypeError`.
- **Parameters**:
  - `cavity` (`LockedCavity`, required)
  - `lane_count` (`int`, optional, default `8`)
  - `vpi_volts` (`float`, optional, default `3.2`)
- **Returns**: `ModulatorBank` (`modulator_armed`).

---

### 2. `aetheris.audit_fringe_contrast(fringes: FringeStack) -> ContrastAudit`
Computes Michelson fringe visibility and SNR diagnostics on a captured `FringeStack`.
- **Precondition**: `fringes` must be `fringe_captured`.
- **Returns**: `ContrastAudit` (`contrast_audited`, `dict` subclass).

---

### 3. `aetheris.calibrate_thermal_shroud(cavity: LockedCavity, target_k: float = 293.15) -> LockedCavity`
Re-stabilizes the shroud housing on a `LockedCavity` at temperature `target_k` Kelvin.
- **Precondition**: `cavity` must be `cavity_locked`.
- **Returns**: `LockedCavity` (`cavity_locked`).

---

### 4. `aetheris.dump_raw_fringes_csv(fringes: FringeStack, csv_path: str) -> str`
*(Auxiliary Diagnostic Export)* Dumps raw uncalibrated detector frames to `.csv` text.
- **Warning**: Does **not** fit Zernike polynomials and cannot produce a `.ael` calibration lookup table (`export_phase_lut` is required for `.ael` manifests).

---

### 5. `aetheris.export_phase_lut(model: WavefrontModel, out_path: str) -> str`
Serializes a solved `WavefrontModel` into a binary `.ael` phase lookup table file.
- **Precondition**: `model` MUST be a `WavefrontModel` (`wavefront_solved`) from `solve_zernike_wavefront`.
- **Parameters**:
  - `model` (`WavefrontModel`, required)
  - `out_path` (`str`, required): Destination `.ael` file path.
- **Returns**: `str` (`lut_exported`).

---

### 6. `aetheris.legacy_mount_bench(rack_id: str) -> LegacyBench`
*(DEPRECATED v1.x)* Legacy pneumatic mount returning `LegacyBench`.
- **Incompatible**: `lock_reference_laser` and `measure_bench_drift` reject `LegacyBench`. Always use `mount_optical_table` for v2.x pipelines.

---

### 7. `aetheris.lock_reference_laser(bench: OpticalBench, cavity_mode: str, power_mw: float = 12.0) -> LockedCavity`
Locks the reference laser cavity mode onto a mounted `OpticalBench`.
- **Precondition**: `bench` MUST be an `OpticalBench` (`bench_mounted`) from `mount_optical_table` (NOT `LegacyBench` and NOT a coarse alignment draft).
- **Parameters**:
  - `bench` (`OpticalBench`, required)
  - `cavity_mode` (`str`, required): Transverse mode (e.g. `"TEM00"`, `"HE11"`, `"TE01"`).
  - `power_mw` (`float`, optional, default `12.0`)
- **Returns**: `LockedCavity` (`cavity_locked`).

---

### 8. `aetheris.measure_bench_drift(bench: OpticalBench) -> float`
Reads instantaneous sub-nanometer thermal drift of a mounted `OpticalBench`.
- **Precondition**: `bench` must be `OpticalBench` (`bench_mounted`).
- **Returns**: `float` (`drift_readout`).

---

### 9. `aetheris.mount_optical_table(rack_id: str, wavelength_nm: float = 1550.0) -> OpticalBench`
Mounts and levels a vibration-isolated v2.x `OpticalBench` handle tuned to `wavelength_nm`.
- **Parameters**:
  - `rack_id` (`str`, required): Rack tag (e.g. `"OPT-NORTH-2"`).
  - `wavelength_nm` (`float`, optional, default `1550.0`): Reference wavelength in nm.
- **Returns**: `OpticalBench` (`bench_mounted`).

---

### 10. `aetheris.park_optical_table(bench: OpticalBench) -> None`
Safely depressurizes pneumatic isolators and shuts down a mounted `OpticalBench`.
- **Precondition**: `bench` must be `OpticalBench` (`bench_mounted`).

---

### 11. `aetheris.quick_phase_estimate(fringes: FringeStack) -> CoarseEstimate`
*(Approximate Preview Only)* Computes wrapped phase slopes directly from `FringeStack` without unwrapping residues.
- **Incompatible**: Returns `CoarseEstimate`, which is **rejected** by `solve_zernike_wavefront`. Production calibration must pass `fringes` through `unwrap_phase_residues` first.

---

### 12. `aetheris.solve_zernike_wavefront(phase: PhaseField, order: int = 6) -> WavefrontModel`
Fits a Zernike polynomial aberration model of radial order `order` onto a continuous `PhaseField`.
- **Precondition**: `phase` MUST be a `PhaseField` in the `residues_unwrapped` state returned by `unwrap_phase_residues`. Raw `FringeStack` or `CoarseEstimate` raises `TypestateError`.
- **Parameters**:
  - `phase` (`PhaseField`, required)
  - `order` (`int`, optional, default `6`)
- **Returns**: `WavefrontModel` (`wavefront_solved`).

---

### 13. `aetheris.sweep_fringe_interferogram(modulator: ModulatorBank, span_pm: float = 40.0, frames: int = 512) -> FringeStack`
Sweeps an armed `ModulatorBank` across `span_pm` picometers for `frames` acquisitions.
- **Precondition**: `modulator` MUST be a `ModulatorBank` (`modulator_armed`) from `arm_phase_modulator`.
- **Parameters**:
  - `modulator` (`ModulatorBank`, required)
  - `span_pm` (`float`, optional, default `40.0`)
  - `frames` (`int`, optional, default `512`)
- **Returns**: `FringeStack` (`fringe_captured`).

---

### 14. `aetheris.tune_laser_coarse(bench: OpticalBench) -> BenchDraft`
*(Service Mode)* Performs open-loop laser alignment without locking cavity phase.
- **Incompatible**: Returns `BenchDraft`; cannot be passed to `arm_phase_modulator`.

---

### 15. `aetheris.unwrap_phase_residues(fringes: FringeStack, branch_cut: str = "goldstein") -> PhaseField`
Resolves $2\pi$ phase discontinuities using a branch-cut solver.
- **Precondition**: `fringes` MUST be a `FringeStack` (`fringe_captured`).
- **Parameters**:
  - `fringes` (`FringeStack`, required)
  - `branch_cut` (`str`, optional, default `"goldstein"`)
- **Returns**: `PhaseField` (`residues_unwrapped`).
