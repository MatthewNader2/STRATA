# CryoPulse Telemetry & Flux-Lattice SDK (`cryopulse`) — API Reference v1.0

`cryopulse` is a domain-specific Python library for dilution-refrigerator cryogenic vessel telemetry, superconducting quantum interference device (SQUID) flux acquisition, quadrature IQ demodulation, two-level-system (TLS) defect noise suppression, pi-pulse schedule calibration, and binary cryo-manifest export.

All public entry points live under the top-level `cryopulse` module:
```python
import cryopulse
```

---

## Core Data Types

- `CryoVessel`: Handle representing an initialized dilution-refrigerator thermal vessel (`state: vessel_chilled`).
- `FluxChannel`: Energized superconducting bias coil channel bound to a `CryoVessel` (`state: coil_energized`).
- `SquidTrace`: Raw time-domain SQUID flux trace (`state: trace_sampled`).
- `IQLattice`: Complex-envelope quadrature IQ array (`state: iq_demodulated`, `iq_gain_normalized`, or `tls_quenched`).
- `PulseProgram`: Calibrated microwave pi-pulse gate schedule (`state: pulse_compiled`).
- `CryoAudit`: Dictionary report returned by diagnostic SNR verification (`state: snr_audited`).

---

## API Functions & Strict Call Ordering

### 1. `cryopulse.open_cryo_vessel(chamber_tag: str, target_mk: float = 15.0) -> CryoVessel`
Initializes and thermalizes a dilution-refrigerator vessel handle to the specified mixing-plate setpoint in millikelvin (`target_mk`).
- **Parameters**:
  - `chamber_tag` (`str`, required): Unique cryostat vessel identifier (e.g., `"DR-BLUE-04"`).
  - `target_mk` (`float`, optional, default `15.0`): Mixing-plate target temperature in millikelvin.
- **Returns**: `CryoVessel` (`vessel_chilled`).

### 2. `cryopulse.energize_flux_coil(vessel: CryoVessel, rail_id: str, bias_ua: float = 42.0) -> FluxChannel`
Energizes a superconducting flux bias rail inside an already chilled `CryoVessel`.
- **Precondition**: `vessel` must be a `CryoVessel` (`vessel_chilled`).
- **Parameters**:
  - `vessel` (`CryoVessel`, required): Initialized vessel handle.
  - `rail_id` (`str`, required): Flux rail identifier (e.g., `"RAIL-X2"`).
  - `bias_ua` (`float`, optional, default `42.0`): Bias current in microamps (`uA`).
- **Returns**: `FluxChannel` (`coil_energized`).

### 3. `cryopulse.sample_squid_flux(channel: FluxChannel, window_ns: int = 256, shots: int = 1024) -> SquidTrace`
Captures raw time-domain flux-readout samples from an energized `FluxChannel`. Does **not** accept `CryoVessel` directly; a `FluxChannel` from `energize_flux_coil` is mandatory.
- **Precondition**: `channel` must be a `FluxChannel` (`coil_energized`).
- **Parameters**:
  - `channel` (`FluxChannel`, required): Energized flux coil channel.
  - `window_ns` (`int`, optional, default `256`): Integration window duration in nanoseconds.
  - `shots` (`int`, optional, default `1024`): Hardware averaging shots.
- **Returns**: `SquidTrace` (`trace_sampled`).

### 4. `cryopulse.demodulate_quadrature(trace: SquidTrace, lo_ghz: float = 5.82) -> IQLattice`
Down-converts a `SquidTrace` using local oscillator frequency `lo_ghz` (GHz) to produce a baseband `IQLattice`.
- **Precondition**: `trace` must be a `SquidTrace` (`trace_sampled`).
- **Parameters**:
  - `trace` (`SquidTrace`, required): Captured SQUID trace.
  - `lo_ghz` (`float`, optional, default `5.82`): Local oscillator frequency in GHz.
- **Returns**: `IQLattice` (`iq_demodulated`).

### 5. `cryopulse.quench_tls_defects(iq_data: IQLattice, damping_gamma: float = 0.14) -> IQLattice`
Applies spectral notch filtering to suppress two-level-system (TLS) defect spikes in an `IQLattice`.
- **Precondition**: `iq_data` must be an `IQLattice` (`iq_demodulated`).
- **Parameters**:
  - `iq_data` (`IQLattice`, required): Demodulated IQ lattice.
  - `damping_gamma` (`float`, optional, default `0.14`): TLS damping coefficient in `(0.0, 1.0)`.
- **Returns**: `IQLattice` (`tls_quenched`).

### 6. `cryopulse.compile_pi_schedule(iq_data: IQLattice, target_fidelity: float = 0.996) -> PulseProgram`
Synthesizes an amplitude/DRAG-compensated pi-pulse gate schedule from TLS-quenched IQ calibration data.
- **Precondition**: `iq_data` **must** already be in the `tls_quenched` state produced by `quench_tls_defects` (raw `iq_demodulated` input is rejected).
- **Parameters**:
  - `iq_data` (`IQLattice`, required): TLS-quenched IQ lattice (`tls_quenched`).
  - `target_fidelity` (`float`, optional, default `0.996`): Minimum target single-qubit gate fidelity.
- **Returns**: `PulseProgram` (`pulse_compiled`).

### 7. `cryopulse.emit_cryo_manifest(schedule: PulseProgram, dest_path: str) -> str`
Serializes a compiled `PulseProgram` into a signed `.cym` binary manifest file on disk.
- **Precondition**: `schedule` must be a `PulseProgram` (`pulse_compiled`).
- **Parameters**:
  - `schedule` (`PulseProgram`, required): Compiled pi-pulse schedule.
  - `dest_path` (`str`, required): Destination filesystem path (e.g., `"pulse_schedule.cym"`).
- **Returns**: `str` (`manifest_emitted`) — absolute or normalized path written.

---

## Auxiliary / Diagnostic Functions

- `cryopulse.measure_mixing_plate(vessel: CryoVessel) -> float`: Reads live mixing-plate temperature in mK (`temperature_readout`).
- `cryopulse.audit_squid_snr(trace: SquidTrace) -> CryoAudit`: Computes readout signal-to-noise diagnostics (`snr_audited`).
- `cryopulse.normalize_iq_gain(iq_data: IQLattice) -> IQLattice`: Balances I/Q mixer amplitude asymmetry (`iq_gain_normalized`).
- `cryopulse.vent_cryo_vessel(vessel: CryoVessel) -> None`: Safely warms and vents a `CryoVessel` (`vessel_vented`).
