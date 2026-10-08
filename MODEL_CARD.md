# Model Card — TRON2 RL deployment policies

> **Status:** every entry below is a **template** with `⚠ TO CONFIRM`
> placeholders. The model owner and legal must complete each field
> before the first public release. Do **not** cut a public tag while
> any `⚠ TO CONFIRM` remains.

This document covers the ten ONNX weight files checked into
`controllers/model/` in this repository. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) §2 for the
license-status summary; this file is the operational / behavioral
model card.

## Model index

| ID | Path | Size (bytes) | SHA-256 | Role |
|----|------|-------------:|---------|------|
| `SF_TRON2A/policy.onnx`  | `controllers/model/SF_TRON2A/policy.onnx`  | 791 050 | `0b353a087c912c33b9ba690560f3501cf7bf2bf25fde91c07ee2bdfb36502d3a` | Policy network for `SF_TRON2A` (sole-ankle biped) |
| `SF_TRON2A/encoder.onnx` | `controllers/model/SF_TRON2A/encoder.onnx` | 588 998 | `5f7e2b8865fda7c284f0dd98b79f5c1d78935c83cbf43bf311d768154937111e` | Observation encoder for `SF_TRON2A` |
| `WF_TRON2A/policy.onnx`  | `controllers/model/WF_TRON2A/policy.onnx`  | 770 148 | `3000df452681056738a15b46fa67f4f8436b34bbb6dcc6b22fa08b1b1f8dd071` | Policy network for `WF_TRON2A` (wheeled-foot biped) |
| `WF_TRON2A/encoder.onnx` | `controllers/model/WF_TRON2A/encoder.onnx` | 503 276 | `507d0630d78873f7aabfeab4eae9d7669610d709fcc903c4296d1908da54b3e7` | Observation encoder for `WF_TRON2A` |
| `DASF_TRON2A/policy.onnx`  | `controllers/model/DASF_TRON2A/policy.onnx`  | 863 234 | `473bae82c4b09420f1013c37b234ab7475c0183b0e7f79df50c72f744b5fa0ca` | Policy network for `DASF_TRON2A` (dual-arm sole-foot humanoid) |
| `DASF_TRON2A/encoder.onnx` | `controllers/model/DASF_TRON2A/encoder.onnx` | 852 848 | `31ff1f3f756298421c3d88e075103e5e9ada560573200f1ab42644ca7d14be51` | Observation encoder for `DASF_TRON2A` |
| `WF_TRON2A_BASE/policy.onnx` | `controllers/model/WF_TRON2A_BASE/policy.onnx` | 2466343 | `986ab5baa65b8b7276ae6361faa5eee8b82618a92ff0a5d320a79f91277b5577` | Base actor 273 -> 10 |
| `WF_TRON2A_BASE/encoder.onnx` | `controllers/model/WF_TRON2A_BASE/encoder.onnx` | 1023376 | `19ee847b56fc6911524ee566ce150a8ba7fd49e7fc33d079e504465eb99dace0` | Base history 360 -> 3 |
| `WF_TRON2A_BASE_BLIND/policy.onnx` | `controllers/model/WF_TRON2A_BASE_BLIND/policy.onnx` | 1520164 | `628ed6c7a3b0ff4aa46ddb1ae2b2d10555c0d252398ed219e5109295e550c2d7` | BaseBlind actor 42 -> 10 |
| `WF_TRON2A_BASE_BLIND/encoder.onnx` | `controllers/model/WF_TRON2A_BASE_BLIND/encoder.onnx` | 1023376 | `9dca7e0bcbdab2fbdc1eb9a34270afa05ca735372665f208b3c4c7ef184a01bc` | BaseBlind history 360 -> 3 |

SF/WF SHA-256 values were recorded 2026-07-16. Those four blobs are byte-identical to
`tron2-rl-deploy-ros/tron2_controllers/config/{SF,WF}_TRON2A/policy/{policy,encoder}.onnx`
in the sibling repository — this file **and** that repo's
`THIRD_PARTY_NOTICES.md §3` must be updated together.

DASF size and SHA-256 values were recorded from the working tree on
2026-08-20. No sibling-repository equivalence or training provenance
is asserted for those two files.

Consumed by:

- `SF_TRON2A/*` → `controllers/SolefootController.py`
- `WF_TRON2A/*` → `controllers/WheelfootController.py`
- `DASF_TRON2A/*` → `controllers/DASFController.py`
- `WF_TRON2A_BASE*/*` -> `controllers/PolicyWheelfootController.py`, via `run_policy.py` only.
  The vendor `main.py` is intentionally unchanged. These policies are localhost/simulation-only.

---

## SF_TRON2A/policy.onnx

- **Path:** `controllers/model/SF_TRON2A/policy.onnx`
- **Size / SHA-256:** 791 050 B — `0b353a087c912c33b9ba690560f3501cf7bf2bf25fde91c07ee2bdfb36502d3a`
- **Checkpoint id / hash:** ⚠ TO CONFIRM (map SHA-256 above to the internal training-run checkpoint id)
- **Training run (framework, commit, date):** ⚠ TO CONFIRM
- **Training data description:** ⚠ TO CONFIRM (simulator + domain
  randomization ranges, or real-world logs, or both — with data
  license status)
- **Evaluation:** ⚠ TO CONFIRM (sim benchmark; real-hardware velocity
  tracking, joint-limit safety, thermal envelope)
- **Intended use:** locomotion control for the `SF_TRON2A`
  (sole-ankle) variant, driven at the tick rate configured in
  `controllers/model/SF_TRON2A/params.yaml`, on a robot suspended /
  mounted for initial bring-up.
- **Out-of-scope use:** any hardware variant other than `SF_TRON2A`;
  any operating envelope outside the training / evaluation
  distribution; unsuspended operation before the operator has
  confirmed the target behavior.
- **Known limitations:** ⚠ TO CONFIRM
- **Redistribution status:** ⚠ TO CONFIRM (Apache-2.0 with the code /
  separate license / controlled external download only)

## SF_TRON2A/encoder.onnx

- **Path:** `controllers/model/SF_TRON2A/encoder.onnx`
- **Size / SHA-256:** 588 998 B — `5f7e2b8865fda7c284f0dd98b79f5c1d78935c83cbf43bf311d768154937111e`
- **Checkpoint id / hash:** ⚠ TO CONFIRM
- **Training run:** ⚠ TO CONFIRM
- **Training data description:** ⚠ TO CONFIRM
- **Evaluation:** ⚠ TO CONFIRM
- **Intended use:** observation encoding paired with the SF policy
  above; not intended to be used with any other policy.
- **Out-of-scope use:** as above.
- **Known limitations:** ⚠ TO CONFIRM
- **Redistribution status:** ⚠ TO CONFIRM

## WF_TRON2A/policy.onnx

- **Path:** `controllers/model/WF_TRON2A/policy.onnx`
- **Size / SHA-256:** 770 148 B — `3000df452681056738a15b46fa67f4f8436b34bbb6dcc6b22fa08b1b1f8dd071`
- **Checkpoint id / hash:** ⚠ TO CONFIRM
- **Training run:** ⚠ TO CONFIRM
- **Training data description:** ⚠ TO CONFIRM
- **Evaluation:** ⚠ TO CONFIRM
- **Intended use:** locomotion control for the `WF_TRON2A`
  (wheeled-foot) variant, driven at the tick rate configured in
  `controllers/model/WF_TRON2A/params.yaml`, on a robot suspended /
  mounted for initial bring-up.
- **Out-of-scope use:** any hardware variant other than `WF_TRON2A`;
  operating envelope outside the training / evaluation distribution;
  unsuspended operation before the operator has confirmed behavior.
- **Known limitations:** ⚠ TO CONFIRM
- **Redistribution status:** ⚠ TO CONFIRM

## WF_TRON2A/encoder.onnx

- **Path:** `controllers/model/WF_TRON2A/encoder.onnx`
- **Size / SHA-256:** 503 276 B — `507d0630d78873f7aabfeab4eae9d7669610d709fcc903c4296d1908da54b3e7`
- **Checkpoint id / hash:** ⚠ TO CONFIRM
- **Training run:** ⚠ TO CONFIRM
- **Training data description:** ⚠ TO CONFIRM
- **Evaluation:** ⚠ TO CONFIRM
- **Intended use:** observation encoding paired with the WF policy
  above; not intended to be used with any other policy.
- **Out-of-scope use:** as above.
- **Known limitations:** ⚠ TO CONFIRM
- **Redistribution status:** ⚠ TO CONFIRM

## DASF_TRON2A/policy.onnx

- **Path:** `controllers/model/DASF_TRON2A/policy.onnx`
- **Size / SHA-256:** 863 234 B — `473bae82c4b09420f1013c37b234ab7475c0183b0e7f79df50c72f744b5fa0ca`
- **Checkpoint id / hash:** ⚠ TO CONFIRM (map SHA-256 above to the internal training-run checkpoint id)
- **Training run (framework, commit, date):** ⚠ TO CONFIRM
- **Training data description:** ⚠ TO CONFIRM (simulator + domain
  randomization ranges, or real-world logs, or both — with data
  license status)
- **Evaluation:** ⚠ TO CONFIRM (sim benchmark; real-hardware velocity
  tracking, joint-limit safety, thermal envelope, and upper/lower-body
  coordination)
- **Intended use:** locomotion control for the `DASF_TRON2A`
  dual-arm sole-foot variant, driven at the tick rate configured in
  `controllers/model/DASF_TRON2A/params.yaml`, on a robot suspended /
  mounted for initial bring-up.
- **Out-of-scope use:** any hardware variant other than
  `DASF_TRON2A`; any operating envelope outside the training /
  evaluation distribution; unsuspended operation before the operator
  has confirmed the target behavior.
- **Known limitations:** ⚠ TO CONFIRM
- **Redistribution status:** ⚠ TO CONFIRM (Apache-2.0 with the code /
  separate license / controlled external download only)

## DASF_TRON2A/encoder.onnx

- **Path:** `controllers/model/DASF_TRON2A/encoder.onnx`
- **Size / SHA-256:** 852 848 B — `31ff1f3f756298421c3d88e075103e5e9ada560573200f1ab42644ca7d14be51`
- **Checkpoint id / hash:** ⚠ TO CONFIRM
- **Training run:** ⚠ TO CONFIRM
- **Training data description:** ⚠ TO CONFIRM
- **Evaluation:** ⚠ TO CONFIRM
- **Intended use:** observation encoding paired with the DASF policy
  above; not intended to be used with any other policy.
- **Out-of-scope use:** as above.
- **Known limitations:** ⚠ TO CONFIRM
- **Redistribution status:** ⚠ TO CONFIRM

---

## WF_TRON2A_BASE/policy.onnx

- Checkpoint: V23ext `2026-09-21_11-03-33_seedB/model_23500.pt`.
- Checkpoint SHA-256: `0d2385f11452e3a0f5900a211c74af8b67e1ef885bb640be9aa5ec2d20cce983`.
- Training: Isaac simulation; full training-data rights and redistribution: ⚠ TO CONFIRM.
- Evaluation: 300-step flat/stairs plus 15-degree roll/pitch reset replay, atol 1e-5, rtol 0.
- Hardware evaluation: none; IMU mount, timestamp clock and firmware watchdog remain unverified.

## WF_TRON2A_BASE/encoder.onnx

Paired exclusively with Base above, same checkpoint/provenance and evaluation.
Input is ten oldest-first 36-element frames. Redistribution: ⚠ TO CONFIRM.

## WF_TRON2A_BASE_BLIND/policy.onnx

- Checkpoint: V22 `2026-09-18_13-49-07_seedB/model_14500.pt`.
- Checkpoint SHA-256: `ceb648783b5c40f0f85eb6411454ad030104b51b6f7e0ef3ccf3703c8ed3cccb`.
- Training: Isaac simulation; full training-data rights and redistribution: ⚠ TO CONFIRM.
- Evaluation: 300-step flat/stairs replay, including an automatic episode reset, atol 1e-5, rtol 0.
- Hardware evaluation: none; same unverified hardware prerequisites as Base.

## WF_TRON2A_BASE_BLIND/encoder.onnx

Paired exclusively with BaseBlind above, same checkpoint/provenance and evaluation.
Input is ten oldest-first 36-element frames. Redistribution: ⚠ TO CONFIRM.

### Base model regeneration

The four Base artifacts use FP64 internal arithmetic and FLOAT32 input/output to avoid
FP32 ONNX accumulation error exceeding the strict Isaac parity threshold. Trained weights
are unchanged. Models embed `tron2.internal_precision` and `tron2.source_sha256` metadata.
The standard filenames are replaced in place; there are no `stable*.onnx` runtime files.
No new inference dependency is needed (`onnx` is only an export-time dependency).

1. From sibling `tron2_rl`, export each checkpoint using `export_onnx.py` into a scratch
   directory, under `OMNI_KIT_ACCEPT_EULA=YES flock -w 1800 /tmp/isaac.lock timeout 900 ./run.sh`.
   Use Base-Play / BaseBlind-Play and the checkpoints above.
2. For each policy's `policy.onnx` and `encoder.onnx`, from this repo run
   `python tools/stabilize_onnx.py --source <scratch-file> --out <deployment-file>`.
   The output paths are the corresponding standard filenames in the model index above.
   Conversion rejects already-converted/unsupported graphs; always begin with fresh exports.
3. Update hashes/sizes here and in the notices. From the parent repository run
   `python3 -m pytest tron2_deploy/tests/test_parity.py -q` with `TRON2_TRACE_DIR` pointing
   to the independent Isaac traces. The test checks the FLOAT32 interface and precision metadata.
4. Re-run both `run_policy.py --policy base --dry-run` and `--policy base_blind --dry-run`.
   Do not use these results as permission for real-robot operation.

## Update procedure

Any change to a `*.onnx` file under `controllers/model/` must, in the
same pull request:

1. Update the size in the "Model index" table above.
2. Update the checkpoint id / hash and training-run fields in the
   corresponding section.
3. Update the matching row in
   [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) §2.
4. Update [`CHANGELOG.md`](CHANGELOG.md) under `[Unreleased]`.

CI enforces (1) and (3): a `*.onnx` addition without a matching
`THIRD_PARTY_NOTICES.md` and `MODEL_CARD.md` reference will fail the
merge.
