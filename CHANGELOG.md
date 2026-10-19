# Changelog

All notable changes to `tron2-rl-deploy-python` will be documented
here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Replace Base weights with user-approved Base-RealMap seed43 @26500; retain
  FP64-internal ONNX arithmetic, the identical hardened deployment contract,
  and byte-identical V22 BaseBlind artifacts. Retrain/perception failures remain
  disclosed in MODEL_CARD.md; this is not hardware qualification.

### Added
- `run_policy.py --standup learned [--standup-model DIR]`: seated route to A'/A'_p
  (as `--standup-stop-at A`), then the learned stand-up actor (`WF_TRON2A_STANDUP`: own 47/10
  contract `standup_contract.py`, incremental joint targets at 1.5 rad/s with anti-windup and the
  URDF 3 degree clamp, wheel feed-forward torque 8 N.m/unit), stand-up tilt latch 1.2 rad, start
  check against A'/A'_p, episode timeout latch, FK+IMU height estimate for the success hold, then
  the Base/BaseBlind handoff. `JointTargets.tau` (feed-forward torque) is published to the SDK.
  The shipped stand-up ONNX is a stand-in that does not stand up.
- `standup.handoff_power_on_seconds` (default 0): after a stand-up handoff the locomotion policy
  starts at once on its pre-filled history. Isaac positive control: the 3 s PD-hold power-on blend
  let 32/32 robots handed off from the ideal standing pose fall; 0 s: 0/32.
- `run_policy.py --standup scripted`: `PolicyWheelfootController` tracks the
  shared `tron2_common` seated stand-up reference (P0 hold, P1-P3, scripted
  P4) with PD, a 1.0 rad/s target slew, the reference's own 3 degree limit
  clamp and a 0.8x torque pre-clip, then hands off to the Base/BaseBlind
  policy (history pre-filled with the final measured state, zero EMA, 0.5 s
  zero command). New `standup:` contract block (stand-up tilt threshold
  1.2 rad while rising; the damping latch stays armed). Certification runs
  on a worker thread with the phase clock paused at the end of P0; rejected
  or over-budget starts trip the latch. Not a hardware qualification: in
  MuJoCo the scripted P4 does not stand the robot (see tron2_rl
  `standup/DESIGN.md` decision 9i) and limp-settled starts are rejected by
  the reference.
- Open-source scaffolding: `NOTICE`, `THIRD_PARTY_NOTICES.md`,
  `MODEL_CARD.md`, `SECURITY.md`, `CONTRIBUTING.md`, `CHANGELOG.md`.
- GitHub CI workflow: `ruff` lint, `python -m py_compile` over every
  `.py`, submodule status check, EXIF sanity over `doc/`, private-IP
  scan with allowlist, ONNX-provenance gate (fails if a new `*.onnx`
  is added without a matching `THIRD_PARTY_NOTICES.md` row and
  `MODEL_CARD.md` section), and a hard deny-list for `*.pt`, `*.pth`,
  `*.ckpt`, `*.so`, `*.dll`, `*.dylib`, `*.lib`, `*.whl`, `*.bag`,
  `*.mcap`.
- Issue templates and PR template under `.github/`, plus `CODEOWNERS`
  routing to maintainers / legal / model / sdk / robotics / safety
  teams.
- `MODEL_CARD.md` at the repo root with template rows for all six
  checked-in ONNX files (`SF_TRON2A/policy.onnx`,
  `SF_TRON2A/encoder.onnx`, `WF_TRON2A/policy.onnx`,
  `WF_TRON2A/encoder.onnx`, `DASF_TRON2A/policy.onnx`, and
  `DASF_TRON2A/encoder.onnx`).
- `DASF_TRON2A` deployment through `DASFController.py`, including
  Centaur SDK lower/upper-body channels and local-MuJoCo versus
  real-hardware joint-coordinate handling.
- `README.md`: SPDX header; "License & attribution" cross-links;
  "Scope / not included" (calls out the ONNX policy / encoder files
  pending provenance, real-hardware control code, and the
  `limxsdk-lowlevel` submodule pending clearance); "Verification"
  section with `python -m py_compile`, dry-run import, ONNX / YAML
  sanity, and sim-only guidance; "Cite & support" with
  `contact@limxdynamics.com`.

### Changed
- Base/BaseBlind ONNX arithmetic stabilized with FP64 internals and FLOAT32 interfaces
  for 1e-5 Isaac trace parity; added a reproducible conversion tool, artifact provenance,
  and five 300-step replay cases plus a zero-default-pose negative control.
- `.gitignore` expanded from a single-line `__pycache__` to a
  Python-appropriate ignore list (venv, caches, editor / OS junk,
  build artifacts), plus a hard deny-list for weight / SDK / bag
  artifacts. The six grandfathered ONNX files under
  `controllers/model/**` are exempted via an explicit `!` rule so the
  deny-list applies everywhere else in the tree.
- `README.md`, `SECURITY.md`, `CONTRIBUTING.md`, and command examples
  in shipped Markdown / YAML now consistently use `<robot-ip>` as a
  placeholder token that users must substitute with their own robot
  or simulator IP. No production or internal-network IP is embedded
  in this repository.

### Resolved (2026-07-16)
- **Private-IP handling** — resolved 2026-07-16 per owner decision.
  All Markdown / YAML command examples now use `<robot-ip>` as a
  placeholder token. No production IP is embedded in this
  repository. The sibling `tron2-rl-deploy-ros` retains a
  documentation-example literal `10.192.1.2` in its source / launch
  files, declared in that repo's `SECURITY.md`.

### Pending owner sign-off (blocks first public tag)
- **ONNX provenance for `controllers/model/SF_TRON2A/policy.onnx`** —
  training run, training data source and license boundary,
  redistribution terms. See
  [`MODEL_CARD.md`](MODEL_CARD.md#sf_tron2apolicyonnx).
- **ONNX provenance for `controllers/model/SF_TRON2A/encoder.onnx`** —
  same fields. See
  [`MODEL_CARD.md`](MODEL_CARD.md#sf_tron2aencoderonnx).
- **ONNX provenance for `controllers/model/WF_TRON2A/policy.onnx`** —
  same fields. See
  [`MODEL_CARD.md`](MODEL_CARD.md#wf_tron2apolicyonnx).
- **ONNX provenance for `controllers/model/WF_TRON2A/encoder.onnx`** —
  same fields. See
  [`MODEL_CARD.md`](MODEL_CARD.md#wf_tron2aencoderonnx).
- **ONNX provenance for `controllers/model/DASF_TRON2A/policy.onnx`** —
  same fields. See
  [`MODEL_CARD.md`](MODEL_CARD.md#dasf_tron2apolicyonnx).
- **ONNX provenance for `controllers/model/DASF_TRON2A/encoder.onnx`** —
  same fields. See
  [`MODEL_CARD.md`](MODEL_CARD.md#dasf_tron2aencoderonnx).
- **`limxsdk-lowlevel` submodule commit clearance** — SDK owner /
  legal must clear the upstream repository's license and confirm
  that consuming the SDK via a submodule pin is acceptable for a
  public release. Do **not** move the pin without written sign-off.
- **Model card completion** — every `⚠ TO CONFIRM` row in
  [`MODEL_CARD.md`](MODEL_CARD.md) resolved.
- **`controllers/model/*/params.yaml` review** — confirm no per-serial
  calibration constants are embedded.
- **Documentation media (`doc/*.jpg`, `doc/*.GIF`, `doc/*.gif`)** —
  EXIF strip and content review for individuals / facilities /
  non-public hardware.

## [0.1.0] — TBD

First public release. Contents:

- Python controller entry point and SF / WF / DASF controllers for
  `SF_TRON2A`, `WF_TRON2A`, and `DASF_TRON2A` variants.
- Six ONNX inference blobs under `controllers/model/*/` (subject to
  the Pending owner sign-off items above).
- LimX SDK consumed via the `limxsdk-lowlevel` submodule; SDK wheel is
  **not** vendored.

[Unreleased]: https://github.com/limx-tron2/tron2-rl-deploy-python/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/limx-tron2/tron2-rl-deploy-python/releases/tag/v0.1.0
