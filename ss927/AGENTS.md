# Repository Guidelines

## Project Context & Source of Truth

This repository is maintained as a long-running engineering project. Do not assume that the current conversation contains the complete or latest project context.

The repository is the source of truth.

When repository contents, documentation, current code, and conversation history disagree, prefer them in the following order:

1. Current source code
2. Current runtime configuration and interfaces
3. `docs/CURRENT_STATE.md`
4. `docs/ARCHITECTURE.md`
5. `docs/PROJECT_CONTEXT.md`
6. Conversation history

Do not rely on remembered conversation details when they can be verified from the repository.

If information cannot be confirmed from the repository, explicitly treat it as unknown rather than guessing.

---

## Session Bootstrap

Before performing a non-trivial implementation, debugging, architectural, or algorithm-design task, establish the relevant project context.

Read these files when they exist:

1. `docs/PROJECT_CONTEXT.md`
2. `docs/ARCHITECTURE.md`
3. `docs/CURRENT_STATE.md`

Then inspect the source files relevant to the requested task.

Do not repeatedly scan the entire repository when the task only concerns a known subsystem. Use the project documents as a map, then inspect the authoritative implementation.

For very small or isolated changes, reading only the relevant source files is acceptable.

---

## Context Recovery Protocol

If the conversation appears compressed, truncated, incomplete, or you are uncertain about previous decisions or the current implementation state, recover context from the repository before continuing.

Use the following recovery sequence:

1. Read `docs/CURRENT_STATE.md`.
2. Read the relevant sections of `docs/PROJECT_CONTEXT.md`.
3. Read the relevant sections of `docs/ARCHITECTURE.md`.
4. Inspect the source files involved in the current task.
5. Inspect current repository changes with `git status` and `git diff` when Git metadata is available.
6. Inspect recent commits when Git history is available and relevant.
7. Reconstruct the current implementation state before modifying code.

Do not continue an old plan merely because it appears in conversation history.

A documented plan may be outdated if the source code has already changed.

When recovering context, distinguish clearly between:

* what is currently implemented,
* what was previously attempted,
* what is planned but not implemented,
* and what is only a hypothesis.

---

## Project Structure & Module Organization

This repository builds a C++17 PTZ single-person tracking library for Huawei Ascend devices.

`LightTracker.cpp` and `LightTracker.h` contain the main tracking pipeline and state machines.

Model wrappers are split by function:

* `Detector.*`
* `PersonReID.*`
* `PoseEstimator.*`
* `FaceRecognition*`
* `Facekps.*`

Motion and association helpers live in:

* `Kalman*`
* `utils.*`
* `IveGmc.*`

Use `c_api/fx_tracker.h` for new integrations.

The root-level `Track.*` and `fx_wrapper_track.*` interfaces are legacy compatibility layers and should not be used for new integrations unless compatibility work explicitly requires them.

Documentation and architecture diagrams are under `docs/`.

Device build instructions and a C example are in `c_api/`.

---

## Project Documentation Roles

The project documentation has different responsibilities.

### `docs/PROJECT_CONTEXT.md`

Contains stable, long-term project knowledge.

Typical contents include:

* project goals and deployment scenario,
* supported hardware and runtime environment,
* major subsystems,
* model responsibilities,
* important data structures,
* coordinate conventions,
* tracking philosophy,
* long-term algorithm decisions,
* hardware constraints,
* performance constraints,
* important design rationale.

Do not fill this document with temporary debugging notes or short-lived experiments.

### `docs/ARCHITECTURE.md`

Contains the technical architecture of the current system.

Typical contents include:

* system pipeline,
* module dependencies,
* major call chains,
* state machines,
* data flow,
* model invocation flow,
* target association flow,
* recovery flow,
* PTZ-related interactions,
* public API boundaries.

Architecture descriptions must reflect the actual implementation.

### `docs/CURRENT_STATE.md`

Contains short- and medium-term engineering state.

Typical contents include:

* current development goal,
* recently completed work,
* current implementation state,
* active problems,
* known failure scenarios,
* experiments and their conclusions,
* attempted approaches that should not be repeated,
* unresolved questions,
* next recommended work,
* important files currently being modified.

This is the primary document for recovering work after a new Codex session or context compaction.

---

## Build, Test, and Development Commands

There is no CMake, Makefile, local executable, or automated test suite.

Follow `c_api/README.md` and cross-compile all maintained sources into `libfxtracker.so`:

```bash
g++ -std=c++17 -O2 -fPIC -shared \
  AclRuntime.cpp LightTracker.cpp KalmanBoxTracker.cpp KalmanFilter.cpp \
  PoseEstimator.cpp Detector.cpp PersonReID.cpp \
  FaceRecognition.cpp FaceRecognitionSystem.cpp Facekps.cpp ModelProcess.cpp \
  utils.cpp c_api/fx_tracker.cpp \
  -Ic_api -I${OPENCV_INC} -I${ASCEND_ACL_INC} \
  -L${OPENCV_LIB} -lopencv_core -lopencv_imgproc -lopencv_video \
  -lopencv_calib3d -lopencv_features2d \
  -L${ASCEND_ACL_LIB} -lascendcl -o libfxtracker.so
```

Do not add `IveGmc.cpp` unless building with `USE_HISI_IVE`.

This source list includes the centralized ACL runtime owner. It has not been verified in this
workspace because the local OpenCV/Ascend cross toolchain is unavailable; check
`docs/CURRENT_STATE.md` before device integration.

For diagnostics, compile with:

```bash
-DFX_TRACKER_MATCH_TRACE=1
```

and inspect:

```text
/tmp/fx_tracker_match_trace.log
```

Do not invent local build or test commands that are not present in the repository.

---

## Coding Style & Naming Conventions

Match the existing C++ style:

* four-space indentation,
* braces on the same line,
* `snake_case` for methods and variables,
* `kPascalCase` for constants.

Preserve the project-wide `xyxy` bounding-box convention.

Do not silently introduce another box representation inside tracking or association code.

The existing explicit exception is `PoseResult::box`, which is a normal OpenCV `xywh` rectangle; convert it at the Pose/tracker boundary before using tracker geometry helpers.

Prefer timestamp-based durations in milliseconds over frame-count timing because runtime frame rate is not guaranteed to be constant.

Keep comments concise.

Use Chinese where surrounding algorithm comments are Chinese.

Do not perform unrelated formatting or large-scale refactoring while fixing an isolated algorithm problem.

Prefer minimal, reviewable changes.

---

## Tracking Safety Principles

This project controls a PTZ camera. Tracking correctness is more important than producing a target box every frame.

Treat following the wrong person as significantly worse than temporarily returning no target.

When uncertain about target identity, prefer a safe hold or explicit lost/recovery state over aggressive reassociation.

Changes to association or recovery logic must consider:

* ID switches,
* false target following,
* occlusion,
* crossing targets,
* PTZ-induced image motion,
* partial-body visibility,
* head-only visibility,
* face-only recovery,
* ReID ambiguity,
* model invocation cost,
* recovery latency,
* output stability.

Do not optimize reacquisition latency at the expense of substantially increasing wrong-target risk without explicitly discussing the tradeoff.

---

## Algorithm Change Guidelines

Before changing tracking, matching, recovery, or target-selection logic:

1. Identify the current state-machine path involved.
2. Identify the existing gates and thresholds.
3. Identify which model calls are triggered.
4. Determine whether the proposed change affects normal tracking, lost-target recovery, or both.
5. Check whether PTZ motion invalidates image-space assumptions.
6. Preserve existing safety behavior unless the requested change explicitly replaces it.

Avoid adding independent heuristics without understanding how they interact with existing gates.

Prefer changes that have a clear role in the pipeline and can be evaluated independently.

When modifying thresholds, explain what distribution or failure scenario the threshold is intended to separate.

When modifying model-call scheduling, consider both worst-case inference count and latency.

---

## Testing Guidelines

Validate meaningful tracking changes on reproducible device-side video replays.

At minimum, consider scenarios covering:

* normal motion,
* front/back crossings,
* occlusion,
* different-clothed entrants,
* similar-looking entrants when available,
* head-only recovery,
* face-only recovery,
* partial-body visibility,
* target leaving and re-entering the frame,
* PTZ movement.

Report relevant metrics when available:

* ID switches,
* false-follow time,
* safe holds,
* reacquisition latency,
* returned-box jitter,
* model-call counts,
* p50 frame time,
* p95 frame time,
* maximum frame time.

For algorithm changes, compare behavior before and after the modification whenever reproducible replay data is available.

A wrong target is worse than a temporary no-output hold.

Do not claim an algorithm improvement solely because a single replay looks better.

---

## Debugging Guidelines

When investigating a tracking failure, first determine which stage failed.

Typical stages include:

1. detection,
2. target prediction,
3. candidate generation,
4. motion gating,
5. ReID matching,
6. face matching,
7. pose/partial-body handling,
8. target confirmation,
9. state transition,
10. returned-box smoothing,
11. PTZ-related compensation.

Prefer identifying the first incorrect decision rather than debugging only the final visible symptom.

Use existing trace infrastructure when possible before adding new permanent logging.

Temporary debug logging should not become part of the normal runtime unless it provides lasting diagnostic value.

---

## Commit & Pull Request Guidelines

This snapshot contains no Git history, so no repository-specific commit convention can be inferred.

Use short imperative subjects, for example:

```text
Fix global ReID candidate rotation
```

Pull requests should describe:

* the failure scenario,
* the root cause,
* changed gates or budgets,
* important state-machine changes,
* replay evidence,
* relevant trace excerpts,
* model-call counts,
* latency impact.

Avoid combining unrelated parameter tuning and structural changes.

When possible, separate:

* algorithm changes,
* threshold tuning,
* refactoring,
* diagnostics.

---

## Runtime Configuration

Device models are loaded from:

```text
/oem/model
```

Do not commit:

* model binaries,
* credentials,
* device logs containing sensitive imagery,
* machine-specific SDK paths,
* temporary inference outputs,
* large generated artifacts.

Do not hard-code developer-machine paths into maintained source code.

---

## Documentation Maintenance

After completing a significant implementation or design change, determine whether project documentation must be updated.

Update `docs/CURRENT_STATE.md` when the change affects:

* current implementation state,
* active work,
* known problems,
* completed tasks,
* current experiments,
* next steps,
* important failure scenarios.

Update `docs/ARCHITECTURE.md` when the change affects:

* pipeline structure,
* module relationships,
* state-machine behavior,
* major call chains,
* API boundaries,
* model invocation flow,
* data flow.

Update `docs/PROJECT_CONTEXT.md` only when the change affects long-term project knowledge, such as:

* major design decisions,
* hardware constraints,
* subsystem responsibilities,
* stable conventions,
* long-term algorithm strategy.

Do not update documentation merely to record every small code edit.

Do not duplicate the same information across all three documents.

When documentation and implementation disagree, update the documentation to match the implementation unless the task explicitly requires changing the implementation to match a documented design.

---

## End-of-Task State Capture

After a substantial development session, especially one involving tracking logic, architecture, recovery strategy, model scheduling, or important parameter decisions, capture enough state for a future session to continue without relying on the current conversation.

Before finishing, verify that `docs/CURRENT_STATE.md` answers:

1. What problem are we currently solving?
2. What was changed?
3. What is currently implemented?
4. What important conclusions were reached?
5. What approaches were rejected or failed, and why?
6. What problems remain?
7. What should be done next?
8. Which source files are most relevant?

Do not store raw conversation summaries.

Store engineering conclusions and repository state.

---

## Working Principle

Treat conversation history as temporary working memory.

Treat repository documentation and source code as persistent project memory.

When context is uncertain, recover from the repository instead of guessing.
