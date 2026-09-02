# PTZ 主目标跟踪算法 — 全面评审报告 (2026-07)

**Scope**: full review of the tracking algorithm for abnormal-matching risks (异常匹配) —
both concrete defects and unreasonable mechanisms — plus optimization directions.
**Verdict in one line**: the cascade's *defensive* design (prefer-no-match, veto stack,
provisional commit) is sound; the residual risk concentrates in a handful of **trust
amplifiers** (face override, head continuity, gallery self-reinforcement) that can convert a
single wrong signal into a persistent wrong lock, plus one **real bug in newly added code**
(fixed with this report) and several **uninitialized/collision edge cases**.

---

## 0. Methodology & coverage

Line-by-line read of `LightTracker.cpp` (3,971 lines — update flow, `setMainTarget`,
`matchPersonFaces`, `extract_detections`, `assign_cascade`, occlusion FSM, candidate
scoring/fusion PASS 1/2, all veto arms, face STEP-3 + full-frame sweep, C-identity /
provisional-commit gate, anchor-gallery admission paths, `BOX_COMPLETE`,
`update_trackers_unified`, secondary tracks/features, quality monitor, visibility assessment,
pose cache, head continuity, coast budget, deferred face registration, cleanup, reset) and
`LightTracker.h` (all tuning constants). Full read of `KalmanBoxTracker.cpp/.h` (body + head
KF, anchor gallery), `KalmanFilter.cpp`, `utils.cpp` (iou/convert/speed),
`PoseEstimator.cpp` (proportions, `find_best_match`), `c_api/fx_tracker.cpp`,
`FaceRecognitionSystem` (threshold layer), `MainTargetPredictor.h`. GMC internals skipped
(`gmc_enabled_ = false`).

**Box convention — definitively resolved** (it caused a real bug, A1): the project stuffs
**xyxy into `cv::Rect`** — `.width/.height` hold **x2/y2**. Proof: `LightTracker.cpp:196`
(`mainBox.width - mainBox.x`), `:659-660` (area as `(width-x)*(height-y)`),
`PersonReID.cpp:203-206` (`x2 = min(box.width, …)`), and the explicit comment in
`try_deferred_face_register`. Detection Mats are xyxy; the live `Utils::iou_single`
(utils.cpp:296) is xyxy-correct; `PoseResult.box` is the *documented exception* (true xywh).
Any new code doing OpenCV ROI cropping must convert explicitly.

Priorities: **P0** = broken now; **P1** = catastrophic-when-hit, plausible; **P2** = real but
bounded; **P3** = hygiene/latent.

---

## 1. Confirmed defects (代码缺陷)

### A1 (P0) — `compute_color_hist` misread xyxy as xywh — **FIXED with this report**
- **Where**: `LightTracker.cpp` `compute_color_hist` (cols 2,3 used as w/h) and the width
  gate in `update_secondary_features`.
- **Mechanism**: detection Mats are xyxy; the ROI was computed from `bw=x2, bh=y2` →
  wildly oversized/offset crop, clamped to frame → histogram of mostly background.
- **Failure**: the SEC_EXCL appearance-exclusion veto's color corroboration (`kSecExclColorMin`)
  compared noise vs noise — could corroborate wrong vetoes or block valid ones. The other two
  guards (sim floor + relative margin) still stood, so blast radius was "exclusion veto less
  reliable," not "primary lost."
- **Fix applied**: `bw = x2-x1, bh = y2-y1` at both sites + comment warning not to imitate
  `compute_embedding`'s xyxy-in-Rect (that is PersonReID's *internal* convention).

### A2 (P1) — `get_predicted_tracks` can emit uninitialized boxes
- **Where**: `LightTracker.cpp:1050-1082`.
- **Mechanism**: `info.trks/velocities/...` are allocated **lazily on the first
  `checkRange`-valid tracker** and **never zeroed**; any tracker whose KF prediction fails
  `checkRange` (NaN/Inf) leaves its row as uninitialized heap memory.
- **Failure**: one NaN KF (degenerate update, division blow-up) → a garbage "ghost box"
  participates in IoU, occluder identification, coexist veto, M1 — nondeterministic matching.
- **Fix**: allocate with `cv::Mat::zeros` upfront + keep a validity mask (or hard-reset a KF
  that produces NaN). Cheap, no behavior change in the healthy path.

### A3 (P1) — `setMainTarget` registers an arbitrary face when the operator box contains 2+
- **Where**: `LightTracker.cpp:204-212` (`face_box` overwritten per detection; last one wins).
- **Failure**: designation in a crowd → a bystander's face inside the main box becomes the
  **identity ground truth**; every later face confirmation then *pulls the lock to the wrong
  person* — inverted identity for the whole session (until deferred re-registration happens to
  replace it, which requires quality conditions the wrong face may satisfy).
- **Fix**: select the face geometrically consistent with the target: inside the head box if
  `headCNT ≥ 1`, else highest+most-central in the upper 45% band (same rule as
  `matchPersonFaces`). Refuse registration when two comparable faces are both plausible
  (register later via `try_deferred_face_register`, which is well-guarded).

### A4 (P2) — always-on per-frame `std::cout` in `update()`
- **Where**: `LightTracker.cpp:725` (`"frame distance : … ms"`).
- All other logging was deliberately null-sunk for FPS (UART I/O is real time on device);
  this survivor prints every frame. `[MATCH]` (kMatchTrace) is intentional — keep, but this
  one should be removed or folded into kMatchTrace.

### A5 (P2) — tracker-id recycling can collide with live ids
- **Where**: `update_secondary_tracks` (`KalmanBoxTracker::count >= 800 → set_count(1)`).
- **Failure**: recycled ids ignore living trackers. A long-lived secondary (person standing
  around for the whole session) can later share an id with a new track; worst case a secondary
  reaches `main_id` → its output row is `main_id+1` → `got_main` / the C API report the
  **wrong box as the main target** to the gimbal.
- **Fix**: `set_count(max_live_id + 1)` — the commented-out line above it is the correct
  intent; finish it.

### A6 (P3) — OC-SORT "no previous observation" sentinel is dead — **FIXED**
- `KalmanBoxTracker::update`: `cv::sum(last_observation)[0] >= 0` is always true because
  `last_observation` initializes to zeros (OC-SORT uses −1). First update computes
  `speed_direction` from the origin box → one-frame garbage velocity/`speed`. Barely consumed
  in the unified path (velocity/speeds fields are effectively write-only — see A9), so P3.
- **Fix applied (user-designed):** `last_observation` initialized to
  `cv::Mat(1,4,CV_32F,cv::Scalar(-1.0f))` at **both** init sites (header default + ctor body —
  changing only one is a landmine), and `generate_final_results`' "has real observation" check
  changed from `countNonZero > 0` (which would output the −1 sentinel as a box) to
  `sum >= 0` — same sentinel convention as OC-SORT. Consumer sweep confirmed nothing else
  reads the sentinel: `info.last_boxes`/`k_observations` are write-only, the
  `update_secondary_features` width gate skips it identically, `try_deferred_face_register`
  is streak-gated, the in-update fallback only runs inside the `sum≥0` branch.

### A7 (P3) — `delta_t` config silently ignored
- `KalmanBoxTracker` ctor hardcodes `delta_t = 1` regardless of `LightTrackerConfig::delta_t=3`.
  Fine for PTZ (comment says intentional) — but then remove the config field or honor it.

### A8 (P3) — comment/code drift
- `collect_nearby_dets`: comments say 2×/4× diag; code uses 4.0×/6.0× (tuning updated,
  comments not). `face_recognition_match` still carries dead "seed" logic and commented-out
  emb resets. Misleading during future tuning.

### A10 (P0) — `get_kps10` unguarded indexing on empty keypoints — **FIXED** (reboot-class)
- **Where**: `LightTracker.cpp` `get_kps10` (indices 37/87/86/65/61 into `points`); call sites
  `face_recognition_inference`, `setMainTarget`, `try_deferred_face_register`.
- **Mechanism**: every failure path of `CFaceKeypoint106::run` (not initialized, preprocess
  fail, **NPU inference fail**, postprocess fail) returns **empty `points`**;
  `std::vector::operator[]` on an empty vector is UB — `data()` is typically `nullptr` → a
  near-null read → **SIGSEGV**. A segfault is not a C++ exception, so `fx_tracker_run`'s
  `try/catch` cannot stop it: the process dies, and under a device watchdog that becomes a
  **machine reboot**. The face path runs many times per second (periodic verification, danger
  period every 3 frames, full-frame sweep), so a single Facekps/ACL hiccup was fatal.
- **Fix applied**: size guard inside `get_kps10` (`< 106 → return empty`) + explicit
  `kps_10.size() < 10` handling at all three call sites (skip face / skip registration —
  deferred registration re-tries later). Found by the reboot investigation's crash-class scan;
  this was the **only** segfault-class defect in LightTracker.cpp (all other raw accesses are
  bounds-guarded, exception-class — caught at the C API — or inside disabled GMC code).

### A11 (P0) — detector decode ignores real buffer size; init() swallows model-load failures — **HARDENED** (crash-loop investigation)
- **Context**: on-device symptom = `onvif_svr` dies within seconds of every launch (`err pid -1`,
  `killall … no process killed` — dead, not hung), watchdog restart loop escalating to machine
  reboot. Pre-designation, almost no new tracking logic executes → fault domain is .so load /
  init / first-frame detector path.
- **Defect 1**: `ModelProcess::OutputModelResultDet` walks the raw NPU output with **hardcoded**
  `10647×7` floats (298,116 bytes) while the true buffer size `len` was fetched and **ignored**.
  Any smaller-than-assumed `.om` output (FP16 dtype, different input resolution, graph-fused
  NMS) → >100 KB OOB read **every frame from frame 1** → SIGSEGV → exactly this crash loop.
  Also `dim_anchor=7` ⇒ 2 classes, contradicting the documented 3-class model (needs 8) — if
  heads were detected by the previously-working binary, that binary was not built from this
  file (device-tree divergence evidence). **Fix**: null/len guards + clamp `num_anchor` to
  `len/(dim*4)` with a loud `ERROR_LOG` when clamped — zero behavior change when the model
  matches the hardcode. Proper dims (7 vs 8) still need the real `.om` output shape.
- **Defect 2**: `LightTracker::init()` checked **no** `loadModel` results — a missing/corrupt
  `.om` under `/oem/model` silently produced uninitialized components that fault at first
  inference (a bad deploy becomes a mystery crash). **Fix**: explicit check of all five model
  buffers → `ERROR_LOG` with per-model sizes + `return -1` (surfaces as `FX_ERR_INIT_FAILED`).

### A12 (P0) — full-file reboot-class sweep #2: init UB + unguarded empty-embedding arithmetic — **FIXED** (crash-loop investigation, round 2)
- **Context**: second whole-file audit of `LightTracker.cpp` (user request: "logic error or any
  other cause of machine reboot"), extended to every function it calls per frame. Sweep classes:
  segfault (OOB/empty-Mat/dangling), SIGFPE (int div/mod 0), fatal-on-legacy-interface
  `cv::Exception` (the old `Track` shim has **no** try/catch — any OpenCV throw = process death
  = watchdog reboot; only the new `fx_tracker` API catches), hangs (unbounded loops), OOM
  (unbounded containers). Verified clean: all det/head Mats row-aligned by construction;
  `matchPersonFaces` face box/score lockstep; B9 greedy `for(;;)` provably terminating (each
  iteration matches or bans one pair from a finite set); `KalmanBoxTracker::observations`
  pruned at 10, anchor gallery capped at 3, `quality_history_` pop_front-bounded,
  `standalone_heads_` cleared per frame → no growth; `velocity`/`last_observation` always
  non-empty (ctor + header defaults); all `best.index`/`face_idx`/pose-keypoint (fixed
  `[17]` array) accesses bounds-guarded; every division epsilon- or clamp-guarded; GMC fully
  dead behind `gmc_enabled_=false`. Four real defects found, all fixed:
- **Defect 1 — `FaceRecognitionSystem::init` fell off the end of a non-void function** (UB,
  once per launch, on the exact startup path): success route ended with two assignments and no
  `return`. At aarch64 `-O2` the compiler may omit the epilogue or emit a trap for that path
  (and the garbage return value would poison any caller check). **Fix**: `return 0;`.
- **Defect 2 — `LightTracker::init()` discarded all five component `init()` return codes**
  (they all return int, 0 = success): ACL/NPU init failure (driver not up at boot; device not
  released after the previous crash — the classic crash-*loop* amplifier) proceeded silently to
  a first-inference fault. **Fix**: capture `rc_det/rc_reid/rc_fk/rc_fr/rc_pose`, any nonzero →
  `ERROR_LOG` with all five codes + `return -1` (→ `FX_ERR_INIT_FAILED`). Complements A11
  defect 2 (which only covered host-side model-file loading).
- **Defect 3 — unguarded `cand_feature.dot(match_emb)`** (match loop, per candidate): either
  side can legitimately be empty (per-candidate ReID NPU failure; or main emb empty since a
  failed designation-time embedding) → `cv::Mat::dot` size-assert throws → fatal on the legacy
  interface, every frame. **Fix**: empty either side → `reid_sim = 0` (candidate then falls to
  the anchor gate — 宁可漏配不崩溃).
- **Defect 4 — `KalmanBoxTracker::update_emb` EMA on empty `new_emb`**: the
  `update_trackers_unified` fallback path passes `compute_embedding(...)` output unchecked; an
  empty sample enters `(1-α)*emb + α*new_emb` → OpenCV MatExpr throw. **Fix**: guard at the
  sink (`new_emb.empty() → return`), covering all present and future callers.
- Also removed the dead `extern unsigned char DEBUG_LOG;` in FaceRecognitionSystem.cpp — never
  used, but it declared a **different type** than the legacy definition (`unsigned short`, in
  fx_wrapper_track.h), and any future use would produce an undefined dynamic symbol in the
  new-API .so (load-time death) or type-punned UB in the legacy build.
- **Residual (not code)**: if the crash loop persists after A11+A12, the remaining domains are
  outside this repo — app-side (frame buffer/stride mismatch into `update()`), ABI drift
  between repo headers and the device build tree, or driver/OOM (discriminate via the §5
  runbook: manual run + dmesg, `ldd`, model dir check, old-.so bisect).

### A9 (P3) — dead code inventory
- `Association.cpp` (entire file — Hungarian/ByteTrack-style association) unreferenced.
- `compute_candidate_pose_score` (superseded by `compute_candidate_pose_detail`).
- `TrackerInfo.k_observations` (write-only, and filled with a duplicate of
  `last_observation` — not the k-th-back observation its name implies), `TrackerInfo.speeds`
  (write-only).
- Commented-out xywh `iou_single` copy in utils.cpp (:219) — delete to prevent resurrection.
- Legacy `Track.cpp`/`fx_wrapper_track.*` — kept intentionally for the old app (per README).

---

## 2. Mechanism-level vulnerabilities (机制性风险 — abnormal matching by design)

These are not bugs; they are places where the *design* can be induced to match abnormally.
Ordered by residual risk.

### B1 — Face hard-override is a single point of failure that self-reinforces
- **Chain**: `face_recognition_inference` gate is a **0.55 cosine** (FaceRecognitionSystem);
  the full-frame sweep accepts faces down to **24 px**; on a hit the override sets
  `best.matched` **ignoring every spatial/motion/veto signal**; then trust amplifies:
  emb refreshed at α=0.4 (`kFaceConfirmEmbAlpha`), anchor gallery gains a sample at
  **q = 0.6–1.0**, face lock (TTL 60 frames) tightens identity gates *around the new lock*.
- **Failure**: one false-positive face (look-alike, small face, motion blur) → instant
  teleport of the lock + template/gallery poisoning. A q≈1.0 impostor gallery sample is
  **immortal** — eviction requires a *higher*-quality sample, and 1.0 can't be beaten.
- **Existing mitigations**: sweep gated by doubt (`anchor < 0.70`), `kFaceSweepMinFacePx`,
  near-first design, `try_deferred_face_register`'s identity gate (registration side is now
  well protected — the *recognition* side is not).
- **Hardening (recommended order)**:
  1. require margin: accept only if best face sim exceeds runner-up by δ (e.g. 0.08);
  2. route **sweep** hits through the existing provisional-commit gate (2-frame same-hypothesis)
     instead of instant override — reuses shipped machinery;
  3. cap face-driven gallery quality at 0.9 so a later genuine confirmation can still evict;
  4. spatial plausibility check for near-field face overrides unless `long_blind`.

### B2 — Occlusion FSM onset blind spots
- (a) Onset requires `overlap_count ≥ 2` — i.e. main's own detection + occluder. **Sneak
  occlusion** (main lost *first*, then someone covers the area) keeps `overlap=1` → FSM stays
  CLEAR → body-IoU stays fully trusted while the occluder stands on the frozen KF box. The
  exit side got this guard (`main_recently_seen`, F7); the onset side did not.
- (b) During long coast, **two strangers** crossing the stale KF box trigger a *spurious*
  OCCLUDED: wrong `occluder_tracker_id_`, danger-mode signal storm (full-candidate ReID,
  per-frame pose).
- (c) The zero-detection early-return path (`update():769-834`) never advances the FSM at all
  — state freezes through detector blackouts.
- **Hardening**: symmetric onset guard (`main_tsu ≤ 1` required to *enter* OCCLUDED), and a
  freshness cap so overlap with a box that hasn't been observed for k frames doesn't count.

### B3 — Anti-shadow guard starves occluder identification
- `kSecTrkNewMainIou = 0.5` refuses to create a secondary track for any detection overlapping
  the main — which is **exactly the approaching occluder**. Result: at occlusion onset the
  occluder often has no track → `occluder_tracker_id_ = -1`, `anti_occ` inert, coexist-veto
  blind for the one person who matters most. The two mechanisms fight each other.
- **Hardening**: create the track but mark it `quarantined` (excluded from coexist-veto until
  it survives N frames *not* overlapping the main) — keeps the anti-shadow intent while
  letting occluder machinery see the occluder.

### B4 — CLEAR-mode ReID prerank is IoU-only against a deliberately sluggish KF
- Body KF is tuned near-static (Q_pos = 0.008, vel_decay = 0.8) on the assumption the gimbal
  re-centers the target; with **GMC off** and the gimbal holding during uncertainty, a
  genuinely moving target's prediction **lags**. Prerank (`kReidMaxCandClear = 2`, IoU-only)
  can then rank a stationary bystander above the true target → true target gets
  `reid/anchor = 0` → **anchor hard-veto kills it** → switch or loss. This is the
  fast-walk-through-crowd switch signature.
- **Hardening**: blend lead-center distance into the prerank key (lead already tracks motion
  intent); or `kReidMaxCandClear` adaptive to `close_det_count`; longer-term, motion-adaptive
  process noise (raise Q when `MainTargetPredictor` reports sustained velocity).

### B5 — Quality-monitor baseline never refreshes
- `baseline_reid_/baseline_anchor_sim_` are built from the **first 10 stable frames** and then
  frozen for the session. After legitimate drift (gallery grew, distance/lighting changed),
  `anchor_drop`/`trend_decline` fire against an obsolete reference — alerts too eager or too
  blind. **Hardening**: slow EMA re-baseline during high-confidence stretches (face-locked or
  anchor ≥ 0.75, CLEAR, isolated).

### B6 — Frame-count TTLs behave inconsistently under variable FPS
- Measured frame gaps ranged 27→374 ms this month. `kFaceLockTTL=60`, `kAlertTimeout=90`,
  `kMaxOcclusionFrames=90`, `kCoexistVetoMaxTsu=2`, face/anchor periods, `kSecEmbRefreshFrames`
  are all **frames** — wall-clock semantics stretch ~14× on slow frames (a "2.4 s" face lock
  becomes ~22 s). The codebase's own convention says timestamp-driven (`frame_dt_sec_`,
  `kOrientHoldMaxMs` already does this). **Hardening**: convert the *identity-critical* TTLs
  (face lock, alert timeout, occlusion timeout, coexist freshness) to ms.

### B7 — Head continuity carries zero identity evidence (strongest remaining vector)
- Self-admitted in comments: `try_head_continuity` snaps to the nearest standalone head near
  the head-KF prediction. If the main truly left and a **lone bystander head** drifts into the
  gate: body is reconstructed from the bystander's head, **output as main at weight 0.7**, and
  — critically — `main_target_predictor.on_matched()` is called, which **resets `miss_count`**,
  so the C-identity `long_blind` review **never arms**. The wrong output can sustain
  indefinitely; the gimbal follows a stranger at 0.7 confidence.
- **Existing mitigations**: shrinking gate with staleness, two-head ambiguity rejection,
  `kHeadReacqMinScore`.
- **Hardening**: (1) **ms-bounded head-only budget** (like `kOrientHoldMaxMs`) — after T ms of
  head-only sustain without a body match, degrade to blind coast; (2) head-size consistency
  vs learned `hb_*` ratios; (3) on body-reappearance near a reconstructed box, require an
  identity spot-check (one ReID against anchor) before treating it as continuation; (4) count
  head-only frames as "soft-blind" for C-identity instead of resetting miss_count. This also
  ties into the known deferred issue that head-continuity re-anchoring moves the coast.

### B8 — `BOX_COMPLETE` can inflate the main box over the occluder
- In-danger completion extends y2 to `main_h_hold_` when the missing lower body overlaps
  another detection — but that overlapping detection **is the occluder**, i.e. the trigger
  condition is satisfied precisely when extension paints the main box over another person; the
  inflated box then feeds the KF (h grows) and IoU gates. `main_h_hold_` is also frozen
  pre-danger (stale if the target sat down / receded).
- **Hardening**: cap extension (e.g. ≤ 25% of h_obs), decay `main_h_hold_` slowly during
  danger, and never extend when the gap-overlapper is the current best-candidate runner-up.

### B9 — Secondary-track identity churn migrates negative evidence
- Secondary association is greedy-IoU only; when two secondaries swap ids, their
  `coexist_with_main`, `emb`, and `color_hist_` blend across people → the arm-B exclusion
  compares candidates against a chimera. Bounded by the sim-floor + relative-margin + pollution
  guards, but it lowers veto precision in exactly the crowded scenes it exists for.
- **Hardening (cheap)**: when both have embeddings, require `emb·emb ≥ 0.5` to accept an IoU
  swap between two live secondaries; else prefer the unmatched-track path.

### B10 — New-view gallery admission × orientation gate (new feature interplay)
- The shoulder-orientation gate keeps turned-away frames matched more often (by design), so
  the "new-view" admission (anchor ∈ [0.33, 0.70), q=0.55, CLEAR+isolated+streak≥15) will add
  back-view samples more readily. Intended (fixes turn-loss deadlock), but since
  `anchor_sim = max(gallery)`, every added view widens the identity gate. **Watch on device**:
  log gallery size/qualities; if same-clothing false accepts rise, raise `kViewAddMinStreak`
  or require frontalness recovery between adds.

### B11 — `matchPersonFaces` smallest-area containment (low)
- A face fully inside two overlapping person boxes goes to the **smaller** (usually nearer)
  person. The main behind loses its face that frame → periodic verification misses. Benign
  (recognition still gates identity), just reduces face availability in overlap scenes.
  Keypoint-based assignment (pose nose→face box) would resolve it if ever needed.

---

## 3. Performance findings (性能)

- **C1 — danger-mode NPU stack is the frame-spike path**: in danger, `do_reid` covers all
  near candidates (≤5) → up to 5 ReID (~6 ms each) + pose every frame + face (2 NPU/face,
  every `kFaceDangerEveryN=3`). Worst frames stack det+pose+5×reid+2×face. Add
  `kReidMaxCandDanger≈3` (danger prerank by IoU+head_match instead of IoU alone, so the cap
  is safe).
- **C2 — winner's embedding recomputed up to 3×/frame**: candidate loop already stores the
  winner's feature in `c.emb`, yet anchor-add (both blocks), `update_trackers_unified`'s
  emb-update, and face-confirm each call `compute_embedding` again on (usually) the same box.
  Thread the winner's `emb` through `MainMatchResult` and reuse → saves ~6–12 ms exactly on
  the heavy frames. (Face-override/sweep hits may lack `c.emb` — keep the recompute fallback.)
- **C3 — logging**: A4's per-frame cout; `[MATCH]` cout is intentional but is UART time —
  consider buffering/rate-limiting for production builds.

---

## 4. Optimization roadmap (优先级路线图)

**Execution status (2026-07, same session as the report):** items 2–12 and the
15-label-fix are **IMPLEMENTED** (grep-verified, NOT compile-verified — device build is the
gate). Still open: item 13's B5 (baseline refresh), item 14 (dead-code removal — device host
only), and the 15-damping (partially mitigated by item 6's head-only budget).

| # | Item | Findings | Effort | Risk | Status / note |
|---|------|----------|--------|------|------|
| 1 | **Device compile + field validation** | D | build only | — | **OPEN — gates all of the below** |
| 2 | Color-hist xyxy fix | A1 | done | none | ✅ done |
| 3 | Zero-init `get_predicted_tracks` + NaN log | A2 | XS | none | ✅ done (rows pre-zeroed; `[KF_NAN]` log) |
| 4 | Registration face selection | A3 | S | low | ✅ done (head-box/upper-band consistent, best-score; comparable 2nd face ⇒ defer registration) |
| 5 | Face hardening | B1 | M | medium | ✅ done: cross-candidate sim margin `kFaceSimMargin=0.08` (`[FACE_MARGIN]`); sweep hits routed through provisional gate (`kFaceSweepConfirmFrames=2`, per-frame re-sweep while pending + staleness stop); trust side-effects (lock/emb/gallery/faceVerified/alert-clear) deferred to near-field confirm; face gallery q capped 0.9 |
| 6 | Head-continuity guards | B7 | M | medium | ✅ done: head-size ratio gate `[0.55,1.8]`; per-frame distance/ambiguity/body-ownership/coexistence guards; no fixed head-only timeout while real head observations keep passing; soft-blind arm `kHeadOnlySuspectMs=1000` protects discontinuous body reacquisition |
| 7 | FSM onset guard | B2 | S | low | ✅ done: onset requires `main_tsu ≤ kOcclusionOnsetMaxTsu=2` (symmetric to F7 exit guard) — kills spurious OCCLUDED from strangers crossing a stale coast box. Zero-det-path FSM freeze documented, not changed |
| 8 | Occluder quarantine tracks | B3 | M | medium | ✅ done: anti-shadow `continue` → create with `quarantined_` (`[QUARANTINE] new/cleared`); quarantined = visible to occluder-id/anti_occ but zero veto power (no coexist accrual, skipped as veto source, no output, no ReID budget); cleared by 5 consecutive own-detections at IoU<0.30 vs main — a shadow twin structurally can't |
| 9 | Prerank + danger cap | B4, C1 | S | low | ✅ done: prerank key = IoU + `kPrerankDistW`·lead-distance score; CLEAR K+1 when `close≥3`; danger capped `kReidMaxCandDanger=3` with head-match in the key |
| 10 | Reuse winner emb | C2 | S | low | ✅ done: `MainMatchResult.emb` threaded; anchor-add ×2 / emb-update reuse with recompute fallback; face override invalidates then repopulates |
| 11 | ms-based identity TTLs | B6 | S | low | ✅ done: face lock 2400ms, alert 3600ms, occlusion timeout 3600ms; coexist freshness = frames AND ≤500ms wall-clock (KBT `last_update_ms_` stamp) |
| 12 | id-recycle + per-frame cout | A5, A4 | XS | none | ✅ done (`set_count(max_live_id+1)`; frame-distance log under kMatchTrace) |
| 13 | Baseline EMA refresh; BOX_COMPLETE cap; secondary emb-gated swap | B5, B8, B9 | M | low | B8 ✅ done (aspect cap ≤3.4×w_obs — doubles as scale adaptation; hold freshness ≤5 s via `main_h_hold_ms_`; runner-up evidence guard vs near-tie candidates); B9 ✅ done as **color-gated** association (dets carry no emb — per-pair ban when `color_hist_sim<0.25`, `[SEC_SWAP]`; rejected pairs stay unmatched, 宁可漏配不错配; same-clothing swaps remain — documented limit); **B5 still OPEN** |
| 14 | Dead-code removal | A9 | S | none | OPEN — deliberately deferred: without a local compiler, deleting files/fields risks breaking the legacy Track build; do on the device host |
| 15 | MTP "(lead)" label + head-continuity re-anchor damping | D | S | low | label fix ✅ done (`is_static` true when vel scales are 0); damping OPEN (partially mitigated by item 6's budget) |

---

## 5. On-device verification protocol (验证)

1. **Compile** per `c_api/README.md` (aarch64, `-std=c++17`) — neither new feature nor this
   report's A1 fix has been compile-verified locally (no OpenCV4/ACL on the dev box).
2. **Log checks**: `[SEC_EXCL]` now prints meaningful `color=` (same person ≈0.6–0.9 rather
   than noise); `[ORIENT]/[ORIENT_HOLD]` engage only when turned; `[MATCH]` per-frame summary
   is the anchor for all scenario traces. New tags from the roadmap execution:
   `[KF_NAN]` (A2, should never appear), `[FACE_MARGIN]` (B1a rejection — verify it never fires
   on a solo genuine face), `[PROVISIONAL] … sweep_face=1` (B1b — sweep hit deferring then
   committing at 2 same-hyp frames), `[HEAD_RECOVERY]`/`[HEAD_GATE]` (B7 — per-frame
   acceptance, ownership rejection, or strong-ReID bypass; there is no fixed-duration stop),
   `[FACE_RECOVERY]` (accepted/rejected face-only identity), and
   `[FACE_REG] ambiguous faces at designation` (A3). Build with
   `-DFX_TRACKER_MATCH_TRACE=1` for these traces; leave `NullSink::verbose=false`.
   The trace is written continuously to `/tmp/fx_tracker_match_trace.log` (64KB stdio
   buffer, event/25-frame flush, no `fsync`, 16MB cap). Override the destination with
   `FX_TRACKER_MATCH_TRACE_FILE`; copy the file before restarting because a new process
   truncates it on its first trace write.
   Danger-frame main-candidate ReID count must remain bounded by
   `kReidMaxCandDanger(4)` plus the separately budgeted secondary refresh (C1). From the B3/B8/B9 batch:
   `[QUARANTINE] new/cleared` (occluder-approach clip must show `occluder_id >= 0` at
   `[STATE] CLEAR -> OCCLUDED`, was −1; quarantined ids must never appear in
   `[COEXIST_VETO]`/`[SEC_EXCL]` or output rows), `[SEC_SWAP]` (crossing different-clothing
   pair: ids must not swap), `[BOX_COMPLETE]` (y2 capped at 3.4×w_obs; stops ≥5 s after the
   last CLEAR full-height refresh). From A11/A12 (crash-loop hardening, stderr `ERROR_LOG`):
   `model load failed under /oem/model: …` (missing/unreadable .om, per-model sizes),
   `component init failed: det=… reid=… faceKps=… faceReco=… pose=…` (ACL/NPU init failure —
   if this appears in a restart loop, the NPU driver/device is the problem, not the tracker),
   `det output smaller than expected … clamp anchors` (deployed .om output ≠ hardcoded decode
   dims). A clean start prints none of these.
3. **Scenario clips** (each maps to a finding):
   - turn-with-bystander (shoulder gate holds lock; no `[SEC_EXCL]` on the true target);
   - sneak occlusion — main lost first, then covered (B2a: watch FSM stay CLEAR and whether
     IoU steals);
   - long coast with two people crossing the stale box (B2b: spurious OCCLUDED?);
   - bystander-head drift during furniture occlusion (B7: does head-continuity glue to it,
     and does `[COAST]`/output weight 0.7 persist?);
   - designation with two faces in the box (A3);
   - danger-frame timing capture (C1: per-stage profiler already exists — `fxprof`).
4. **Regression**: frontal/normal clips — orientation gate and exclusion arm must be no-ops
   (`[ORIENT]` absent, vetoes silent), frame time unchanged.

---

## 6. Cross-references

- Related design docs in memory: multi-target appearance exclusion (Jul 2026), shoulder
  orientation gate (Jul 2026), GMC-disabled bindings, provisional-commit gate, association
  audit (Jul 2026).
- Both 2026-07 features and this report's fix await device compile (§5.1).
