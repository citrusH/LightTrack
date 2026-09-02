# Parameter Tuning Guide — PTZ single-target tracker

All tunable constants live in `LightTracker.h` (`static constexpr …`) plus the
`MainTargetPredictorConfig` struct in `MainTargetPredictor.h`. This guide groups them
by subsystem and, for each, gives the purpose and **when to raise / lower** it.

Guiding principle for this system (per the design + operator intent):
**prefer no match / hold over a wrong match — "better to stay still than move erratically."**
So when in doubt, tune toward *stricter identity gates* and *more hysteresis*, accepting
a bit more coast, rather than toward looser matching.

Two orthogonal notions of "how strict":
- **Raise a veto/gate threshold ⇒ stricter ⇒ fewer (and safer) matches ⇒ more coast.**
- **Raise a hysteresis/stickiness value ⇒ steadier lock ⇒ less oscillation, but slower to
  correct a genuine mistake.**

Timing note: speeds are **px/sec** and TTLs are in **ms** (timestamp-driven), so these are
FPS-independent. Frame-count constants (`*Frames`, `*EveryN`, `max_age`) implicitly assume
~25 fps.

---

## 1. Core tracker / SORT config (`Config` struct, `LightTracker.h:47-55`)

| Param | Val | Purpose | ↑ Increase when | ↓ Decrease when |
|---|---|---|---|---|
| `det_thresh` | 0.7 | Detector confidence floor to accept a box | Too many false detections/ghost tracks | Target missed at distance / low light (its box scores low) |
| `appearance_thresh` | 0.7 | ReID cosine floor for appearance match | ID switches to look-alikes | True target rejected after appearance change (lighting/pose) |
| `max_age` | 30 | Frames a **non-main** track survives unmatched before deletion | Tracks flicker in/out, losing coexist-veto identity | Stale ghost tracks linger and mis-veto (memory says keep bounded) |
| `min_hits` | 3 | Consecutive hits before a track is "confirmed" | Spurious short tracks pollute | Slow to trust a new real person |
| `iou_threshold` | 0.4 | IoU gate for detection↔track association | Wrong boxes associate across people | Fast motion / big frame gaps break association |
| `delta_t` | 3 | OCR-KF velocity-estimation lookback (frames) | Velocity too jumpy | Velocity too laggy on maneuvers |
| `inertia` | 0.2 | OCR direction-consistency weight | Direction noise causes bad assoc | Over-penalizing legitimate turns |
| `gate` | 0.2 | Association gating distance | — | Fast targets fall outside the gate |

---

## 2. Compute throttling (NPU budget — perf vs. accuracy)

| Param | Val | Purpose | ↑ Increase when | ↓ Decrease when |
|---|---|---|---|---|
| `kReidMaxCandClear` | 2 | Max candidates ReID'd per frame in CLEAR | Crowded scenes mis-rank the true target out of the ReID set | NPU frame-time spikes |
| `kReidMaxCandDanger` | 3 | Same, in danger/occlusion | Occlusion crowds need more identity checks | NPU too slow in danger |
| `kPoseInferEveryN` | 3 | Run pose every N frames (CLEAR) | Need fresher pose/shape | Pose is a frame-time hog |
| `kFaceMaxCand` | 3 | Max faces verified per pass | Missing face confirmations in crowds | Face NPU cost too high |
| `kFaceDangerEveryN` | 3 | Face cadence during danger | Want faster face re-confirm in danger | Perf |
| `face_recognition_every_n_frames` | 15 | Normal face cadence | Identity drifts between face checks | Perf |
| `kSecReidBudgetPerFrame` | 1 | Max non-main tracks ReID'd per frame | Secondary-track identities go stale (weaker veto) | Perf (hard cap) |
| `kSecEmbRefreshFrames` | 30 | Age (frames) after which a non-main embedding is "stale" | — | Others' appearance changes fast |
| `kPrerankDistW` | 0.5 | Weight of lead-center distance in the ReID-candidate pre-rank | Fast movers get pre-ranked out (KF lags with GMC off) | A nearby stranger keeps entering the ReID set |
| `kPrerankHeadW` | 0.5 | Weight of head-match in the pre-rank (danger) | Head is the reliable cue in occlusion but true head loses the slot | Head cue noisy |

> These trade frame-time for identity robustness; watch the `[MATCH]`/profiler frame time.

---

## 3. Lost-target output policy

`MainTargetPredictor` is not wired into `LightTracker`. After a stable BODY track loses
reliable measurement, `try_short_prediction()` may freeze the last reliable BODY KF
posterior velocity and emit a constant-velocity PTZ-only row for a bounded time and
distance. BODY/HEAD/FACE observations use weight `1.0`; short prediction uses a decaying
`(0,1)` weight; rejected/expired prediction emits no main row and weight `0.0`.
Prediction never updates identity features, the real-observation timer, or a Kalman
measurement.

| Active constant (`LightTracker.h`) | Value | Purpose | Too large | Too small |
|---|---:|---|---|---|
| `kShortPredictionMaxDurationMs` | 400 ms | Maximum PTZ-only output window | More wrong motion after stop/turn | Occlusion brake remains visible |
| `kShortPredictionNominalDtMs` | 40 ms | Unit conversion for frozen KF velocity (`px/40ms`); not a tuning knob | Wrong velocity scale | Wrong velocity scale |
| `kShortPredictionMaxBodyDiag` | 0.75 | Relative displacement cap | Small/near body can move too far | Fast target stops early |
| `kShortPredictionMaxFrameDiag` | 0.12 | Absolute frame-relative cap | PTZ may move too far in one blind window | Large/far target stops early |

Do not tune these from a single live run. Replay short occlusion, stop/turn behind the
occluder, target exit, and PTZ-motion cases, and compare wrong-motion duration against
the original no-output brake.

### 3b. Inactive `MainTargetPredictorConfig` reference (`MainTargetPredictor.h`)

> These parameters currently have no effect because the component is disconnected from
> `LightTracker`; re-enabling GMC alone does not reactivate it. Keep this table only as a
> reference if predictor output is deliberately wired back in later.

| Param | Val | Purpose | ↑ Increase when | ↓ Decrease when |
|---|---|---|---|---|
| `center_vel_scale` | 0.65 (→0 GMC-off) | Damping on extrapolated center velocity | Coast lags a genuinely moving target | Coast overshoots / moves erratically (operator's complaint) |
| `size_vel_scale` | 0.3 (→0 GMC-off) | Damping on box-size extrapolation | Box lags scale changes | Box grows/shrinks unrealistically |
| `max_center_vel_px_sec` | 180 | Hard cap on coast center speed | Legit fast targets clipped | Coast jumps too far |
| `max_size_vel_px_sec` | 150 | Hard cap on coast size-rate | — | Size predictions jumpy |
| `max_displacement_ratio` | 0.25 | Max center jump (× image width) before falling back to static | — | Single-frame teleports look wrong |
| `max_extrapolation_sec` | 1.2 | Max time to extrapolate before holding at endpoint | Longer smooth bridging needed | Long coasts drift far from reality |
| `warm_vel_max_age_sec` | 0.7 | How long a cached regression velocity stays reusable after re-match | Recovery dead-zone (stalls right after re-lock) | Warm velocity outlives its validity |
| `window_size` / `min_obs_for_velocity` | 5 / 5 | Regression window; obs needed before trusting velocity | Velocity too noisy | Too slow to arm velocity after re-lock |
| `weight_decay` | 0.9 | Per-frame coast confidence decay | Want coast to persist longer | Want faster hand-off to "no box" |
| `min_size_scale`/`max_size_scale` | 0.5 / 2.0 | Allowed size drift vs. last obs | — | Reject implausible size changes harder |
| `zoom_in/out_ratio_thresh` | 1.5 / 0.67 | Area change flagged as zoom | — | More sensitive zoom detection |

---

## 4. Occlusion state machine (`LightTracker.h:198-205, 234-236`)

| Param | Val | Purpose | ↑ Increase when | ↓ Decrease when |
|---|---|---|---|---|
| `kMaxOcclusionFrames` / `kMaxOcclusionMs` | 90 / 3600ms | Force OCCLUDED→CLEAR after timeout | Real occlusions are longer than this | Stuck in OCCLUDED too long |
| `kRecoveryFrames` | 30 | Protected RECOVERING window after separation | ID switches right after separation | Recovery caution lingers, slow to resume normal |
| `kSeparationConfirmFrames` | 2 (F7) | Clean frames needed OCCLUDED→RECOVERING | Single noisy frame triggers false "separated" | Too slow to recognize real separation |
| `kOcclusionOnsetMaxTsu` | 2 | Freshness required to *enter* OCCLUDED | — | Enters occlusion on stale data |
| `kAlertTimeoutMs` (`kAlertTimeout` 90f) | 3600ms | Auto-clear ID-switch alert | Alerts clear before recovery completes | Alerts stick too long, freezing adaptation |

---

## 5. Visibility state machine (`LightTracker.h:221-224`)

| Param | Val | Purpose | ↑ Increase when | ↓ Decrease when |
|---|---|---|---|---|
| `kVisEmaAlpha` | 0.4 | Smoothing of the visible-ratio estimate | State reacts too slowly | State chatters between HALF/UPPER/etc. |
| `kVisHysteresisFrames` | 4 | Frames of stability before switching visibility state | Visibility state flickers | Slow to react to genuine occlusion onset |

---

## 6. Face lock / recognition / sweep

| Param | Val | Purpose | ↑ Increase when | ↓ Decrease when |
|---|---|---|---|---|
| `kFaceLockTTLMs` | 2400ms | How long a confirmed face overrides ReID/fusion | Same-clothing crowds need longer face authority | Face lock outlives its validity (person turned away) |
| `kFaceConfirmEmbAlpha` | 0.4 (F4) | EMA weight when a verified face refreshes the body template | Want faster template adaptation to new appearance | A single face false-positive poisons the template |
| `kFaceSimMargin` | 0.08 | Best face must beat 2nd-best by this | Similar faces cause wrong override | Legit face rejected as "ambiguous" |
| `kFaceSweepConfirmFrames` | 2 | Full-frame face hypothesis must persist N frames | Sweep grabs wrong person on one frame | Too slow to re-anchor via sweep |
| `kFaceSweepPeriod` | 15 | Full-frame sweep cadence (frames) | Faster re-acquisition after long loss | Perf |
| `kFaceSweepDoubtAnchor` | 0.70 | anchor ≥ this ⇒ identity certain ⇒ skip sweep | Sweeping too aggressively when already locked | Not sweeping when it should doubt the lock |
| `kFaceSweepMaxFaces` | 2 | Max faces sent to recognition per sweep | Missing the target's face in crowds | Perf |
| `kFaceSweepMinFacePx` | 14 | Min face height worth recognizing | Wasting NPU on tiny unrecognizable faces (raise) | Target's face is small/distant and skipped (lower) |
| `kFaceRegisterRetryInterval` | 15 | Retry face-register cadence when unregistered | — | Want faster initial registration |
| `kFaceReregisterInterval` | 300 | Re-register cadence once registered (quality upgrade) | — | Want fresher template |
| `kFaceTemplateGoodEnough` | 0.80 | Template quality that stops re-registration | Want higher-quality template always | Never stops re-registering |
| `kFaceTemplateUpgradeMargin` | 0.05 | Quality gain needed to replace template | Template churn | Never upgrades to a better face |

---

## 7. Head continuity & standalone-head reacquisition
*(the "no body track available" path — recently tightened, §G2)*

| Param | Val | Purpose | ↑ Increase when | ↓ Decrease when |
|---|---|---|---|---|
| `kHeadPredMaxAge` | 15 | Max head-KF prediction age (frames) usable for continuity | Longer occlusions need head bridging | Stale head pred drifts onto neighbors |
| `kHeadReacqGateRatio` | **2.5** | Adopt radius for a standalone head (× head size), then reduced as the head prediction ages | True target's head sits just outside the gate (missed) | **Adopts neighbor heads — drift (lower / keep low)** |
| `kHeadReacqMinScore` | **0.5** (was 0.4) | Min **detection confidence** to adopt a standalone head | True target's head over-rejected (drop toward 0.45) | **Half/low-conf heads get adopted — arbitrary match (raise)** |
| `kHeadReacqAmbigRatio` | **0.5** (was 0.6) | Nearest head must be ≤ this × runner-up distance (else ambiguous→reject) | Rejecting too often when two heads are close but one is clearly right | **Two heads too easily "resolved" onto the wrong one (lower)** |
| `kHeadSizeRatioMin/Max` | 0.55 / 1.8 | Standalone/pred head diagonal ratio band (depth check) | Rejecting the real head due to size noise (widen) | Adopting a foreground/background head of wrong depth (narrow) |
| `kHeadOnlySuspectMs` | 1000ms | After this, a body re-acq is treated as long-blind (arms C-identity gate) | — | Want the identity re-check armed sooner |
| `kObservedPartWeight` | 1.0 | PTZ control weight after a head/face observation has passed all identity/geometry gates | — | Do not tune for identity safety; tighten the acceptance gates instead |
| `kHeadMatchFalloff` | 2.5 | head_match score falloff (× head size; score→0 distance) | Tolerate larger head displacement per frame | Tighten head spatial discrimination |
| `kHeadMatchVetoMin` | 0.20 (F6) | Danger-period head hard-veto threshold | ID switches through occlusion (raise → stricter) | True target vetoed on large head motion (lower) |
| `kHeadVetoMaxAge` / `kHeadVetoIou` | 8 / 0.10 | Freshness & IoU gate enabling the head veto | — | — |
| `hb_h_ratio_`/`hb_w_ratio_`/`hb_dy_ratio_`/`hb_dx_ratio_` | 7.0/2.5/3.5/0.0 (learned) | Head→body reconstruction geometry (EMA-learned defaults) | — (auto-learned) | — |
| `kHbGeomAlpha` | 0.2 | EMA rate for the head↔body geometry learning | Geometry adapts too slowly | Geometry over-reacts to noisy frames |

---

## 8. Incumbent hysteresis — anti-oscillation (`LightTracker.h:394-396`, §G1)

| Param | Val | Purpose | ↑ Increase when | ↓ Decrease when |
|---|---|---|---|---|
| `kIncumbentHysteresis` | 0.07 | Bonus to `total` for the candidate continuous with last frame's real hit — stops left-right flip between look-alikes | **Still oscillates / flips between two similar people** | Sticks to a wrong lock too long after a real switch |
| `kIncumbentMinIou` | 0.30 | Min IoU with last observation to count as the incumbent | Incumbent lost on fast motion (no candidate qualifies) | A neighbor wrongly qualifies as incumbent |
| `kIncumbentMaxTsu` | 3 | Max frames-lost for the last observation to still anchor hysteresis | Want stickiness to survive slightly longer losses | Stale last-obs anchors hysteresis to the wrong place |

---

## 9. Appearance ambiguity & fusion arbitration (`LightTracker.h:399-403`)

| Param | Val | Purpose | ↑ Increase when | ↓ Decrease when |
|---|---|---|---|---|
| `kReidAmbiguousMin` | 0.55 | Both top-2 appearance scores above this ⇒ consider "ambiguous" (same-clothing) | Ambiguity mode misfires on low scores | Missing real same-clothing ambiguity |
| `kReidAmbiguousGap` | 0.08 | …and top1−top2 < this ⇒ ambiguous ⇒ cut ReID weight | Detecting ambiguity too rarely | Flagging clear cases as ambiguous |
| `kAmbiguousGapDanger` | 0.08 | Danger: reject match if winner−runner `total` gap < this (coast instead) | **Wrong-target commits in danger (raise → more coast)** | Over-coasting, refusing valid matches |
| `kAmbiguousGapClear` | 0.04 | CLEAR: same, looser | ID switches in normal tracking | Refusing valid matches when uncrowded |
| `kFaceLockVetoBoost` | 0.08 | Extra anchor-veto raise while face-locked + danger | Impostors slip through during face lock | True target rejected while face-locked |

---

## 10. Anchor gallery & identity veto (`LightTracker.h:414-440`)

| Param | Val | Purpose | ↑ Increase when | ↓ Decrease when |
|---|---|---|---|---|
| `kAnchorVetoRelaxed` | 0.28 | Relaxed anchor floor for spatially-continuous candidates | Look-alikes admitted via the relaxed path | Turning/receding true target vetoed |
| `kVetoRelaxIou` | 0.30 | Min IoU with KF box to qualify for the relaxed floor | Relaxed path too easy to enter | Genuine self not "continuous enough" |
| `kEmbAdaptAnchorFloor` | 0.35 | Min anchor to allow slow template drift | Template drifts toward an impostor | Template can't adapt to appearance change |
| `kAnchorAddAnchorMin` | 0.70 | Min anchor to admit a non-face view into the gallery | Gallery polluted by uncertain views | Gallery too sparse, poor view coverage |
| `kAnchorAddPeriod` | 15 | Min frames between non-face gallery adds | Gallery churns | Slow to accumulate views |
| `kViewAddMinStreak` | 15 | Consecutive-hit "certificate" before adding a view | Bad views enter the gallery | Rarely captures new legit views |
| `kViewAddQuality` | 0.55 | Quality floor for an added view | Low-quality gallery entries | Good views excluded |
| `kTeleportBaseDiag` / `kTeleportPerMissDiag` / `kTeleportMissCap` | 1.0 / 0.5 / 8 | Re-acq search budget (× box diag; grows per miss; capped) | Fast/teleporting target not re-found in the gate | Gate so wide it grabs a stranger |
| `kReacqProbationGap` | 12 | miss_count ≥ this ⇒ long-blind identity re-check armed | Long blind re-acqs grab wrong person (lower to arm sooner) | Re-check too aggressive on short losses |
| `kReacqAnchorConfident` / `kReacqReidConfident` | 0.60 / 0.75 | Appearance certainty to allow a pure-teleport direct accept | Teleport accepts an impostor (raise) | Legit teleport never accepted (lower) |
| `kReacqMaxDefer` (K) | 4 | Same-hypothesis frames required before committing a risky re-acq | **Separation poison / stranger theft on re-acq (raise 3→5)** | Re-acquisition feels sluggish (lower) |
| `kProvisionalPosTolFactor` | 1.5 | Center-jump tolerance for "same hypothesis" (× box diag) | Legit small jumps break the hypothesis chain | Different people counted as the same hypothesis |

---

## 11. Secondary tracks & coexistence exclusion (`LightTracker.h:448-470`)

| Param | Val | Purpose | ↑ Increase when | ↓ Decrease when |
|---|---|---|---|---|
| `kSecTrkMax` | 8 | Max non-main tracks kept (compute guard) | Losing useful "other-person" identities in crowds | Perf |
| `kSecTrkIou` | 0.30 | Track↔detection association IoU for secondaries | Secondary tracks fragment | Wrong dets absorbed into a track |
| `kSecTrkNewMainIou` | 0.50 | Don't spawn a new track overlapping the main box this much (anti-shadow) | Shadow tracks of the main target spawn | Legit near-main people never get a track |
| `kCoexistConfirm` | 10 | Coexistence frames to confirm a track is "another person" | False vetoes from not-yet-confirmed tracks | Slow to arm the coexist veto |
| `kCoexistVetoIou` | 0.50 | Candidate-on-confirmed-other IoU ⇒ exclude (same-clothing immune) | Impostors standing on a known-other track slip through | True target vetoed when it overlaps an old track |
| `kCoexistVetoMaxTsu` / `kCoexistVetoMaxMs` | 2 / 500ms | Freshness required for a track to hold veto power | — | Stale ghost tracks mis-veto the true target (keep low) |
| `kQuarantineClearIou` / `kQuarantineClearFrames` | 0.30 / 5 | When a suspected shadow track is rehabilitated | — | — |
| `kSecExclSimMin` | 0.65 | Candidate↔known-other similarity to qualify appearance exclusion | Weak appearance vetoes | Missing clear "this is that other person" cases |
| `kSecExclMargin` | 0.10 | (other-sim − main-anchor) needed to exclude (same-clothing guard) | Excluding the true target in same-clothing | Impostor not excluded |
| `kSecExclColorMin` | 0.4 | Color-histogram corroboration for exclusion | Color false-corroboration | Exclusion never corroborated |
| `kSecAssocColorMin` | 0.25 | Color gate for secondary association | — | — |
| `kSecColorHistAlpha` | 0.3 | Color histogram EMA rate | Slow color adaptation | Color over-reacts |
| `kSecPollutionSim` | 0.55 | If an "other" track looks this much like the main anchor ⇒ suspect a main-target split ⇒ don't let it veto | Main-target splits are mis-vetoing the real target | Real others wrongly disqualified as veto sources |

---

## 12. Motion-consistency veto (`LightTracker.h:317-318, 496-501`)
*(**disabled while GMC is off** — KF velocity includes camera pan; listed for completeness / GMC-on)*

| Param | Val | Purpose | ↑ Increase when | ↓ Decrease when |
|---|---|---|---|---|
| `kStationaryPxSec` | 40 | Below this ⇒ main considered stationary | Jitter reads as motion | Slow drift missed as "still" |
| `kCandMovingPxSec` | 75 | Above this ⇒ candidate/track "moving" | False motion vetoes | Missing fast impostors |
| `kMotionVetoAssocIou` | 0.30 | Candidate↔moving-track IoU to trigger veto | — | — |
| `kMotionDirCosVeto` | 0.30 | Direction-opposition cosine to veto (main moving) | Over-vetoing crossing paths | Missing opposite-direction impostors |
| `kMotionVetoFactor` / `kMotionVetoMinGapDiag` | 2.5 / 1.0 | Same-clothing spatial-outlier veto (relative distance × / abs gap in diags) | Spatial outliers not rejected | True target rejected as outlier |

---

## 13. Orientation / shoulder continuity (turn & lighting robustness, `LightTracker.h:474-487`)
*(No-op when the target faces the camera — `frontalness_` gating.)*

| Param | Val | Purpose | ↑ Increase when | ↓ Decrease when |
|---|---|---|---|---|
| `kFrontalLowThresh` | 0.55 | frontalness below this ⇒ appearance degraded ⇒ engage shoulder routing | Turn-handling engages too late | Engaging on near-frontal (over-trusting shoulders) |
| `kFrontalnessAlpha` | 0.4 | frontalness EMA rate | Reacts slowly to turns | Chatters on pose noise |
| `kOrientMinTorsoPx` | 60 | Min box height for orientation to be trusted | Trusting orientation on tiny boxes | Ignoring orientation on valid mid boxes |
| `kOrientReidMul` / `kOrientShapeMul` | 0.25 / 0.5 | ReID/shape weight floors when fully back-facing | ReID still over-trusted when back-turned | Under-using ReID during mild turns |
| `kShoulderContBudget` | 0.35 | Max fusion weight routed to shoulder continuity in turns | Turn lock still lost (give shoulders more) | Shoulders over-driving the match |
| `kShoulderContFalloff` | 2.0 | Shoulder-continuity score falloff (× shoulder width) | Tolerate larger shoulder displacement | Tighten shoulder spatial discrimination |
| `kShoulderContIncumbent` | 0.5 | Shoulder score ⇒ "geometrically-continuous self" (relaxes anchor) | Impostors pass the shoulder gate | True turning self not recognized |
| `kRelEngage` | 0.99 | appearance_rel below this ⇒ orientation gating on | — | — |
| `kAnchorVetoRelaxTurned` | 0.20 | Relaxed anchor floor for a turning self (below normal 0.28) | Impostor admitted during turns | Back-facing self vetoed frame-by-frame |
| `kOrientHoldMaxMs` | 1500ms | Max time geometry may "hold" (not commit) before falling back to strict/coast | Longer turns cut short | Geometry sustains a wrong hold too long |
| `kLightingPrevAnchorMin` | 0.5 | Prior anchor baseline needed to call a drop a "lighting" event | Lighting handling misfires | Missing real lighting drops |
| `kAnchorDropSuspect` | 0.25 | Anchor drop vs. last frame to suspect lighting | Over-attributing drops to lighting | Missing lighting-caused drops |
| `kLightingRel` | 0.3 | How far to push appearance_rel down when lighting suspected | — | — |

---

## 14. Box-completeness / occlusion evidence (`LightTracker.h:528-535`)

| Param | Val | Purpose | ↑ Increase when | ↓ Decrease when |
|---|---|---|---|---|
| `kBoxCompleteTrig` | 0.85 | Observed height < held×this ⇒ check for truncation | Missing partial occlusions | Firing on normal height variation |
| `kBoxCompleteAlpha` | 0.15 | Full-height EMA rate | Height baseline adapts slowly | Baseline over-reacts |
| `kBoxCompleteEvidIou` | 0.10 | Min IoU of the missing region with another box (occlusion evidence) | — | — |
| `kBoxCompleteMaxAspect` | 3.4 | Aspect cap for a "complete" box | — | — |
| `kBoxCompleteHoldMaxMs` | 5000ms | Max hold on a completed box | Cutting legit holds short | Holding a bad completion too long |

---

## 15. Quality monitor / ID-switch alert (`LightTracker.h:274-282`)

| Param | Val | Purpose | ↑ Increase when | ↓ Decrease when |
|---|---|---|---|---|
| `kQualityWindowSize` | 30 | Sliding window for reid/anchor baselines | Baselines noisy | Slow to notice quality changes |
| `kSuspectThresh` | 3 | Suspect frames before raising an ID-switch alert | Alert trigger too jumpy | Alert too slow to catch a switch |
| `kAlertTimeout` | 90f (`kAlertTimeoutMs` 3600ms) | Auto-clear alert | Alert clears before recovery | Alert freezes adaptation too long |

---

## 16. GMC (camera-motion compensation) — currently **disabled** (`gmc_enabled_=false`)

| Param | Val | Purpose | ↑ Increase when | ↓ Decrease when |
|---|---|---|---|---|
| `kGmcWorkWidth` | 640 | Down-sampled width for optical-flow/RANSAC | Need finer motion estimate | Perf |
| `kGmcMinInliers` | 12 | RANSAC inlier floor (else give up → identity) | Accepting bad affine fits | Rejecting valid fits (falls back to no-comp too often) |
| `kGmcMaxScaleDev` | 0.5 | \|scale−1\| cap flagged as a bad transform | — | Reject wild transforms harder |

> While GMC is off, motion-veto (§12), lead-center warp, and the predictor velocity
> knobs (§3b) are inert. Re-enabling GMC brings all of them back into play — retune
> §12 and §3b together at that point.

---

## Quick "it's doing X, change Y" index

| Symptom | First knobs to try |
|---|---|
| **Left-right oscillation between two similar people** | ↑ `kIncumbentHysteresis` (§8); ↑ `kAmbiguousGapDanger`/`kAmbiguousGapClear` (§9) |
| **Drifts onto a look-alike when target has no body track** | ↑ `kHeadReacqMinScore`, ↓ `kHeadReacqGateRatio`/`kHeadReacqAmbigRatio` (§7); inspect head ownership/ambiguity trace gates |
| **ID switch during occlusion/crowding** | ↑ `kHeadMatchVetoMin` (§7); ↑ `kReacqMaxDefer`, `kAnchor*` floors (§10); ↑ `kCoexistVetoIou` gating (§11) |
| **Loses the true target too easily (turns/receding/lighting)** | ↓ `kAnchorVetoRelaxed`/`kAnchorVetoRelaxTurned`; ↑ `kOrientHoldMaxMs` (§13); ↑ `appearance_thresh` tolerance |
| **Gimbal moves on an unsafe head/face result** | Keep `kObservedPartWeight=1.0`; tighten the corresponding head/face acceptance gates and verify `[HEAD_RECOVERY]`/`[FACE_RECOVERY]` traces |
| **NPU frame-time spikes** | ↓ `kReidMaxCand*`, ↑ `kPoseInferEveryN`, ↑ `face_recognition_every_n_frames`, ↓ `kFaceSweepMaxFaces` (§2, §6) |

> **Values are conservative first-pass defaults.** None of the recent changes are
> compile-verified in this dev env (no aarch64/OpenCV4/ACL) — validate against footage
> after the on-device cross-compile (`c_api/README.md`).
