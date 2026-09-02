# 主目标匹配算法优化 — 设计与审计（2026-06-25）

> 状态：**审计中 / 待评审**。本文记录对 `LightTracker` 匹配链路的静态审计结论，
> 以及针对已知失败模式 #1–#4 的**预草拟补丁**。除 #0（纯日志、已应用）外，
> 所有行为性改动**均未应用**，待现场（live-on-device）抓到决策 trace 后逐条评审。

## 1. 问题与约束

- 场景：PTZ 单目标跟踪。难点 = 上半身/近距大占比 + 人群 + 主目标从他人身后穿过 +
  同衣 + 侧身骨架不可靠 + 正面遮挡者骨架反而更像缓存的主目标。
- 失败现象：ID switch / 间歇丢失 → 输出轨迹差 → 云台"乱动"（**不可接受**）。
- **验证只能在设备上实跑**（无离线回放）。⇒ 每个测试周期昂贵；改动必须
  **批量、安全、可评审**；禁止盲目调阈值（本匹配器有 ~10 个相互作用的否决，
  盲改一处会"修好一个场景、悄悄弄坏两个"）。

## 2. 关键事实：用户的设想大多已实现

`match_main_target_unified`（`LightTracker.cpp:1258+`）已包含：
- `VisibilityState`（FULL→HEAD_ONLY）按可见度对各信号重加权：UPPER 时 ReID=0.15/
  shape=0.05/oks=0；HEAD_ONLY 时 ReID/shape/oks=**0**，头部预算 floor 升到 0.50。
- 头部信号注入 + 头部硬否决（`HEAD_VETO`）。
- 外观歧义自适应降权；同衣靠 共存排除 / 运动否决(M1) / 空间连续否决 / 人脸硬覆盖。
- 遮挡 FSM「宁可不匹配也不匹配错」+ 单候选拒绝；teleport gate；C-identity 重捕复核。
- body_shape 已降为软 tie-breaker（正解：避免正面遮挡者骨架夺锁）。

⇒ 优化目标 = **定位现有机制为何在该场景失效（调参/盖盲点/估计错）**，而非新建算法。

## 3. 审计结论

### 3.1 ✅（已应用，#0）决策 trace 是诊断前提，原先被丢弃
全部 `null_sink` 日志默认 `verbose=false`（`LightTracker.cpp:26`）→ 运行时丢弃。
**已加** `kMatchTrace`（默认 true）：每帧一行 `[MATCH]` 摘要（状态/可见度/候选数/
遮挡/头部新鲜度 + 最终决策信号 + top2 总分差 + alert/blind）。非行为，零回归。

### 3.2 ⭐（高价值，安全）`coast_weight_` 命中即恒置 1.0 —— "云台乱动"的直接机制
`LightTracker.cpp:745-752`：一旦找到主匹配，无条件 `coast_weight_ = 1.0f`，
**完全不看匹配把握度**。于是当 ID switch 越过否决（top2 差极小 / `id_switch_alert_`
激活 / anchor 刚过否决线）时，输出仍告诉云台"100% 确定"→ 云台全权扑向冒充者。
而 `suspect_streak_` / `id_switch_alert_`（`update_quality_monitor`，在 matcher 内
2096 行已更新）此刻即是现成的"身份不确定"信号，且在 update() 751 行作用域内可读。
→ **预草拟补丁 P2**。

### 3.3 ✅ 可见度估计无 bug（排除一个静默退化源）
`assess_visibility`（`2931+`）稳健：近距上半身底部截断 → 不误触"远处全身"救援 →
正确判 UPPER；遮挡期无 pose 退回滑行 KF 框（全身形）→ 维持 FULL，不误清 ReID。
唯一注意：EMA+4 帧迟滞带来几帧滞后（反抖动，非 bug）。

### 3.3b ✅ 头部连续性几何无 bug（排除）
`standalone_heads_` 存 xyxy-in-Rect（`Detector.cpp:226` 证实 `box=Rect(x1,y1,x2,y2)`）。
`find_standalone_head_near` 中心计算 `(h.x+h.width)/2=(x1+x2)/2` 正确；门控用真实像素距
+ 歧义双候选拒绝 + 按 tsu 收缩吸附半径。`reconstruct_body_from_head` 收发一致。无 bug。

### 3.3c ✅ MainTargetPredictor coast 无 bug（排除）
线性回归测速 + 上限钳制（`max_center_vel_px_sec`）+ `center_vel_scale` 阻尼（防"引导框
飞得比目标快"）+ 垂直位移钳 5%（防云台上下抖）+ 尺寸钳 + 热速度 TTL 复用 +
`max_extrapolation_sec` 封顶 + 持在 coast 终点（不回扫丢失点）。设计稳健，非"乱动"成因。

### 3.4 ⚠ 共存否决（coexist-veto）对"遮挡中才出现的冒充者"结构性失效
`coexist_with_main` 仅在「主目标命中 且 该非主轨迹关联到自身检测」时 +1
（`2515-2519`），否决需累计 `kCoexistConfirm=10` 帧。快速穿越时冒充者：
① 中途才出现 → 计数=0；② 其检测常被主目标占用或重叠 → 始终建不起稳定独立轨迹
→ 永远确认不了 → **恰在最需要的场景下无法否决**。修复较敏感（可能延迟正常重捕），
**需 trace 确认确属此路径** → 预草拟补丁 P1（低置信、待验证）。

## 4. 预草拟补丁

### #0 `[MATCH]` trace — **已应用**（`LightTracker.cpp:43` 开关 + 决策点摘要）

### P2 — 命中置信度 → 衰减 `coast_weight_`（**推荐先上，最安全**）
**只改输出置信度，不改"匹配谁"**，故不可能引入新的 ID switch；只在 matcher 自身
不确定时让云台收敛更稳，直接缓解"乱动"。

新增成员（`LightTracker.h`）：
```cpp
float pending_match_conf_ = 1.0f;                 // 本帧主匹配把握度（喂 coast_weight_）
static constexpr float kMatchConfFloor = 0.45f;   // 下限：真目标仍可被跟住
```
在 `match_main_target_unified` 末尾（`best` 终态、`[MATCH]` 摘要附近）计算：
```cpp
float conf = 1.0f;
if (best.matched && !best.from_face) {            // 人脸确认 = 满置信
    if (best.anchor_sim < 0.50f) conf *= 0.75f;   // 身份刚过否决线 = 弱
    float gap = (candidates.size() >= 2)
              ? (candidates[0].total - candidates[1].total) : 1.0f;
    if (gap < 0.06f) conf *= 0.70f;               // 歧义：runner 紧贴
    if (id_switch_alert_)          conf *= 0.40f; // 质量监控告警
    else if (suspect_streak_ >= 2) conf *= 0.60f;
    else if (suspect_streak_ == 1) conf *= 0.85f;
    conf = std::max(conf, kMatchConfFloor);
}
pending_match_conf_ = conf;
```
消费（`LightTracker.cpp:751`）：
```cpp
- coast_weight_ = 1.0f;           // 真实命中：满置信
+ coast_weight_ = pending_match_conf_;   // 命中置信度（matcher 不确定→云台收敛更稳）
```
并在 `[MATCH]` 行追加 `conf=` 字段，便于现场观察。
**风险**：身份真弱时云台略保守（通常正是期望）。无 ID-switch 回归面。

### P1 — 遮挡期"非人脸 + 空间不连续"的匹配 → 优先 coast（**需 trace 验证**）
针对 3.4。思路：`OCCLUDED` 期接受的非人脸匹配，若既非人脸确认、又与未污染的
头部/KF 预测空间不连续（head_match 与 body-IoU 均低）→ 极可能是遮挡中冒出的
冒充者 → 延迟接受（`best.matched=false`，复用 `reacq_defer_count_` 预算）。
与现有 单候选拒绝 / VIS_GATE / M2 有重叠，需确认是否真有未覆盖的漏网路径，
避免叠加导致正常重捕被过度延迟。**先抓 trace（st=OCC matched=1 face=0 且 top2 差小）
确认，再定稿。**

### P3 — 人群中 ReID 限流漏掉真目标（观察项）
CLEAR 期 `kReidMaxCandClear=2`：仅对 cheap 预排前 2 名跑 ReID，其余 reid=0 →
被 anchor 否决。人群里真目标若不在 IoU+center 前 2 名即被漏。`in_danger` 时关限流，
但需确认人群是否稳定触发 danger。trace 信号：`st=CLEAR near≥4` 时丢失。
候选改法：把"上一帧 winner 的 IoU 最近邻"强制纳入 ReID 集 / 提高上限。

### P4 — 头部被更高者遮挡 → 头部连续性失效（低优先）
"头永远在"在穿越更高者身后时不成立；此时退化为盲 coast（安全），改进只在重捕侧。

## 5. 下一步
1. 现场用带 `[MATCH]` 的版本复现 1–2 次失败，回传失败帧附近的 `[MATCH]` 行。
2. 据 trace 定位实际触发的失败模式 → 优先定稿 P2（安全）→ 再按证据处理 P1/P3。
3. 每次只改一条、单独构建测试，用 `[MATCH]` 的 `top2`/`conf`/`alert` 验证无回归。
