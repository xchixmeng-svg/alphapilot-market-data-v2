# BSG V4 HANDOFF — 2026-10-20

Repo: `xchixmeng-svg/alphapilot-market-data-v2`

Work branch: `bsg-learning-system-20261019`

Do **not** touch the production site yet. Do **not** touch the old `become-stock-god-v2` project/deployment.

## Product goal

Turn 《從菜雞到股神 / 成為股神》 into a true zero-to-competent K-line learning game. Core teaching rule:

`teach -> demo -> guided practice -> reduced hints -> independent test -> final daily-reveal simulation`

Hard rule: nothing may be tested before it has been explicitly taught.

## Current scope

Current course has **50 missions / 8 worlds**.

- Foundation: M01-M36
- Indicators / advanced evidence: M37-M46
- Integration: M47-M50
- Commercial defaults: free through M15; lifetime unlock NT$149

The user now wants the real complete course expanded further to roughly **80 missions plus 5-10 daily-reveal practical scenarios**. The existing 50 missions should be treated as the core curriculum skeleton, not the final amount of practice.

## Already implemented / locked

- Taiwan stock legal tick-size model and legal simulated prices
- Price axis must not show impossible prices
- Exact-price questions must show the relevant exact price on-chart; player must not be forced to infer only by relative vertical position
- M01 explicit OHLC teaching and anti-answer-leak behavior
- Teach-before-test contract
- Support/resistance / exact price guide infrastructure
- MACD, KD, RSI14 indicator panels / computation
- Drawing-practice infrastructure
- Daily reveal / anti-future-leak infrastructure
- PWA packaging/icons
- Mobile-first landscape validation path

Learning contract audit at the latest checked build reported:

- `currentMissions: 50`
- `worlds: 8`
- `legalAxis: true`
- `exactPriceGuides: true`
- `teachBeforeTest: true`
- `drawingPractice: true`
- `dailyReveal: true`

## Current blocker

The stricter mobile Playwright validation currently fails at **M47 touch drawing**.

Observed failure:

`M47 touch did not create price line`

The test uses an 844x390 mobile/touch viewport, dispatches `pointerdown` + `pointerup` to the chart canvas, and expects `window.BSG_V4_LAST_PICK` to become a finite legal-tick price.

Build, syntax and learning-contract checks pass before this step. The blocker is specifically the real mobile touch path for creating the price line.

### Do not fake-fix the test

Find and repair the actual root cause. Inspect at minimum:

- pointerdown / pointerup / click / touch event path
- pointer capture
- Canvas CSS size vs backing-store size
- clientY -> chart price mapping
- listener binding time / lifecycle
- whether `pricePickMode` and `onPricePick` actually reach the live chart instance
- whether overlay layers intercept touch input

The real device-equivalent path must work, and the chosen price must snap to a legal Taiwan tick.

## CI state at handoff

Latest checked commit before this handoff file: `f9dfd0ebc26874f01671413e0f92f1326c462acf`

- `Validate BSG V4 release candidate` — **success**
- `Validate BSG V4 complete course` — **failure** at the M47 mobile touch test
- Learning contract itself — **success** before the browser failure

Do not deploy production until all relevant validation suites are green.

## Architecture task after the M47 bug

Before expanding 50 -> ~80+ missions, inspect whether mission behavior is too heavily hard-coded by mission ID.

Implement the smallest safe refactor toward a data/config-driven lesson schema, e.g. fields such as:

- `phase`: teach / demo / guided / practice / exam / simulation
- `pricePickMode`
- `priceGuides`
- `indicatorPanel`
- `dailyReveal`
- drawing requirements / tolerance
- reveal policy
- explanation / feedback configuration

Goal: adding another 30+ missions and 5-10 daily simulations should mostly require content/config changes, not a growing chain of `if (mid === xx)` UI logic.

Constraints:

- Do not regress M01-M46 behavior
- Keep legal tick rules
- Keep exact-price visibility rules
- Keep teach-before-test
- Keep future-data leakage protections
- Keep current commercial defaults unless explicitly asked otherwise
- Run build + syntax + learning audit + mobile Playwright after changes

## Product direction for expansion to ~80 missions

Add practice depth rather than filler. Priority areas:

- OHLC / wick / red-green / tick-size repetition
- swing highs/lows and HH/HL/LH/LL
- trend vs range
- support/resistance: demo -> guided -> self-draw -> find-the-wrong-line
- volume and moving averages
- breakout / false breakout / retest
- stabilization vs reversal confirmation
- entry / invalidation / risk-reward on-chart interaction
- MACD / KD / RSI individually, then conflict / failure cases
- "no trade / insufficient evidence" as a valid answer
- multiple daily-reveal practical cases at the end, not just one
- score decision quality separately from eventual P/L

## Expected Claude return

After working, report:

1. root cause of M47 touch bug
2. files changed
3. commit SHA
4. CI results
5. whether any blocker remains
6. short architecture note for 50 -> ~80+ expansion

Do not deploy the production site unless explicitly told to do so.
## Update 2026-10-20 (Claude)

### M47 touch — fixed (`84bca09`)
Root cause: `chart.js` pointerdown called `cv.setPointerCapture()` unguarded as its first statement. When a browser
rejects capture (`NotFoundError`, pointer no longer active — WebKit after an early touch release/cancel, or a
synthesized event) the throw aborted the handler before the pointer was recorded; pointerup then found no pointer
and dropped the tap, so no line and `BSG_V4_LAST_PICK` stayed null. Now the pointer is recorded first and capture is
optional. Coordinate mapping verified: tapping the support guide returns exactly that price at DPR 1/2/3.
The complete-course CI keeps the synthetic-dispatch test and adds a real touchscreen tap through the live M47 lesson
(tap support guide → legal tick → confirm → graded correct).

### Lesson schema — M37+ are data (`parts/app.0298.part` engine, `parts/app.02985.part` content)
- Every mission from M37 is registered with `defineLesson({...})`; generic hooks serve suitable / question / qtext /
  explain / promptIntro / answerHTML, the chart layer (`app.0444`) and drawing grade (`app.0445`).
  No app code branches on a mission id ≥ 37 (the learning audit fails if one appears).
- Exercise count is per lesson (`exercises`, default 3): step dots, `Q n / N`, the solo last question and mission
  completion follow it, so a daily simulation can have 5–10 reveal steps.
- Proof of no behaviour change: a golden snapshot of M01–M50 (questions, options/correct, text, explanations for every
  answer, hints, intro, answer markup incl. feedback state, sample suitability over 400 rounds, final chart contract)
  is byte-identical before and after (sha256 prefix `dfe36d1f522c9e60`).
- M01–M36 still use the legacy per-module layers (277 id comparisons). They are untouched on purpose: migrate them
  only module by module behind the same snapshot check.

### Adding a mission (M51+) or a daily simulation
1. Metadata in content.js (`CT.MISSIONS.push({id, world, title, sub, skill, objective, rule, anti, focus})`).
2. Behaviour in a lesson part:
```js
defineLesson({id:51,
  sample:function(s){/* s.R, s.i → is this historical case suitable? */return true;},
  intro:[{label:'共同原則',text:'…'}],
  ask:[ {text:'…？',options:['correct','wrong','wrong']},
        {text:function(c){return'…'+c.px(c.q.sup);},options:function(c){return{options:[…],correct:k};}},
        {text:'…',options:[…]} ],
  explain:function(c,ans){return'…';},                 // or a string; default = rule
  chart:{indicator:'KD'} /* or */ function(c,kit){kit.line(c.sup,'#24d79b',c.quiz?'':'支撐',true);},
  interaction:{type:'pricePick',exercise:2,target:function(q){return q.res;},tolerance:function(q){return q.a*.5;},
               title:'現在換你畫',prompt:'…',readout:'尚未畫線',confirm:'確認 →'}
});
// daily simulation: exercises:6, reveal:{days:[0,1,2,3,4,5]} — exercise k sees only D0..D+days[k-1]
```
`c` = {m, R, ex, i, q:criterion, ind:{macd,kd,rsi14}, px, E, minA, maxA, ui}; chart `c` also has
{quiz, sup, res, atr, revealDay, out}. `BSG_LESSONS.validate()` reports malformed specs (CI runs it).

### Still open (not blockers)
- `skillPlan()` (400 practice / 100 exam mix) draws from the original 36 skills; decide whether M37+ and simulations
  join the mixed pools.
- Check the cover image source: building this branch from source emits `dist/landing.webp` from `assets/landing.b64`,
  which is 17,444 bytes while its RIFF header declares 98,510 (truncated); Chromium cannot decode it and the cover is
  black there. If the playable deployment shows the cover art, that image is coming from outside this build — keep it
  that way or replace `assets/landing.b64` with the intact file so a fresh build cannot regress.

## V5 complete product (Claude, 2026-10-20)

Goal: a beginner who finishes the game can open an unfamiliar chart and, unprompted, state the structure, the
location, the evidence, whether to trade, what to wait for, where to enter and where they are wrong.
Five layers, each only using what earlier layers taught:

| Layer | Data | Engine | Size |
|---|---|---|---|
| 1 核心教學 | `content.04.part` (M51–M63 metadata + `CT.ORDER`), `app.02987.part` (lesson specs) | `app.0297` course order, `app.0298` lesson schema | 63 lessons |
| 2 技能訓練場 | `SKILL_FAMILIES`, `LESSON_FAMILY`, `DRILL_GEN` in `app.02992.part` | same file | 205 drill types / 13 families, 400 per round |
| 3 混合判圖盲測 | `MIXED_STEPS` in `app.02993.part` | same file | 100 unseen charts × 5–7 judgements |
| 4 逐日實戰 | `tools/build_scenarios.mjs` → `build/scenarios.json` | `app.02994.part` | 12 scripts × 8 checkpoints |
| 5 股神畢業考 | `FINAL_DIMS`, steps in `app.02995.part` | same file | 3 unseen charts, 17 steps, 10-dimension report |

- **Teaching order** (`CT.ORDER`): new lessons sit right after the idea they extend (e.g. M51 tick size after M05,
  M56 量價衝突 after M26, M59 MACD 零軸 after M41). Unlock, "next lesson", lists and the free tier follow the order;
  the free tier is still "first 15 lessons" (`CT.FREE_THROUGH`), price unchanged.
- **M50 (逐日揭曉)** stays the last core lesson (it teaches the mechanic); it is excluded from the 400 / 100 pools.
- **Training ground**: a drill enters the pool only when its lesson is completed (`drillAvailable`). Chart-reading
  drills weigh 3×, definition drills 1× (auto-detected by asking the question on several charts).
- **Blind mixed exam**: EXAM pool only (2024–2025), every chart distinct, no topic labels, evidence panel rotates
  MACD / KD / RSI / volume; trade cases add entry + invalidation, wait cases add "what to wait for" + invalidation.
  Old V4 exam progress is kept in `S.examLegacy`; the new exam starts at 0 (`S.exam.format = 'mixed-v5'`).
- **Daily scenarios**: unlock after all core lessons + 60 training answers + 20 blind charts. Chart and rubric get
  arrays truncated at the current day (`upTo`), checked at every checkpoint by the V5 CI. Decision quality and P/L
  are scored separately (entry decisions weigh most); the case name is revealed only after finishing.
- **Final exam**: reserve 2024–2025 charts never used anywhere else; trend, click swing high/low, draw
  support/resistance, breakout, reversal, indicator noise, enter/wait/abandon, entry price, draw the stop, R:R,
  compare two plans, manage a position day by day. Report: 10 dimensions, strongest/weakest, common mistakes,
  chasing, indicator over-trust, early reversal guessing, respecting stops, overtrading.
- **Rubric** (`analyze`/`decide` in `app.02991`): structure first; down-trend, R:R < 1 or > 10% above MA20 → no
  trade; up-trend above MA20, < 8% extension, R:R ≥ 1.5 and not pressing resistance (unless a 1.5× volume
  breakout) → trade; otherwise wait. `build_scenarios.mjs` carries the same rules; CI checks they agree.

### Adding content later
- Lesson: metadata in `content.04.part` + id in `CT.ORDER` + `defineLesson` in a lesson part.
- Drill: lesson exercises become drills automatically; extra types go in `DRILL_GEN` (`family`, `lesson`, `sample`, `ask`, optional `chart`/`panel`).
- Mixed step: add to `MIXED_STEPS` and to `mixedPlan`.
- Daily scenario: add a script to `SCRIPTS` in `tools/build_scenarios.mjs` (detector may read the future; the game never does).

### CI
`bsg-v5-product-ci.yml` drives the real page (844×390 iPhone-landscape and 412×915 Android UA, real taps):
course order + 13 new lessons on 40 charts each, drill unlock by progress, 60 training answers, blind-exam pool
isolation / panel rotation / No-Trade share / prices visible on chart, 5 blind cases, daily gating + 12 scripts +
leak probe at all 96 checkpoints + a good-decision-but-stopped case, final exam gating + unseen pool + touch
picks + 10-dimension report, PWA files, zero console errors.

## V6 learning upgrades (Claude, 2026-10-20)

Why: multiple choice alone teaches recognition, not doing; the V5 blind exam could be passed by always
answering "no trade" (81% of answers); practice volume and feedback loops were too thin.

- **Balanced, clear cases** — `tools/build_cases.mjs` (shares `tools/rubric.mjs` with `build_scenarios.mjs`) picks
  the graded chart positions at build time: blind exam 18 trade / 41 wait / 41 no-trade (the rubric rarely says
  "trade", so 20 was the ceiling), each case clearly inside its class. Writes `build/cases.json` (embedded as
  `#case-data`). CI checks the game's `analyze()` returns the same class for every case.
- **Do it yourself** — `app.02996.part`: draw support / resistance, mark swing highs / lows, draw the stop and the
  target, scored by distance (0.4 ATR for levels, 0.3 ATR for stops); the reference answer is drawn next to the
  player's line. ≈ 40% of training questions and 3–4 steps of every blind-exam chart are these.
- **Outcome reveal** — after the decision is committed the next 20 days are shown with what a long at D0 (structural
  stop, first target) would have done, plus the player's running tally per decision. Before that step every chart
  is truncated at D0 (CI checks each step).
- **Review** — a missed skill returns on a new chart after 5, 20 and 60 questions (`S.v5.review`).
- **Volume** — 72 day-by-day cases (6 sets × 12 scripts, shuffled inside each set); the final exam opens after one set.
- **Start test / retest** — 10 + 10 unseen charts (reserve C #1–10, early vs late windows, same answer mix);
  final exam moved to reserve C #11–20. Report compares the two.
- **Review build** — `window.BSG_OPEN_ALL=true` (injected only into the preview copy) opens every lesson and layer.
  The production build never sets it (static CI check).
- **Measured** (training rounds, 2020–2023, see `cases.json → base`): avg return of a long at D0 over the next
  20 days — rubric "trade" −0.65%, "wait" +0.07%, "no trade" +0.75%; forward returns by class are close to random.
  The decision rubric teaches a process; the data does not show it has an edge. Decide before marketing it as one.

## Background music (2026-10-20)
- `app.05.part` now plays a calm, consonant C-major loop (I – vi – IV/C – V/B), pure sine pads (no detuned pairs,
  so no beating), soft felt-piano arpeggio every 2 s, master low-pass 2.2 kHz; every voice stays ≥ 220 Hz.
  Mixer profile id is `calm-felt-piano-v3` (`tools/audit36.mjs` updated).
- When this branch is merged into the production pipeline, `.github/workflows/bsg-v3-build.yml` line ~102 still
  expects `airy-piano-bell-v2` and must be updated to `calm-felt-piano-v3` (left untouched here: production pipeline).
- Later the same day: background music is now Chi's track **Focus Flow** (`assets/bgm.mp3`, made with Suno,
  loudness-normalised to −20 LUFS, 112 kbps, 1.5 s fade-in / 2 s fade-out, 2.9 MB) copied to `dist/bgm.mp3` and looped
  at gain 0.45 through Web Audio (iOS ignores element volume). If the file cannot play, the synthesized calm loop
  takes over (profile ids `focus-flow-file` / `calm-felt-piano-v3`). Commercial use needs a paid Suno plan at the
  time the track was generated.
