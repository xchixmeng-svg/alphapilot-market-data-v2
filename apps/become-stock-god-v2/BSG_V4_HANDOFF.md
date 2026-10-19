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
