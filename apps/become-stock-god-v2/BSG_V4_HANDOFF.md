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