# Follow-up: numeric arguments that are computed (from Task 9 Step 6)

Not started. Written 7 Nov 2026 because the triage found at least half of the fired `num` samples are computed.

## Evidence (read by hand, `runs/.../ceilings/num` on both sft-v2 + check_refs ASR baselines)

| Class | Holdout (17 fired) | Dev (5 fired) |
|---|---|---|
| Computed from tool values or from the request | 9 | 4 |
| Quantity copied from a converted value (`shopping_list.add` quantity 300 instead of 1) | 6 | 0 |
| Extra optional argument the gold call omits | 1 | 1 |
| Wrong digits from ASR | 1 | 0 |

Computed cases: a delay from a clock time to an alarm or event time (`delay_minutes` 85, 150, 245, 475), a threshold
equal to a current value minus an amount (250 = 300 - 50, 108 = this month minus the air conditioner), a maximum times a
number of days (14 x 30 = 420), a group size from a tool result (quantity 4), minutes between two spoken times
(7:30 to 8:10 = 40, 9:00 to 9:45 = 45), steps from a distance, and a savings term in months to an event date.

Ceiling from the same run: replacing every `num` error by gold gives holdout +2.97 [+1.35, +4.94], dev +1.37 [+0.33, +2.63].
That is a ceiling, not a forecast: the 6 quantity cases are a different error and are not arithmetic.

## Open design question (K decides before any code)

Every call on the ToolEnv is scored, so a calculator tool call is not possible. The candidates that send no extra call:

1. Records that show the arithmetic in the assistant text before the call (the `sft-r1` reasoning sentences lost on
   holdout, -6.87 [-11.43, -2.35], so this needs a narrower form: one line, only before a call with a computed number).
2. A rewrite of a call's numeric argument from values in the request or tool results, found by schema type and description
   only (no app names). It needs a rule per computation class above and the simulator first (`lifemate.sim`).

## Steps

- [ ] **Step 1:** Write a simulator rewrite for the two cheapest classes (minutes between two spoken times; amount subtracted from a value read earlier) and measure it on dev, then holdout, both ASR baselines. Stop if holdout is below 0 or under 3 samples fire.
- [ ] **Step 2:** Only if Step 1 passes the gate in the 6 Nov plan (holdout >= +0.5 with the CI above 0), add it to the agent behind an option and run it once on holdout and dev.
- [ ] **Step 3:** Close it with a `RESULTS.md` row either way.
