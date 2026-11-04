# Tuning the agent: intent router and chat history

Research note for K, written 4 Nov 2026, as a companion to `docs/hybrid_rules.md` (PR #33). It asks where agent tuning
can still gain points, and whether an intent router or history work should be part of it. Nothing here is built or run
on the agent. Every number comes from train labels and gold text, so it shows room to gain, not a score gain.

## Summary

- **Do not build an intent router yet.** An act/ask/out-of-scope classifier is already rejected in the score-push plan.
  An app filter either loses the right app too often or removes almost nothing.
- **Do not trim history or tools to save tokens.** On gold text, the first prompt peaks at 10,398 tokens against the
  16,384 limit, and no dev or holdout sample drops a history turn.
- **Measure history errors first.** 22-27% of samples with history take a gold id only from history results. We do not
  yet know whether these samples lose more points. That split needs no GPU.
- **Keep the existing candidates first:** `freetext`, `minimal_reads` and the date correction from PR #33. They target
  larger buckets than a router does.

## Intent router

"Intent router" can mean two things, and they fail for different reasons.

**1. Classify act, ask or out of scope before the first call.** The score-push plan already rejects this: ask depends on
read results, so it cannot be decided before the reads. Nothing here changes that.

**2. Pick likely apps and hide the others.** I ranked each sample's installed apps two ways and checked whether all of
its gold apps stayed in the top k. An ask counts the app of the tool it asks about.

- **Schema:** words shared between the request and the app's tool names and descriptions, weighted by IDF from the pool.
  This works on unseen apps.
- **Bank:** the app's best word match against solved pool requests.

Samples with gold calls keep all their gold apps this often. Installed apps average 9.5 on dev and 8.5 on holdout.

| Router | Top k | Dev (275) | Holdout (290) |
|---|---|---|---|
| Schema | 3 | 52.4% | 62.1% |
| Schema | 5 | 69.1% | 80.0% |
| Schema | 8 | 86.9% | 96.2% |
| Bank | 3 | 79.3% | 19.3% |
| Bank | 5 | 96.7% | 30.0% |
| Bank | 8 | 100.0% | 66.9% |

Adding the apps used in history helps little, except for low k on holdout: schema top 5 reaches 70.2% on dev and 83.8%
on holdout.

What this means:

- **Bank fails where we need it most.** Held-out apps are never in the pool, and 42% of public samples call a public-only
  app (`docs/error_analysis.md`).
- **Schema has no useful setting.** Top 5 loses the right app on 20-31% of samples, and hiding a gold app should cost
  most of that sample. Top 8 is safer but hides only about one app.
- **The target bucket is small.** Different write tools cost 1.35 points and no calls at all cost 1.01 on holdout gold
  text, out of 49 lost (`docs/hybrid_rules.md`).
- **ASR noise would remain.** On holdout, 26 of the 61 samples that lose points to ASR call other tools, but a router
  reads the same noisy text. I did not simulate a router on ASR text.
- **It shifts the input.** The current models trained on full tool lists. This is the same objection your plan raises
  for tool-description compression.

## Chat history

| | Pool | Dev | Holdout |
|---|---|---|---|
| Samples with history | 606 of 948 (64%) | 192 of 300 (64%) | 220 of 297 (74%) |
| Mean turns (max) | 6.5 (24) | 6.1 (24) | 8.4 (24) |
| Gold id found only in history results | 154 (25.4%) | 43 (22.4%) | 60 (27.3%) |
| Gold app also used in history | 230 (38.0%) | 69 (35.9%) | 115 (52.3%) |

The last two rows count samples with history.

First-step prompt size, with four examples and gold text. I counted it with the Qwen3 chat template and a 512-token
output reserve, which is stricter than the current 384.

| | Mean | p95 | Max | History turns dropped |
|---|---|---|---|---|
| Dev | 5,889 | 9,019 | 10,398 | 0 of 192 samples |
| Holdout | 5,271 | 7,956 | 9,381 | 0 of 220 samples |

What this means:

- **Space is not the problem.** `build_messages` can drop old turns without logging it, but this never happened here.
  Later steps add tool results, and I did not measure those.
- **History cannot be cut by rule.** It holds the only copy of a gold id in about a quarter of these samples, and those
  ids may sit in old turns.
- **The few-shot bank never shows this pattern.** `_sees_its_ids` removes 154 of 948 pool samples, all with history,
  because examples are shown without history. That rule is sound, and SFT records still include history. But the
  examples never demonstrate reusing an id from an earlier turn. Whether this costs points is not measured.
- **History ASR is low priority.** On holdout, 30 samples have errors only in history. They lose 3.3 points on average,
  and 1 of 30 loses points. Errors in the final request cost far more.

## Suggested order

1. **Split existing runs, no GPU.** Use the per-sample scores of sft-v2 + check_refs on gold and ASR, on dev and holdout.
   Compare samples without history, with history, and with a gold id only in history.
2. **Count wrong-app errors, no GPU.** From the same traces, count samples whose first write or ask goes to an app outside
   the gold apps. A router can only help these. If the count is small, drop the router for good.
3. **Act on the split.** If samples with history ids lose clearly more, simulate or test the smallest fix for what the
   traces show, for example a history-id candidate list in the tool message. I have no evidence for a specific fix yet.
4. **Run the PR #33 candidates.** Simulate `freetext`, `minimal_reads` and the date correction with `lifemate.sim`, if
   that has not been done.
5. **Confirm each change** with one GPU run and `lifemate.compare`, under your existing gate.

Please avoid re-testing these, which are already measured:

| Option | Result |
|---|---|
| `date_hints` without retraining | Dev -2.11, holdout -2.05 |
| `leak_guard` | +0.0 |
| `refusal_check` | 0 or -0.67 |
| Prompt vote | Dev -0.33, holdout +0.43 |
| `ask_guard` | Lost at every setting: best dev -14.93, holdout -6.64 |

Raising `max_steps` also has little room: on public, only 2 samples hit it (`docs/error_analysis.md`).

## Constraints

- **Measurement:** `sft-final` has seen dev and holdout, so compare on sft-v2 + check_refs. Tune against one baseline at
  a time, because `sft-v4` with synthetic data (PR #34) will change the model.
- **Unseen apps:** rules must work from the schema, not from app names.
- **No public_test tuning:** write and tune on pool and dev only. The rules ban labelling public_test to train or tune.
- **Evidence limits:** the probes were quick scripts and are not in the repo. A30 timing is still unmeasured.

## Questions for K

1. Are the sft-v2 + check_refs per-sample files and traces still available? Steps 1 and 2 depend on them.
2. Do you have a router design in mind that differs from the two above, such as a hint in the prompt instead of a filter?
3. Should this wait until `sft-v4` is measured, so that we tune the model we are likely to ship?
