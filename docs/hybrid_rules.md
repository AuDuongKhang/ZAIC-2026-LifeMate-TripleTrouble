# Hard-coding what we can, LLM for the rest

Research note for K, written 4 Nov 2026. It asks one question: which agent decisions can Python make, while the LLM keeps
the rest? Nothing here is built or run on the agent yet. Every number comes from train labels or from existing docs, so
it estimates room to gain, not a score gain.

## Summary

- **Worth trying first:** a Python date resolver that corrects date arguments, then the existing `freetext.py` and
  `minimal_reads.py`. None of the three needs a retrain.
- **Do not hard-code id picking or the ask decision by word match.** On gold text it picks the right id only about 40%
  of the time, and `ask_guard` already lost at every setting.
- **Keep the LLM for choosing the tool.** Wrong write tools cost only 1.35 points on holdout, and keyword lookup is far
  weaker than the model there.
- **Expect a few points, not a jump.** The top 5 needs 95.0 and we have 53.58. I see no evidence that these rules close
  that gap on their own.

## Why look at this now

- Public leaderboard, SLM column: 96.51, 96.25, 95.72, 95.3, 95.0 for the top 5. Those teams score about the same as
  their own APIs/Others entries, so model size does not look like the limit.
- Perfect ASR would not close the gap: sft-v2 + check_refs on gold text scores 72.86 on dev and 50.96 on holdout.
- A teammate reports that teams on the other track go from question to intent to hard-coded Python, including for
  cases like asking for more information. I could not verify this.
- `context/RULE.md` does not ban rules in the system. The final ranking uses private_test, not the public leaderboard.

## Where holdout points go

From `docs/error_analysis.md`, holdout, gold text, 49 points lost in total:

| Category | Points |
|---|---|
| State correct, F1 < 1 (read calls, free text) | 17.90 |
| Same write tools, wrong arguments | 11.11 |
| Ask/act wrong (5.72 acted, 3.54 asked) | 9.26 |
| Missing or extra write (4.04 each) | 8.08 |
| Different write tools | 1.35 |
| No calls at all | 1.01 |
| Out-of-scope sample, agent made calls | 0.34 |

## Evidence by decision

| Decision | Evidence | Verdict |
|---|---|---|
| Relative dates in arguments | Resolver agrees with gold on 116 of 121 pool samples and 42 of 47 dev samples where it finds a phrase | **Try first** |
| Finish free text from tool results | `freetext.py` exists, not wired into `react.py`. Upper bound 15.14 points on holdout ASR, overlapping other buckets | **Try** |
| Drop unstated optional read filters | `minimal_reads.py` exists, not wired in. Estimated +2.2 to +2.6 on holdout. 60% of gold reads have no arguments | **Try** |
| Numeric conditions and arithmetic | LLM would extract numbers, Python would compute. Not measured, and it needs a structured model output | Later, likely needs a retrain |
| Read calls from the write tool | The most common read pattern per tool explains 56.8% of 1942 write and ask calls | Not a lookup table |
| Pick an id by word match | See below: about 40% right | **Do not hard-code** |
| Ask decision by word match | See below, and `ask_guard` lost | **Do not hard-code** |
| Choose the tool by keyword | Top-1 tool 22% on dev, 33% on holdout. Top-3 48% and 62% | Keep the LLM |

I found no `RESULTS.md` row for `freetext` or `minimal_reads`. If you already simulated them and they failed the gate,
drop them from the list.

### Date resolver

I matched "mai", "hôm nay", "ngày mốt", "thứ X này" and "thứ X tuần sau" in the gold request, resolved them from
`current_time`, and checked whether that date appears in the gold arguments. The script was a quick probe and is not in
the repo.

| Subset | Samples with a date argument | Phrase found | Agrees with gold |
|---|---|---|---|
| pool | 265 | 121 | 116 |
| dev | 89 | 47 | 42 |

- **Coverage is about half.** The rest use phrases I did not handle, or the date comes from a tool result.
- **Disagreements come mostly from several phrases in one request**, such as "hôm nay" for the condition and "sáng mai"
  for the reminder. A safe rule corrects a date only when the request holds exactly one date phrase.
- **The gain is capped.** The model already gets most of these right. On holdout ASR, wrong dates are at most 3.70 points
  and wrong times 3.03, and those buckets overlap others.
- **Why it may beat `date_hints`:** the date table lost about 2 points on the model that never trained on it (dev -2.11,
  holdout -2.05). A correction after generation leaves the prompt unchanged, so it needs no retrain.

### Id picking and the ask decision

For each gold write argument with an `x-ref`, I took the read result list holding that id and scored every candidate by
words shared with the gold request, without stop words. A unique top score counts as a pick. Otherwise the rule asks.

| Multi-candidate ids where gold picks | Right pick | No unique winner (would ask) | Wrong pick |
|---|---|---|---|
| pool, 686 | 276 (40%) | 311 (45%) | 99 (14%) |
| dev, 212 | 89 (42%) | 87 (41%) | 36 (17%) |

- **Gold asks are caught well, which is not enough.** The rule asks on 78 of 90 pool cases and 22 of 23 dev cases where
  gold asks, but it would also ask on over 40% of the cases above.
- **Same result with your scoring.** `ask_guard` scores differently and already lost: best dev -14.93, holdout -6.64.
- **Likely cause, not measured:** requests pick by earliest, cheapest, a date, or an earlier turn rather than by a name.
- **What remains possible:** only 12% of pool ids and 10% of dev ids have a single candidate, so that easy case is small.

## Suggested order

1. Check the status of `freetext` and `minimal_reads`. Simulate them with `lifemate.sim` if that has not been done.
2. Build the date correction as a `lifemate.sim` rewrite, on the sft-v2 + check_refs ASR runs.
3. Wire whatever passes into `ReActAgent` as opt-in kwargs, applied before `env.validate`.
4. Confirm each with one GPU run and `lifemate.compare`, under your existing gate.

## Constraints

- **Unseen apps:** rules must work from the schema (`kind`, `x-ref`, `required`, `format`), not from app names. Train and
  public apps differ, so we expect the same for private. I did not find the organisers saying so.
- **No rules from public_test:** the rules ban labelling it to train or tune. Write and tune rules on pool and dev only.
- **Your rewrite rule:** the score-push plan says a rewrite must not send a call the model would not have sent. A date
  correction changes an argument of a call the model did send. Please decide whether that is acceptable.
- **Measurement:** `sft-final` has seen dev and holdout, so comparisons stay on sft-v2 + check_refs.

## Questions for K

1. Did `freetext` and `minimal_reads` get simulated, and with what result?
2. Is a correction of a model's date argument acceptable under the rewrite rule?
3. Do you want this tried before the `sft-final2` retrain? Date hints in that retrain may make the date correction
   unnecessary.
