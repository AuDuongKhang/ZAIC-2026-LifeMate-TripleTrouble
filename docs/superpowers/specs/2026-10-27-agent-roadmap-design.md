# LifeMate agent roadmap: tool-use agent, verified SFT, then RL on the metric

Date: 2026-10-27. Status: draft for team review.

## Intent

The only goal is the **private test leaderboard rank**. Public test is a rehearsal; private test is expected to be harder and to contain apps we have never seen. Everything below is chosen to maximise the score on *unseen apps with our own ASR*, inside the competition limits, with a two-person team that has never trained an LLM, in the ~3 weeks left before the Docker deadline (16 Nov 2026 23:59 GMT+7).

Today the repo has the runner, the local scorer and two reference agents (`oracle`, `empty`). There is no real ASR and no real agent yet, and `RESULTS.md` is empty.

## The idea in one paragraph

Build the system in tiers, where each tier is a complete, submittable system and the next tier is only started when the previous one works. Tier 0 is a training-free tool-use agent around an off-the-shelf <5B instruct model (ReAct-style loop, tool retrieval to fit 16k tokens, free `env.validate()` checks before every scored call). Tier 1 builds training data that teaches *reading schemas* instead of memorising apps (Hammer-style function masking, ASR-noisy transcripts, optional teacher-generated requests that are kept only if they verify). Tier 2 is LoRA SFT on that data. Tier 3 is GRPO whose reward is **exactly the competition metric**, computed locally by replaying calls in the train `ToolEnv` with `lifemate.scoring`. Tiers 2 and 3 can run because this competition gives us a verifiable reward that is identical to the leaderboard metric. Most tool-use RL papers have to approximate theirs.

## Papers and what we take from each

| Paper | What we take |
|---|---|
| ReAct (Yao et al., ICLR 2023) | The think → call → observe loop. Ask decisions depend on tool results, e.g. several contacts named "Minh". |
| TinyAgent (Erdogan et al., EMNLP 2024 demo) | Tool RAG for small models: put only the relevant apps' schemas in the prompt. |
| CRITIC (Gou et al., ICLR 2024) | Self-correction driven by an external checker. Here the checker is `env.validate()`, which is free and not scored. |
| When2Call (Ross et al., NAACL 2025) | Treat act / ask-for-info / do-nothing as an explicit decision and train on all three. |
| Hammer (Lin et al., ICLR 2025) | Function masking: randomise tool and parameter names during training so the model relies on descriptions. This is our main tool against unseen apps. |
| ToolACE (Liu et al., ICLR 2025), APIGen (Liu et al., NeurIPS 2024 D&B) | Synthetic tool-use data kept only after layered verification (format, execution, semantics). |
| STaR (Zelikman et al., NeurIPS 2022) / rejection-sampling fine-tuning | Keep only trajectories that score 1.0 when replayed, then fine-tune on them. |
| GRPO (Shao et al., DeepSeekMath 2024), ToolRL (Qian et al., 2025), Tool-N1 (Zhang et al., 2025) | RL with verifiable reward for tool calling. ToolRL-style partial credit (tool name, argument keys, values) avoids an all-zero reward early in training. |
| HyPoradise (Chen et al., NeurIPS 2023 D&B) | Let the LLM see the ASR N-best list rather than only the 1-best, so it can fix Vietnamese names and numbers. |

## Goals

1. A real ASR backend and a real agent, registered in `asr/__init__.py` and `agents/__init__.py`, running end to end on public_test within limits.
2. An internal score we trust: **leave-app-out validation on train with our own ASR** (details under Evaluation).
3. A Docker image that reproduces our best system offline, deterministically, under 30 s per sample, ready well before 16 Nov.
4. Each tier is kept only if it beats the previous one on held-out apps.

## Non-goals

- Full-parameter fine-tuning, pre-training, or any model > 5B (LLM) / > 6B (system).
- Long "thinking" traces at inference: the 30 s budget on an A30 does not allow them.
- Training a custom ASR model. We pick and wrap an existing one, and fine-tuning ASR is out of scope for this roadmap.
- Using anything from public_test or private_test (audio, transcripts, apps, labels) as training or tuning data. The rules forbid labelling test splits. Public_test is only for leaderboard submissions.

## Anti-goals: any of these counts as a failure even if train scores look good

- A model that scores well on train apps but collapses on apps it has not seen.
- Any sample over 30 s or any crash in Docker. Both score 0.
- Non-deterministic output between two runs of the Docker image.
- Spending the last days on RL with no submittable system ready.
- Reading `<split>_environment/` from solution code, or creating more than one `ToolEnv` per sample at inference.

## Constraints (from `context/` and `CLAUDE.md`)

- Model sizes: LLM < 5B params with context ≤ 16,384 tokens, whole system < 6B.
- Hardware and time: scoring runs on 1× A30 24 GB, ≤ 30 s per sample including ASR.
- Runtime: offline Docker, fixed seeds, greedy decoding.
- Every `env.call` is scored. Schema-rejected calls count as unmatched, so validate first.
- Out-of-scope samples (gold `[]`) score only with an empty submission.
- Tool specs: ~13k characters on average, ~25k max. Train has 46 apps, public_test 36, with 25 shared, so private test will very likely have unseen apps.
- Compute: molab sessions of up to 12 h, and the number of sessions per week is unknown. Plan each training job to fit in one session and to resume from a checkpoint (below).
- Team: two people, both new to LLM training. The **Lead** (repo owner) takes the main track. **Each person has their own molab account**, so the two tracks do not compete for sessions.
- **Docker runtime is unknown** as of 27 Oct. `context/` only says "separate guidelines", and the portal page could not be read automatically. Unknown are the base image, the CUDA/driver version on the A30 host, and whether model loading or warm-up counts toward the 30 s. Design for the worst case until the organisers say otherwise:
  - the inference engine must also have a working HF-transformers path, in case vLLM does not fit the host;
  - load the models once before the first sample and warm them up;
  - keep per-sample time at roughly half the limit on A30-class hardware;
  - track B asks the organisers (portal or Discord/Facebook group) and updates this section.
- Submission (`context/SUBMISSION_GUIDANCE.md`):
  - The Docker package must include **training code and data**, not only inference. Keep every training run reproducible from scripts in the repo: data generation, SFT and GRPO with fixed seeds and pinned configs. Synthetic data generated by external models is shipped as files.
  - At most 2 Docker submissions, and only the last one counts.
  - At most 5 public leaderboard submissions per day, so use them for finished systems only.
- Git rules from `CLAUDE.md`: branch per topic, PR into `main`, `main` always works on molab.

## Architecture (target)

```
audio ──ASR (N-best)──► Task(request, request_nbest, history, context, tools)
                              │
                     tool selection (keep top-k apps + sys.ask_missing_info)
                              │
              ┌───────── agent loop (one ToolEnv) ─────────┐
              │ LLM → tool call (or stop)                   │
              │ env.validate → if error: feed back, retry   │  ≤ N steps, token + time budget
              │ env.call → result appended to context       │
              └─────────────────────────────────────────────┘
                              │
                 submission line (calls + state hash)
```

- The agent stops when the model emits no tool call. If no call was made at all, the submission is empty, which is the out-of-scope answer.
- A schema error found by `env.validate()` is fed back to the model and **never sent through `env.call`**. After a fixed number of failed retries, the agent drops that call.
- A hard wall-clock guard inside the agent stops the loop before 30 s, counting from sample start and including ASR time.
- `Task` gets one optional new field, `request_nbest: list[str]`, filled by ASR backends that support it. History turns can get the same treatment if it helps.

## Work split

Tightly coupled work goes to one person. The two tracks meet at three file-level interfaces (next section), so they can work in parallel.

### Track A, the model side (Lead)

Everything the model sees and produces, because the prompt format, the training data rendering and the RL rollouts must share one piece of code:

- Tier 0 agent: prompt rendering, tool selection, agent loop, validate-and-retry, time guard, inference engine.
- Base LLM choice.
- Tier 2 SFT pipeline: rendering training records into chat format, LoRA training, checkpoint/resume.
- Tier 3 GRPO: rollouts through the train `ToolEnv` with the same agent loop, plus the reward.

### Track B, inputs, evaluation and delivery (Partner)

Everything around the model:

- ASR backend: pick the model, implement N-best, measure WER and latency.
- An ASR transcript cache for all train audio, so training uses realistic noisy text.
- Evaluation: the leave-app-out split and per-group score breakdowns.
- Tier 1 data generation: function masking, ASR-noise variants, and optional verified teacher data.
- Docker packaging, offline test, timing and determinism checks.

### Interfaces between the tracks

1. **ASR cache**: `asr_cache/<split>/<asr_name>.jsonl`. One line per audio file: `{"audio": "<file name>", "text": str, "nbest": [str, ...]}`. Track A's training reads it, and the ASR backend at inference produces the same fields.
2. **Training records**: `train_records/<name>.jsonl`. One line per (possibly augmented) sample, with:
   - the `Task` fields (with ASR text);
   - `gold_calls`: a list of `{name, arguments}`;
   - `name_map`: masked name → real name, empty if not masked;
   - `source`: `gold`, `masked` or `synthetic`;
   - `split`: always `train`.

   Track B writes these records and Track A renders them. Records must contain no prompt formatting.
3. **Holdout list**: `configs/holdout_apps.json`. Track B writes it, together with the seen-app dev sample ids. Every training job excludes samples that use any listed app and excludes the dev samples, except the final retrain. Every report shows seen-app dev and held-out-app scores separately.

## Tiers

### Tier 0: training-free agent (Track A), with real ASR (Track B)

- Pick the base LLM by running two or three instruct models under 5B with native tool calling on a small slice of train with `--asr gold`. The winner must also fit the time budget on A30-class hardware. The candidate list is the builder's call. Qwen3-4B-Instruct is the obvious first try.
- Use the model's native tool-call chat format, so that SFT later changes as little as possible.
- Tool selection must keep the prompt within 16k tokens in the worst case (~25k characters of specs plus up to 24 history turns). Start simple, for example embedding or BM25 over app descriptions plus the transcript, keeping the top-k apps. Compressing schemas, i.e. dropping verbose fields, is allowed.
- Exit criterion: the agent runs on all of public_test with real ASR, with no sample over 30 s on molab at a safety margin (target mean < 10 s), and the result is submitted to the leaderboard.

### Tier 1: data (Track B)

- **Gold records**: all train samples outside the holdout, with ASR-cache text replacing gold transcripts. Keep a gold-text copy as well.
- **Function masking (Hammer)**: for each record, produce K variants that rename app prefixes, tool names and parameter names to random but plausible identifiers, and shuffle the tool order. Descriptions and enum values stay as they are. `name_map` lets the agent map calls back to real names before `env.validate`/`env.call`, so masked rollouts are still executable and scorable.
- **Optional teacher data (ToolACE/APIGen)**: a large LLM, allowed during development only, writes new Vietnamese requests for train apps along with expected calls. Keep a record only if its calls pass `env.validate`, replay without errors in a fresh train `ToolEnv` for the same sample state, and pass a second-model semantic check. This step is optional. Skip it if Tier 2 shows the gold and masked data are enough.
- Balance act / ask / out-of-scope roughly as in train (1217 / 217 / 111), per When2Call.

### Tier 2: LoRA SFT (Track A)

- Train on rendered trajectories, i.e. the full sequence of calls with the real tool results produced by replaying in `ToolEnv`. Compute the loss on assistant tokens only.
- Each job must fit one 12 h molab session and save a checkpoint about every 30 minutes. A checkpoint holds the adapter, optimizer, scheduler and RNG state. It is pushed to a private HF model repo, never committed. A new session resumes with `resume_from_checkpoint`. The HF repo name and token are typed in the notebook UI, like the data repo.
- Decide merge versus runtime LoRA at the Docker stage. Both are fine as long as the result is deterministic.
- **Training monitoring plots**, requested by the team so both members can analyse runs. They appear in the molab training notebook, update live, and are saved as PNG/HTML next to the checkpoint. SFT plots:
  - train/eval loss;
  - learning rate;
  - gradient norm;
  - per-checkpoint score on held-out apps and on the seen-app dev set, split by act / ask / out-of-scope;
  - share of invalid or unparsable calls;
  - mean calls per sample.

  GRPO adds:
  - mean/min/max reward per step;
  - reward components;
  - KL to the reference policy;
  - response length;
  - per-sample latency.
- Exit criterion (Gate 2): SFT beats Tier 0 on held-out apps with our ASR.

### Tier 3: GRPO on the metric (Track A), only if Gate 2 passes with ≥ 5 days left before freeze

- Start from the SFT checkpoint and run rollouts with the same agent loop in the train `ToolEnv`. The rule of one env per sample applies to inference. During training, every rollout gets its own fresh env.
- Reward: the local official score of the rollout, plus small partial credit in the ToolRL style for correct tool names and argument keys. Add a penalty per unmatched call and for exceeding the step or token budget.
- Use masked variants during RL too, so it does not undo the robustness to unseen apps.
- Exit criterion: beats SFT on held-out apps. Otherwise ship SFT.

## Evaluation

- **Leave-app-out split**: Track B picks roughly 8–10 train apps as holdout. They should be spread across domains, and some should appear in multi-app samples, so the holdout has a meaningful number of samples. All samples using a holdout app are held out from training. This set stands in for private-test apps we have not seen.
- **Seen-app dev set** *(design call)*: also hold out a random ~10% of the remaining samples, stratified by act / ask / out-of-scope. This makes SFT/RL overfitting visible and lets Tier 0 and Tier 2 be compared on samples neither was trained on.
- **Final retrain** *(design call)*: once the gates have picked the recipe, retrain the shipped model once with the same recipe and settings on all train samples, holdout and dev included. More app variety is the main defence against unseen apps. Accept it if it is not worse than the gated model on the public leaderboard and still passes the Docker time and determinism checks. Otherwise ship the gated model. Plan this run for week 3 and skip it if compute does not allow.
- Every report lists:
  - overall train score;
  - seen-app dev score and held-out-app score;
  - act / ask / out-of-scope breakdown;
  - gold-ASR and own-ASR scores;
  - mean and max seconds.

  Add the row to `RESULTS.md` as usual.
- Public leaderboard submissions confirm the trend. They are never used for tuning beyond picking between finished systems.

## Timeline

| Week | Track A (Lead) | Track B (Partner) | Gate at end of week |
|---|---|---|---|
| 1: 27 Oct – 1 Nov | Choose LLM. Tier 0 agent. First `RESULTS.md` rows. | ASR choice and backend with N-best. Train ASR cache. Holdout split and report breakdowns. Docker skeleton running `empty` offline. | **Gate 1**: Tier 0 + real ASR on public_test, submitted |
| 2: 2 – 8 Nov | SFT pipeline with checkpoint/resume. First SFT run. | Function masking and ASR-noise records. Optional teacher data. Docker running Tier 0 offline with timing. | **Gate 2**: SFT beats Tier 0 on holdout? |
| 3: 9 – 14 Nov | GRPO if Gate 2 passed and time allows, otherwise SFT iteration. Final retrain on all train data if compute allows. | Final Docker with the chosen model. Full public_test run inside Docker, run twice, outputs identical. | **Freeze 14 Nov** |
| 15 – 16 Nov | Buffer only: fixes, no new features | Submit Docker | Deadline 16 Nov 23:59 |

At every point, the best system that has passed its gate is the one packaged in Docker. A new tier replaces it only after beating it on held-out apps.

## Left to the builder

- The exact candidate LLMs and the inference engine (vLLM or HF transformers), chosen on measured latency and determinism.
- The ASR model, as long as the system stays under 6B in total.
- Prompt wording, the tool-selection method and k, the retry limit and the step budget.
- LoRA rank and hyperparameters, GRPO group size, KL weight and the reward weights.
- The exact holdout apps, following the rules above.
- The `RESULTS.md` notes format for the new breakdowns.
