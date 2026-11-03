# Error analysis: where sft-final loses points

Checked on 3 Nov 2026, when the team's public_test score was **53.58** (sft-final + check_refs, pho-large-ft-b). The analysis
uses the runs below together with `data/train` and `data/public_test`. Vietnamese strings are data: requests, gold
values and model outputs.

| Run | Split / subset | Score |
|---|---|---|
| `20261102-093954_3c4fd00_train_sft-v2_gold_dev_check_refs` | train dev, gold text | 72.86 |
| `20261102-101746_3c4fd00_train_sft-v2_pho-large-ft-b_dev_check_refs` | train dev, ASR | 66.86 |
| `20261102-092303_3c4fd00_train_sft-v2_gold_holdout_check_refs` | train holdout, gold text | 50.96 |
| `20261102-104252_3c4fd00_train_sft-v2_pho-large-ft-b_holdout_check_refs` | train holdout, ASR | 41.37 |
| `20261103-012737_07e28c6_public_test_sft-final_pho-large-ft-b_check_refs` | public_test (no labels) | 53.58 (leaderboard) |

The holdout subset contains every train sample that touches one of 8 held-out apps (`appliances`, `bike_rental`,
`calculator`, `exchange_rate`, `home_energy`, `loyalty`, `news`, `subscriptions`). It is our stand-in for unseen apps.

## 1. Summary

- Public (53.58) falls between dev (66.86, seen apps) and holdout (41.37, unseen apps). **268 of 634 public samples
  (42%) call at least one of the 11 public-only apps** (`ecommerce`, `event_tickets`, `grocery`, `habit_tracker`,
  `pantry`, `parking`, `public_transit`, `recipes`, `stocks`, `train_bus_tickets`, `translator`). Generalising to
  unseen apps is therefore the main lever, more than polishing the apps the model has already seen.
- The largest losses on unseen apps come from **free-text arguments** and **read-call conventions** (wrong read tool,
  extra optional filters). These samples keep a correct state hash but lose F1.
- On public there is a **date bug** that costs whole samples: the prompt gives no weekday, so "thứ X tuần sau" is
  resolved wrong. The search then returns empty, the agent repeats the same call and stops as `invalid`.
- Ask/act decisions go wrong in both directions about equally often, and the model wrongly refuses some requests that
  belong to unseen apps.

## 2. Point loss by category

Each sample that is not fully correct is placed in exactly one bucket. "Points" is the score lost, on the 0–100 scale
of the whole subset.

| Category | dev, gold | dev, ASR | holdout, gold | holdout, ASR |
|---|---|---|---|---|
| Hash OK, F1 < 1 (unmatched reads / free text) | 6.03 | 6.81 | **17.90** | **16.54** |
| Hash fail: same write tools, wrong arguments | 8.33 | 11.67 | 11.11 | 13.47 |
| Gold asks, agent acted | 3.11 | 2.33 | 5.72 | 5.39 |
| Agent asked, gold acts | 1.67 | 2.67 | 3.54 | 6.73 |
| Hash fail: missing write | 2.67 | 4.00 | 4.04 | 5.39 |
| Hash fail: extra write | 1.67 | 2.33 | 4.04 | 4.71 |
| Hash fail: different write tools | 1.33 | 1.67 | 1.35 | 4.38 |
| Out-of-scope sample, agent made calls | 2.00 | 1.33 | 0.34 | 0.67 |
| No calls at all (gold act/ask) | 0.33 | 0.33 | 1.01 | 1.35 |

The table below types the wrong arguments on calls whose tool name matches gold (ASR runs). Buckets overlap, so read
each number as an upper bound on what fixing that bucket alone could recover.

| Argument error | dev, ASR | holdout, ASR |
|---|---|---|
| Free text (`x-match: fuzzy`) token-F1 ≤ 0.6 | 5.68 | **15.14** |
| Other exact string (query, phone number, e-mail, place) | 7.70 | 8.60 |
| Wrong id | 6.50 | 6.75 |
| Missing key | 3.31 | 6.66 |
| Extra optional key | 3.78 | 5.39 |
| Wrong number | 2.67 | 4.55 |
| Wrong date | 1.00 | 3.70 |
| Wrong time of day | 2.33 | 3.03 |

## 3. Findings

### A. Free-text arguments are the largest loss on unseen apps

Fuzzy keys (`title`, `content`, `body`, `subject`, `name`, ...) are left out of the hash, so a miss here costs only
F1. The bar is still high: gold values are short (median 7 tokens), and a short sentence easily drops below token-F1
0.6. Gold follows a consistent style. It copies concrete facts from tool results word for word: article titles,
numbers, units.

| Call | Gold | Ours |
|---|---|---|
| `tasks.add.title` (train_0506) | "Đọc bài Logistics xanh: kho hàng Bắc Ninh giảm phát thải nhờ xe tải điện" | "Đọc bài logistics xanh" |
| `reminders.create.content` (train_0537) | "Đọc bài Mái nhà dùng pin mặt trời" | "Đọc bài mới nhất về điện mặt trời" |
| `shopping_list.add` (train_0505) | `{"name": "tấm vải 500 cm", "quantity": 1}` | `{"name": "tấm vải dài 500cm"}` |
| `home_automation.create_rule.title` (train_0696) | "61.9 kWh" | "Tắt đèn biển hiệu khi hết 61.9 kWh tuần" |

Over 92 free-text misses on dev and holdout, our value was shorter than gold 47 times and longer 31 times.

**Idea:** in the prompt and in the few-shot examples, ask for a short phrase that copies names and numbers verbatim
from the tool results and from the request. Check whether `sft.records` already shows enough of that style.

### B. Read calls: the gold convention is "minimal arguments" and "the right read tool"

In train gold, **60% of read calls have no arguments** and **76% pass only required parameters**. Our agent adds
optional filters that gold leaves out:

- `appliances.list {"type": "dishwasher"}`, gold `{}` (train_0450, 0478, 0545, 0704)
- `bike_rental.find_bikes {"near": "Cổng Tây"}`, gold `{}` (train_0616)
- `smart_home.get_status {"room": "phòng khách"}`, gold `{}` (train_0637)

It also picks the wrong read tool: `news.headlines` where gold uses `news.search {"query": ...}` whenever the user
names a topic (14 missing `news.search` calls on holdout). It skips reads that gold makes, e.g.
`calculator.calculate` before writing a computed value (train_0805).

Optional read parameters that gold does use: `contacts.search.query` (204 times), `gold_price.get_price.type`,
`calendar.list_events.date_from/date_to` (18 times, but omitted 87 times).

Dropping only the unmatched extra reads on hash-OK samples would add **+2.2 to +2.6 points** on holdout and about
+0.7 to +0.9 on dev.

**Idea:** a rule in the prompt or the training data: use the search/query tool when the user names a topic or
item, and leave optional filters out unless the user states them.

### C. Dates: the prompt has no weekday

`agents/prompting.py` passes `current_time` as an ISO string only. A 4B model has to work out the weekday on its own
and often gets it wrong. In train gold, "thứ X tuần sau" means weekday X **of the next calendar week**:

| Sample | Now | Phrase | Gold date |
|---|---|---|---|
| train_0633 | 2026-10-01 Thu | thứ ba tuần sau | 2026-10-06 |
| train_0843 | 2026-10-05 Mon | thứ sáu tuần sau | 2026-10-16 |
| train_1435 | 2026-12-04 Fri | thứ sáu tuần sau | 2026-12-11 |

Failures on public:

| Sample | Now | Request | Our date | Expected |
|---|---|---|---|---|
| public_0187 | 2026-09-24 Thu | "thứ bảy tuần sau" | 2026-09-26 | 2026-10-03 |
| public_0287 | 2026-10-01 Thu | "sáng thứ bảy tuần sau" | 2026-10-03 | 2026-10-10 |
| public_0093 | 2026-11-16 | "ngày mồng bốn tháng mười hai" | 2026-11-24 | 2026-12-04 |

In all three, `flights.search` returns `[]` and the sample ends with no write at all, so it scores 0 (see D).

**Idea:** add the weekday and a small date table to the context (today, mai, mốt, "thứ X này", "thứ X tuần sau"),
computed in Python from `current_time`. The SFT model was trained on the current prompt, so re-render the records
and retrain rather than only changing the prompt at inference.

### D. No recovery from empty results

On public, 6 samples stop as `invalid` and 2 hit `max_steps`. When a search returns empty, the agent proposes the
same call again. The guard rejects it ("already called"), and after the retries the agent gives up. Counts of
validator rejections on public:

| Rejection | Count |
|---|---|
| `flights.search: already called` | 10 |
| `event_tickets.search: already called` | 7 |
| `event_tickets.book: unknown id ...` (invented ids) | 7 |
| `banking.transfer: unknown id from_account_id=...` | 4 |
| `train_bus_tickets.search: already called` | 2 |

**Idea:** when a read returns an empty list, answer the model with a hint instead of a bare rejection. For example:
"no results: try another date (±7 days), fewer filters, or a shorter query". If the second attempt is still empty,
fall back to `sys.ask_missing_info` or stop cleanly. A wrong date (C) plus no recovery (D) loses the whole sample.

### E. Wrong refusals on unseen apps

On public, 49 of 634 submissions are empty (7.7%, close to the train prior of 7.2%). Many are correct: the user
cancels themselves ("à thôi khỏi"), the app is not installed, or the request is unsafe. About 10, though, look like
refusals of supported requests:

| Sample | Request (ASR) | Likely tool |
|---|---|---|
| public_0590, 0609, 0662, 0674 | "kết thúc phiên gửi xe ..." (noisy ASR: "ghi xe", "lượt đồ") | `parking.end_session` |
| public_0265 | "đánh dấu hôm nay hoàn thành bài tập thở" | `habit_tracker.log` |
| public_0194 | "đặt mua cho tôi sữa tươi ít đường" | `grocery.search` + `add_to_cart` |
| public_0466 | "món bò kho ... cất vô mục đã lưu" | `recipes.save` |
| public_0598 | "lưu giùm tui quán Bún bò ..." | `maps.save_place` |
| public_0654 | "đặt cảnh báo BrightBolt Holding xuống dưới ba mươi nghìn" | `stocks.set_price_alert` |
| public_0043 | "vàng SJC ... dưới 132 triệu thì sáng mai nhắc" | `gold_price.get_price` (+ reminder) |

That is roughly 1.5 points. The refusal text is always "Xin lỗi, tôi không làm được việc này.", which the model
learned from the out-of-scope samples in train.

**Idea:** before accepting a refusal, check whether an installed tool's name or description overlaps the request.
If one does, run one more step that hints at that tool. Masked/perturbed SFT copies with more unusual tool names
should also help.

### F. Ask vs act goes wrong in both directions

On holdout (ASR) there are 16 "should ask, acted" and 20 "asked, should act" samples, about 12 points together.
What gold does:

- **Ask** when a read returns several candidates, or when the user has not said a required value. Example: train_1075
  "cứ hạ cái cảnh báo xuống" gives no threshold, so gold asks for `threshold_kwh`; we set 200.
- **Act** when exactly one record matches the description. In train_1160 (cancel the Tuesday-next-week flight) and
  train_1171 (cancel the subscription renewing next Friday), we asked although the read results pin down one record.
- Ask also appears at the end of a multi-step request, after the earlier steps are done (train_1123: climate, alarm,
  then ask for the TV `channel`).

**Idea:** count the candidates that match after the read and decide by rule. Revisit `agents/ask_guard.py` with this
in mind.

### G. ASR errors

The ASR costs −6.0 points on dev and −9.6 on holdout (paired comparison in `RESULTS.md`). Typical errors:

- Digits in phone numbers: `09887755447` becomes `0987755447`, `09012345677` becomes `19002345677`.
- Lexical: "Xoan" for "Soạn", "váy" for "vé", "bangkok", "Toronto Tower" for a parking lot name, "chuyệt" for "duyệt".

pho-large-ft-b was fine-tuned on the pool clips, which may share speakers with dev and holdout. Public speakers may
differ, so the real public WER is probably above the 0.08–0.13 measured on train.

**Ideas:** check spoken phone numbers against the contacts results; match spoken names fuzzily against
`contacts.search` results; keep the ASR-noised SFT copies.

### H. Smaller issues

- **"Đã xong" leaks into arguments.** It shows up as `content` or `sms.send.content` in public_0327, 0441, 0578
  (and train_1204, 1236, 1442, 1578). It is probably the fixed final text from `ReplayLLM` in the SFT records. A
  post-check should reject a free-text argument equal to the final-answer text.
- **Gold noise and idiosyncrasies cap the ceiling.** Examples: train_0724 uses `home_energy.get_usage {"period":
  "week"}` for "hôm qua". `maps.get_route.origin` is sometimes the literal "nhà" (2 times) rather than the profile
  address (17 times). `flights.search.origin` is mostly an IATA code (`HAN` 6 times) but once "Đà Nẵng".
- **Out of scope in train (111 samples).** 44 are self-cancellations ("à thôi khỏi"). The rest are apps that are not
  installed, unsafe requests, chit-chat, or questions already answered in the history. Note that 37 train samples
  contain "à thôi" and still have gold calls, because the user switches to a different action.
- **Contacts.** `contacts.search.query` keeps kinship or role words when the stored contact name has them ("Sếp
  Dũng", "Cô giáo Mai", "Bố"): 73 of 204 queries.

## 4. Suggested order

1. **Dates (C) and empty-result recovery (D).** Cheap, deterministic, and each failure costs a whole sample.
2. **Read conventions (B).** Prefer the search/query tool for named topics and drop optional filters. Measurable on
   holdout right away.
3. **Free-text style (A).** Short phrases copying names and numbers from tool results.
4. **Ask/act (F) and refusals on unseen apps (E).** Rules after the read results, plus more mask/perturb data.
5. **ASR-aware fixes (G)** for phone numbers and contact names.

Report each change on dev and holdout with `python -m lifemate.compare`, as usual. Holdout is the closest proxy for
the public-only and private-test apps.

## 5. Note on the Docker config

On the `feat/docker` branch at the time of writing, `configs/shipped.json` points to `agent_preset: empty` and
`asr_preset: none`. Set it to the sft-final presets before building the final image.
