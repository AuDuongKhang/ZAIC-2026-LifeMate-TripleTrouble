# LifeMate — Evaluation

Each sample is scored on two things: whether the agent leaves the app data in the correct final state, and how closely its tool calls match the reference (gold) calls.

## Metrics

- **State match:** the submitted _state_hash_ must equal the gold _state_hash_, otherwise the sample scores 0. Free-text (fuzzy) fields do not affect the hash. The _state_hash_ must be computed by the _zaic_tools_ environment (`env.get_state_hash()`) and can be reproduced by the organisers, who replay the submitted _tool_calls_ in order with the matching version of _zaic_tools_; if the results differ, the organisers' result is final.
- **Tool-call match:** a submitted call matches a gold call when the tool name is the same, every non-fuzzy argument is identical, and every fuzzy argument has token-F1 > 0.6 (after lowercasing and Unicode NFC). Each call can match at most one call; order is not considered.
- **Out-of-scope samples:** if the gold list is empty, the sample scores 1 only when the submitted list is also empty.
- **Missing information:** `sys.ask_missing_info` is scored like any other tool.

## Formula

If the gold list is empty, the sample scores 1 only when the submitted list is also empty; otherwise it scores 0.

Otherwise (the gold list is not empty):

$$
\text{Score} =
\begin{cases}
0, & \text{if } \text{hash}_{\text{sys}} \neq \text{hash}_{\text{gold}} \text{ or } M = 0 \\[4pt]
\dfrac{2 \cdot P \cdot R}{P + R}, & \text{otherwise}
\end{cases}
\qquad
P = \frac{M}{M + U}, \quad R = \frac{M}{|G|}
$$

$$
\text{Final score} = \frac{100}{N} \sum_{i=1}^{N} \text{Score}_i
$$

Where _P_ and _R_ are precision and recall; _M_ is the number of matched calls; _|G|_ is the number of gold calls; _U_ is the number of unmatched submitted calls (including calls rejected with a schema error); _N_ is the number of samples in the test set.
