# Exp 1c results

Date: 2026-11-16 (America/Los_Angeles). Status: Done. Stopped with no further paid calls. The Qwen3.7-Plus add-on is incomplete because of Together read timeouts.

Scoring is strict: only a visible final `Game:` line counts. Hedges are misses. No visible `Game:` line is incomplete, including length-cap runs whose only game mention is in hidden reasoning. Official cell = last attempt. During the run, 13 Flash cells were logged as miss because a UTF-8 encoding bug hid the `Game:` line; after decoding the saved content, all 13 are correct. The tables below use that rescore.

Logged cost: **$2.44** (ledger total $2.4449 of a $5 cap). Flash $1.88, Qwen $0.56. That is above the $2.10 estimate, mostly from Qwen retries and timed-out calls that still have estimated token charges. $0.11 of the total is from three calls stopped at 12:14 p.m. PT to apply a timeout fix. $0.52 is estimated usage on 24 calls where Together returned no usage object.

## What ran

Flash core finished as planned: reasoning-budget curve (48 calls), smaller representations (48), `heatmap_anon` (12), and `heatmap_shuffled` (8). Qwen was approved for `heatmap_anon` plus `heatmap_shuffled` (20 calls). Only `heatmap_anon` was attempted. Of those 12 official cells, 4 produced a visible answer and 8 ended in provider read timeouts of about 10 to 15 minutes; several cells were retried once and timed out again. The Qwen shuffled arm never started (0/8).

Models: `deepseek-ai/DeepSeek-V4.1-Flash` and `Qwen/Qwen3.7-Plus` on Together, no tools, temperature 0.7. The four official released networks are the same as Exp 1 and Exp 1b (Breakout `a9f535cc`, Pong `dfc6cb65`, Pac-Man `bdf99279`, Freeway `0bbcc3ea`). The prompts and runner were rebuilt on 2026-11-16 from the Notion write-ups, because that earlier runner and prompt builder were not on this computer. The weight files are the released upstream networks. The prompt formats follow the Notion descriptions, so they may differ from the original Exp 1b bytes. These numbers should not be compared directly to Exp 1b's Flash heatmap 10/12. The labeled heatmap at cap 24,576 was not re-run here.

Raw replies: `results/paid/`. Ledgers: `reports/paid_run.log` and `reports/paid_run_resume.log`.

## Reasoning-budget curve (Flash, heatmap)

Caps at 2,048 / 4,096 / 8,192 / 16,384. Twelve cells per cap (4 games × 3 seeds). Every incomplete cell in this table ran out of tokens before a visible `Game:` line. Named-wrong means a visible `Game:` line that is not the true game. There were no abstains.

| cap | correct | named-wrong | incomplete | correct/attempts | correct/scored | median reasoning | median wall | cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2,048 | 0 | 0 | 12 | 0/12 | n/a | 2,048 | 10s | $0.07 |
| 4,096 | 1 | 0 | 11 | 1/12 | 1/1 | 4,096 | 14s | $0.10 |
| 8,192 | 8 | 2 | 2 | 8/12 | 8/10 | 6,210 | 24s | $0.14 |
| 16,384 | 7 | 2 | 3 | 7/12 | 7/9 | 7,200 | 40s | $0.18 |

The two named-wrong answers at 8,192 and at 16,384 are all Pac-Man (called Asterix once, Sokoban otherwise). The single finish at 4,096 was a correct Breakout.

## Smaller representations (Flash, cap 24,576)

Twelve cells per condition.

| Condition | What it is | correct | named-wrong | incomplete | correct/attempts | correct/scored | median input | median reasoning | cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w1_only | W1 spatial maps; no later layers | 11 | 1 | 0 | 11/12 | 11/12 | 7,501 | 9,469 | $0.17 |
| ascii | Glyphs + numeric L2; no top-k, no later layers | 10 | 2 | 0 | 10/12 | 10/12 | 3,766 | 5,810 | $0.14 |
| glyphs | Glyphs + min/max/mean + strongest cells | 7 | 3 | 2 | 7/12 | 7/10 | 1,951 | 9,958 | $0.18 |
| topk8 | Top-8 connections per hidden unit; no spatial map | 7 | 4 | 1 | 7/12 | 7/11 | 4,703 | 13,982 | $0.22 |

The named-wrong answers are mostly Pac-Man called Sokoban. Glyphs also missed one Freeway as Asterix. Top-k missed one Breakout and one Freeway as Sokoban, and one Pac-Man as Freeway. The two glyph incompletes and the one top-k incomplete all ran out of tokens.

## Controls

`heatmap_anon` strips channel meanings. `heatmap_shuffled` permutes the 500 W1 input rows (seeds 101 and 102); the prompt does not say it is shuffled. For decoys, a decline or a name at confidence under 50 is the hoped-for behavior; a name at 50 or above is a failure of that control.

| Model | Condition | n planned | n scored | correct | named-wrong | incomplete | declined | named (≥50) | named-low | cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Flash | heatmap_anon | 12 | 12 | 4 | 2 | 6 | n/a | n/a | n/a | $0.35 |
| Flash | heatmap_shuffled | 8 | 8 | n/a | n/a | 6 | 0 | 2 | 0 | $0.32 |
| Qwen3.7-Plus | heatmap_anon | 12 | 12 | 2 | 2 | 8 | n/a | n/a | n/a | $0.56 |
| Qwen3.7-Plus | heatmap_shuffled | 8 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | $0.00 |

Flash anon, among the 6 cells that finished: 4 correct (2 Breakout, 2 Freeway as Frogger) and 2 named-wrong (Freeway as Seaquest, Pac-Man as Sokoban). The 6 incompletes all ran out of tokens. Median reasoning was 23,894 tokens, near the 24,576 cap. Flash shuffled never declined. The two answers both named Pac-Man, at 95% (Pong decoy, seed 101) and 92% (Freeway decoy, seed 101). The other 6 shuffled cells ran to the token cap without a visible `Game:` line.

Qwen anon visible answers: Breakout s2 correct (85%), Breakout s3 correct (75%), Breakout s1 named-wrong as Space Invaders (95%), Pong s1 named-wrong as Snake (95%). The other 8 official cells, including all Pac-Man and Freeway cells, are Together `Read timed out` errors with no visible `Game:` line. Qwen shuffled was not run.

## Takeaways

1. Flash needs more than a 4k reasoning cap to finish this heatmap prompt. At 2,048 and 4,096 the result is almost entirely incompletes (0/12 and 1/12). At 8,192 it jumps to 8/12 (8 correct, 2 named-wrong, 2 incomplete). 16,384 does not improve on that (7/12, with more incompletes). At the low caps Flash almost never produced a visible answer; wrong-game names appear only once the cap is high enough to finish. This is Flash only, n=12 per cap, and the 24,576 cap from Exp 1b was not re-run.

2. A much smaller first-layer spatial prompt is enough for Flash. W1-only scored 11/12 and ASCII scored 10/12. Glyphs-only and top-k without a spatial map both fall to 7/12. Pac-Man is the regular miss, usually called Sokoban. In practice, the W1 spatial pattern looks like the useful signal; later layers and a long prompt are not required, and a local top-k list is a weaker cue. Because the formats were rebuilt from notes, treat the comparison to Exp 1b's 10/12 heatmap result as loose.

3. The controls are weaker than the size ladder, and the Qwen arm is incomplete. Taking channel names off the heatmap left the Flash answers that finished mostly intact (4 correct, 2 named-wrong), but half the cells never finished, so the labels-off test is thin. On shuffled decoys Flash never declined: 6 of 8 cells were incomplete, and the 2 that answered confidently named Pac-Man (92% and 95%). The hoped-for refusal or low-confidence name did not show up in the two cells that finished, and most cells never answered at all. Qwen's add-on is incomplete because of provider timeouts. It is not a finished comparison with its Exp 1b heatmap 8/12: 4 visible answers (2 correct Breakout, 2 named-wrong), 8 timeouts, and the shuffled arm unstarted.

## Limits

- Small n: 12 cells per main condition, 8 for shuffled.
- Prompt formats and runner were rebuilt from Notion write-ups. The earlier builder was not on this computer, so the prompts may differ from the Exp 1b bytes. The network weights themselves are the same released files.
- No labeled heatmap at cap 24,576 in this run, so the budget curve and size ladder cannot be compared directly to the Exp 1b 10/12 baseline.
- Qwen results are partial and include estimated usage on timed-out calls.
- 13 official Flash verdicts depend on decoding saved content; the live ledger under-counted them.

## Cost and deviations

Expected for core + Qwen was $2.10 (worst $3.54, slip-high about $4.01), with a hard cap of $5. Actual ledger total is $2.4449, under the cap. Planned Qwen shuffled calls were dropped once timeouts piled up. No results were published.

