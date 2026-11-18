# Exp 4b: reaching the paper's car result with open models (Nov 16–17, 2026)

Goal: an open-weight model edits the Kinetix car network so it passes the paper's test as fully as Astra's released edit (1024/1024).

Result: met with assistance. DeepSeek-V4.1-Flash, given Python tools, a physics simulator, a compressed scene view, and feedback, produced `runs/repair_flash_1/submission_t13.npz`: 1024/1024 on the held-out test, 2,957 weights changed (1.72%). The strict no-tools one-shot replication still failed for every open model.

| Run | Model | Setup | Held-out test | Weights changed | Cost |
|---|---|---|---:|---:|---:|
| raw1shot_flash_32k | Flash | strict 1:1, 32k cap | 0/1024 | 386 | $0.124 |
| raw1shot_kimik3_32k | Kimi K3 | strict 1:1, 32k cap | 0/1024 | 386 | $1.340 |
| raw1shot_qwenmax_32k | Qwen3.7-Max | strict 1:1, 32k cap | 0/1024 | 1,275 | $1.318 |
| rawfb_flash_32k | Flash | raw weights + 4 feedback rounds | 0/1024 | 386 (scored; a later edit of 165,102 was not the best) | $0.117 |
| comp_fb_flash_32k | Flash | compressed view + feedback, no tools | 1022/1024 | 165,102 (96%) | $0.060 |
| tools_flash_1 | Flash | tools + simulator, no change limit | 1024/1024 | 165,103 (96%) | $0.105 |
| repair_flash_1 | Flash | tools + simulator + feedback, repair limit 8,600 | 1024/1024 | 2,957 (1.72%) | $0.064 |
| repair_flash_5 | Flash | repeat of repair setup | 913/1024 | 7,751 (4.5%) | $0.111 |
| repair_flash_2/3/4 | Flash | repeat attempts | no submission | n/a | $0.072 total ($0.0382 / $0.0113 / $0.0229) |
| repair_qwenplus_1 | Qwen3.7-Plus | repair setup | no submission, timeouts | n/a | $0.172 incl. reserve |

Strict runs: 7 runs, 7 Together model IDs, all 0/1024. Exp 4 (logs on the old box): DeepSeek-V4.1-Flash-1017, Qwen3.7-Plus, DeepSeek-V4-Pro-0813, Muse-Spark-Open-816B-A42B. Exp 4b: DeepSeek-V4.1-Flash, Kimi-K3, Qwen3.7-Max. That is 6 IDs if Flash and Flash-1017 count as one.

Verification (repair_flash_1): the network scored 1024/1024 on fresh seed 424242, 1024/1024 on fresh seed 99, 1024/1024 on fresh seed 87654 using the environment's collision flag, and greedy success 1/1. The sandbox mounted only the shown prompt and weights. It could not see Astra's answer or edited_actor.npz. Overlap with Astra's edit is 1,298 changed positions, mostly where both cleared units 126 and 127. Only 7 identical nonzero values overlap.

Repeat runs: 10 attempts of the winning assisted setup (tools, simulator, feedback, 8,600-weight cap). 4 solved (repeat_seq_03, 2,932 weights, units 0/1; seq_05, 2,297, units 10/11; seq_07, 2,312; seq_09, 386, head only), 0 partial, 1 scored but failed (seq_10, 0/1024, 1 weight changed), and 5 with no submission (repeat_flash_01 hit the $0.10 run cap after 22 turns and never called save_edit; seq_02, 04 and 06 hit the $0.15 cap; seq_08 used all 30 turns and 30 sim passes and never saved an edit). Killed orphans (repeat_flash_02/03/09/10 and the first seq_08 try, cut off by a box restart) are not counted as attempts. Fresh-seed checks of the four successes: 03, 05 and 07 scored 1024/1024 on seeds 424242, 99 and 87654 (collision flag); 09 scored 1022, 1022 and 1024. All four had greedy success 1.0.

Controls (no language model, same 8,600-weight cap, same 1,024-episode strict test):
- Random sparse hill-climb: 0/1024 on seeds 1–3 (120 calls) and seed 4 (480 calls). Best training score in every seed was 0/64.
- Greedy unit-rewire search (take over two hidden units and rewire the motor): 1 of 3 seeds at 120 calls scored 1023/1024 (greedy_s3, 2,546 weights, units 10 and 103; first full training score at call 70). Seeds 1 and 2 at 120 calls scored 0/1024. Longer runs: seed 5 at 240 calls scored 920/1024 (best training score 57/64, 2,541 weights); seeds 4 (480 calls) and 6 (240 calls) scored 0/1024. So 1 of 6 greedy seeds reached 1023/1024, and one more reached 920/1024.

Leakage check: the same assisted setup, on a function-preserving hidden-unit permutation of the shown network (networks/perm_shown.npz, seed 31337; max logit difference vs the base 1.2e-5), with the prompt's edit-language example changed from `actor0.W[6, 122]` to `actor0.W[17, 45]` and no unit hints. All 3 attempts solved 1024/1024: leak_perm_01, 5 weights, $0.0875, head only, new unit 105 (= original actor4 unit 110); leak_perm_02, 386 weights, $0.0828, head only, 15 units; leak_perm_03, 386 weights, $0.1168, head only, new units 11/73/105 (= original 104/7/110). None used the permuted images of units 126/127 (actor4 images: new units 7 and 16).

Costs: Exp 4b ledger $5.184 over 438 calls, under the $6.00 ceiling inside the $10 budget. The 10-repeat median was $0.143 per attempt (range $0.071–$0.161), or $0.353 per success (total repeat-tagged spend $1.414 / 4, including $0.138 on killed orphans that do not count as attempts). repair_flash_1, the first winning run, cost $0.064. The $1.356 "run cost" in the repair_flash_2 log is wrong: an older runner used ledger difference and included the Kimi K3 and Qwen3.7-Max strict calls that were running in parallel; the ledger tag and summary are $0.0382. Total project (Exp 1 + 1b + 1c + 2 + 4 + 4b) $11.67. Same-token repricing at gpt-6-astra and Claude Fable 5.2 prices is in reports/COSTS.md.

Practical meaning: an open model can write a working weight edit for this car network when it can test ideas. It did not reproduce Astra's unaided one-shot ability. The working repair is blunt. It takes over two units and rewires the motor output. A simple non-LLM greedy search also found that kind of edit (1023/1024) in 1 of 6 seeds, so the model is not finding something a simple search cannot. The trained network in the first Flash repair matches its original motor choice on 30% of steps (Astra's edit: 33%).

Limits: early assisted attempts were 1 full success out of 6 (five Flash and one Qwen3.7-Plus), plus one partial (913/1024) and four with no usable submission. The 10-repeat batch was 4 of 10. The harness fixed parser issues during the early runs. The simulator is a major assist. Kimi K3 and Qwen3.7-Max were tested only on strict 1:1. No GitHub push or publishing.

Scratch dry-runs (`_sbtest`, `mock_dryrun`, `perm_mock_dryrun`) and killed orphan run directories are not included in this copy. They are noted above and were not counted as attempts.
