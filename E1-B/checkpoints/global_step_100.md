# E1-B checkpoint global_step_100

- path: `/data01/wyy/Graph-Agent-Planning/experiments/DAPO-GAP3B-MHQA-Agent-E1B-step70to120-4gpu/global_step_100`
- tracker (`latest_checkpointed_iteration.txt`): 120
- actor/model_world_size_4_rank_0: 3397345082 bytes
- actor/optimizer_world_size_4_rank_0: 6171911046 bytes
- actor/extra_state_world_size_4_rank_0: 14632 bytes
- actor/model_world_size_4_rank_1: 3397345082 bytes
- actor/optimizer_world_size_4_rank_1: 6171911046 bytes
- actor/extra_state_world_size_4_rank_1: 14696 bytes
- actor/model_world_size_4_rank_2: 3397345082 bytes
- actor/optimizer_world_size_4_rank_2: 6171911046 bytes
- actor/extra_state_world_size_4_rank_2: 14696 bytes
- actor/model_world_size_4_rank_3: 3397345082 bytes
- actor/optimizer_world_size_4_rank_3: 6171911046 bytes
- actor/extra_state_world_size_4_rank_3: 14760 bytes
- data.pt: {'exists': True, 'size': 1492, 'sha256': '3c88daef01c6a11dbf9c155e5200692d47f6a131c77d63303ff4bae9d141e7ec'}
- merged HF present: True

## statistics (window_10step)
- n_rollouts: 124160
- n_groups: 15520
- mean_reward: 0.45695795747422707
- mean_em: 0.4568460051546392
- mean_efficiency_bonus: 0.002239046391752577
- active_efficiency_groups: 69
- retained_groups_shaped: 341
- retained_groups_baseline_em: 275
- newly_added_by_shaping: 66
- mean_search_batches: 1.3330299613402061
- mean_queries: 1.9256201675257731
- parallel_factor: 1.5624045723922466
- mean_conversation_turns: 2.3329655283505155
- mean_response_tokens: 1248.1319346005155
- zero_search_rate: 0.0002577319587628866
- zero_search_correct_count: 0
- metadata_invalid: 0

## statistics (cumulative)
- n_rollouts: 376320
- n_groups: 47040
- mean_reward: 0.4563510221797055
- mean_em: 0.45624468537414964
- mean_efficiency_bonus: 0.0021267361111111114
- active_efficiency_groups: 205
- retained_groups_shaped: 1014
- retained_groups_baseline_em: 823
- newly_added_by_shaping: 191
- mean_search_batches: 1.3836282950680272
- mean_queries: 1.9555909863945578
- parallel_factor: 1.5377363903267354
- mean_conversation_turns: 2.3835645195578232
- mean_response_tokens: 1263.9163610756802
- zero_search_rate: 0.00025510204081632655
- zero_search_correct_count: 0
- metadata_invalid: 0
