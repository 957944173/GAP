# E1-B checkpoint global_step_120

- path: `/data01/wyy/Graph-Agent-Planning/experiments/DAPO-GAP3B-MHQA-Agent-E1B-step70to120-4gpu/global_step_120`
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
- data.pt: {'exists': True, 'size': 1492, 'sha256': '2afc69dfa0ced1208ff258603db8d2737c93829edde0d8c9e19cd566e4d1b150'}
- merged HF present: True

## statistics (window_10step)
- n_rollouts: 134400
- n_groups: 16800
- mean_reward: 0.4596164434523812
- mean_em: 0.45955357142857145
- mean_efficiency_bonus: 0.0012574404761904762
- active_efficiency_groups: 37
- retained_groups_shaped: 338
- retained_groups_baseline_em: 304
- newly_added_by_shaping: 34
- mean_search_batches: 1.2334002976190477
- mean_queries: 1.8439955357142856
- parallel_factor: 1.5969610922551705
- mean_conversation_turns: 2.2334002976190477
- mean_response_tokens: 1206.5610044642858
- zero_search_rate: 0.00035714285714285714
- zero_search_correct_count: 0
- metadata_invalid: 0

## statistics (cumulative)
- n_rollouts: 641280
- n_groups: 80160
- mean_reward: 0.45717095756403886
- mean_em: 0.45707959081836325
- mean_efficiency_bonus: 0.0018273349135063208
- active_efficiency_groups: 291
- retained_groups_shaped: 1685
- retained_groups_baseline_em: 1412
- newly_added_by_shaping: 273
- mean_search_batches: 1.3302223677644711
- mean_queries: 1.9179937000998004
- parallel_factor: 1.5605740029353392
- mean_conversation_turns: 2.3301849426147703
- mean_response_tokens: 1244.5145396706587
- zero_search_rate: 0.000249500998003992
- zero_search_correct_count: 0
- metadata_invalid: 0
