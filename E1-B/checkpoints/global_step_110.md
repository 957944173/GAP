# E1-B checkpoint global_step_110

- path: `/data01/wyy/Graph-Agent-Planning/experiments/DAPO-GAP3B-MHQA-Agent-E1B-step70to120-4gpu/global_step_110`
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
- data.pt: {'exists': True, 'size': 1492, 'sha256': '569df93e6bcd3ee40c4102c80bf37eb0f1b460f712c7e7fddee32cf12932da6c'}
- merged HF present: True

## statistics (window_10step)
- n_rollouts: 130560
- n_groups: 16320
- mean_reward: 0.4570168887867649
- mean_em: 0.4569393382352941
- mean_efficiency_bonus: 0.0015510110294117646
- active_efficiency_groups: 49
- retained_groups_shaped: 333
- retained_groups_baseline_em: 285
- newly_added_by_shaping: 48
- mean_search_batches: 1.2759574142156862
- mean_queries: 1.8857996323529411
- parallel_factor: 1.5889428267936667
- mean_conversation_turns: 2.275957414215686
- mean_response_tokens: 1227.6614583333333
- zero_search_rate: 0.00012254901960784314
- zero_search_correct_count: 0
- metadata_invalid: 0

## statistics (cumulative)
- n_rollouts: 506880
- n_groups: 63360
- mean_reward: 0.4565225332754632
- mean_em: 0.4564236111111111
- mean_efficiency_bonus: 0.0019784432870370372
- active_efficiency_groups: 254
- retained_groups_shaped: 1347
- retained_groups_baseline_em: 1108
- newly_added_by_shaping: 239
- mean_search_batches: 1.3558948863636364
- mean_queries: 1.9376144255050505
- parallel_factor: 1.5509272252692317
- mean_conversation_turns: 2.355847537878788
- mean_response_tokens: 1254.5779770359848
- zero_search_rate: 0.00022095959595959597
- zero_search_correct_count: 0
- metadata_invalid: 0
