# E1-0 XML logical batch vs structured search round

| metric | value |
|---|---|
| both_valid_count | 147200 |
| exact_match_count | 147200 |
| exact_match_rate_pct | 100.0 |
| mean_abs_difference | 0.0 |
| mismatch_count | 0 |
| mismatch_rate_pct | 0.0 |
| correct_group_ranking_agreement | 8456 |
| correct_group_ranking_groups | 8456 |
| ranking_changed_group_count | 0 |
| ranking_changed_group_rate_pct | 0.0 |
| mismatch_direction | logical_search_batches >= structured_search_rounds always (a round is a distinct assistant turn, a batch is a distinct (turn, XML-block) pair), so every mismatch is the multiple-XML-blocks-in-one-turn case |
