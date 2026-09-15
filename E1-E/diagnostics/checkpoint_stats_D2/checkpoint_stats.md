# E1-E per-checkpoint statistics

run dir: `/data01/wyy/Graph-Agent-Planning/E1-E/runs/D2_20260914_165545`

steps observed: [71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]

## last-10-step window

| checkpoint | rollouts | groups | mean reward | mean EM | mean E | active E groups | retained (shaped) | retained (EM) | search batches | queries | parallel factor |
|---|---|---|---|---|---|---|---|---|---|---|---|
| global_step_80 | 11520 | 1440 | 0.4793 | 0.4790 | 0.0069 | 0 | 0 | 0 | 1.5000 | 2.0682 | 1.5124 |
| global_step_90 | 3840 | 480 | 0.5037 | 0.5026 | 0.0219 | 0 | 0 | 0 | 1.7229 | 2.3185 | 1.4882 |
| global_step_100 | 11520 | 1438 | 0.4780 | 0.4774 | 0.0109 | 0 | 0 | 0 | 1.5244 | 2.1072 | 1.5223 |
| global_step_110 | 8960 | 1120 | 0.4837 | 0.4833 | 0.0083 | 0 | 0 | 0 | 1.5013 | 2.0869 | 1.5271 |
| global_step_120 | 7680 | 960 | 0.5125 | 0.5118 | 0.0128 | 0 | 0 | 0 | 1.4833 | 2.0651 | 1.5244 |

## cumulative (step71..checkpoint)

| checkpoint | rollouts | groups | mean reward | mean EM | mean E | active E groups | retained (shaped) | retained (EM) | search batches | queries | parallel factor |
|---|---|---|---|---|---|---|---|---|---|---|---|
| global_step_80 | 11520 | 1440 | 0.4793 | 0.4790 | 0.0069 | 0 | 0 | 0 | 1.5000 | 2.0682 | 1.5124 |
| global_step_90 | 15360 | 1916 | 0.4854 | 0.4849 | 0.0106 | 0 | 0 | 0 | 1.5557 | 2.1308 | 1.5063 |
| global_step_100 | 26880 | 3339 | 0.4822 | 0.4817 | 0.0108 | 0 | 0 | 0 | 1.5423 | 2.1207 | 1.5132 |
| global_step_110 | 35840 | 4447 | 0.4826 | 0.4821 | 0.0101 | 0 | 0 | 0 | 1.5321 | 2.1122 | 1.5167 |
| global_step_120 | 43520 | 5396 | 0.4879 | 0.4873 | 0.0106 | 0 | 0 | 0 | 1.5235 | 2.1039 | 1.5180 |

