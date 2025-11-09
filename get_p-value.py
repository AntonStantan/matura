from scipy.stats import ttest_1samp
import math

#values taken from the notebooks, where the p-value was calculated slightly inaccurately.
benchmark_list = ['10.600075', '12.701344', '10.435802', '10.592991', '9.183982']
[int(i) for i in benchmark_list]

log_transform = math.log(benchmark_list)

print(log_transform)

stats, p_value = ttest_1samp(benchmark_list, popmean = 0)

print(f"p-value of the benchmarks: {p_value}")