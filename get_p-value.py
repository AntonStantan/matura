from scipy.stats import ttest_1samp
import math

# Values taken from the notebooks, where the p-value was calculated slightly inaccurately.
benchmark_list = [3.186406532665616, 3.0491308393076566, 7.784895952475832, 5.3125465253608235, 4.522614555350064]
numeric_values = [float(i) for i in benchmark_list]

log_transform = [math.log(x) for x in numeric_values]
print(log_transform)

stats, p_value = ttest_1samp(log_transform, popmean = 0)

print(f"p-value of the benchmarks: {p_value}")