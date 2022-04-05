from collections import defaultdict
import os

number_of_configurations = 5
number_of_folds = 5
number_of_repetitions = 5

processed_results = defaultdict(lambda: {})
for result in os.listdir('./results'):
    domain = result[:-4]
    result_data = [float(line.split(',')[-1]) for line in open(f'./results/{result}').read().split('\n')[1:-1]]
    for i in range(number_of_configurations):
        i_summed_results = 0
        for j in range(number_of_folds):
            for k in range(number_of_repetitions):
                i_summed_results += result_data[k * number_of_configurations * number_of_folds + i * number_of_folds + j]
        i_average_results = i_summed_results / (number_of_configurations * number_of_folds)
        processed_results[f'Config. {i+1}'][domain] = f'{i_average_results:.2f}'

print('\n\n'.join(map(str, processed_results.items())))