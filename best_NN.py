import pandas as pd
results = pd.read_csv('gridsearch.csv')

results = results.sort_values('r2_test',ascending = False)

print(results.head(n=10))

# import ast

# results = pd.read_csv('gridsearch.csv', engine='python')
# results['layers'] = results['layers'].apply(ast.literal_eval)

# print(results)