import numpy as np 
import pandas as pd 
from collections.abc import Hashable

print(isinstance("no_ops", Hashable))
# n_states_6 = [-24.10542593139329,-23.42322274430872, -30.25212157223115,-23.07387727085122, -22.855473735190806] 
# #[-24.10542593139329, -23.42322274430872]
# # [-23.432169840574453, -24.70643724698869, -24.61863174413091, -22.395372185205854, -22.465909692790895]
# n_states_5 = [-25.39271022169258, -25.134310266830813, -24.009501640247713, -23.593880476787557, -23.31005141154891]
# n_states_7 = [-23.99918504006178, -24.839157442748636, -26.45228727947142, -23.49605994363116, -23.990978644987013]
# n_states_4 = [-25.08335404524726, -24.504601023166725, -24.414900929208063, -26.589390944333175, -23.71076975934144]
# n_states_8 = [-25.367415386720435, -27.361826320066314, -23.831535480715278, -23.111846299018513]


# print(np.mean(n_states_4))
# print(np.mean(n_states_5))
# print(np.mean(n_states_6))
# print(np.mean(n_states_7))
# print(np.mean(n_states_8))

# print(np.isin(['a', 'b'], ['c,d']))
test = set(np.array(['b','c']).tolist())
s1 = [np.array(['a']), np.array(['b']), np.array(['b', 'c']), np.array(['a','b', 'c'])]
row_condition = [set(i).issubset(test) for i in s1]
# row_condition = [np.isin(i, test).any() and not(np.isin(i,test, invert = True).any()) for i in s1]

print(row_condition)
abc = [0].extend(['a', 'b'])
print(abc)
# s2 = pd.Series(['c','d'])
# print(pd.Series(list(set(s1).intersection(set(s2)))))