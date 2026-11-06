import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ramanujan import solve_linear, solve_knapsack, max_cut
print('linear:',solve_linear([[2,1],[1,3]],[5,6]).as_dict())
print('knapsack:',solve_knapsack([6,10,12],[1,2,3],5).as_dict())
print('maxcut:',max_cut(4,[(0,1),(1,2),(2,3),(3,0),(0,2)]).as_dict())
