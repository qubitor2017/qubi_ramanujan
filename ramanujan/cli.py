import argparse,json
from .number_theory import is_prime,prime_factors
from .combinatorial import solve_knapsack,solve_tsp,max_cut

def main():
 p=argparse.ArgumentParser(prog='ramanujan');s=p.add_subparsers(dest='cmd',required=True)
 q=s.add_parser('prime');q.add_argument('n',type=int)
 q=s.add_parser('factor');q.add_argument('n',type=int)
 q=s.add_parser('knapsack');q.add_argument('--values',required=True);q.add_argument('--weights',required=True);q.add_argument('--capacity',type=int,required=True)
 a=p.parse_args()
 if a.cmd=='prime':out={'n':a.n,'is_prime':is_prime(a.n)}
 elif a.cmd=='factor':out={'n':a.n,'factors':prime_factors(a.n)}
 else:out=solve_knapsack([float(x) for x in a.values.split(',')],[int(x) for x in a.weights.split(',')],a.capacity).as_dict()
 print(json.dumps(out,indent=2,default=str))
if __name__=='__main__':main()
