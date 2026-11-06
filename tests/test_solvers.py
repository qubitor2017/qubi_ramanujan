import unittest,math
import ramanujan as r
class SolverTests(unittest.TestCase):
 def test_linear(self):self.assertAlmostEqual(r.solve_linear([[2,1],[1,3]],[5,6]).solution[0],1.8)
 def test_qr(self):self.assertAlmostEqual(r.qr_least_squares([[1,0],[1,1],[1,2]],[1,3,5]).solution[1],2)
 def test_roots(self):self.assertAlmostEqual(r.bisection(lambda x:x*x-2,0,2).solution,math.sqrt(2),places=7)
 def test_integral(self):self.assertAlmostEqual(r.integrate_simpson(lambda x:x*x,0,1),1/3,places=8)
 def test_ode(self):self.assertAlmostEqual(r.rk4(lambda t,y:[y[0]],0,[1],1,100).solution[-1][1][0],math.e,places=6)
 def test_optim(self):self.assertAlmostEqual(r.gradient_descent(lambda x:(x[0]-3)**2,lambda x:[2*(x[0]-3)],[0],lr=.1).solution[0],3,places=4)
 def test_graph(self):self.assertEqual(r.dijkstra({'a':{'b':2},'b':{}},'a').solution['distance']['b'],2)
 def test_knapsack(self):self.assertEqual(r.solve_knapsack([6,10,12],[1,2,3],5).objective,22)
 def test_tsp(self):self.assertEqual(r.solve_tsp([[0,1,9],[1,0,2],[9,2,0]]).objective,12)
 def test_number(self):self.assertTrue(r.is_prime(104729));self.assertEqual(r.prime_factors(84),[2,2,3,7])
 def test_crt(self):self.assertEqual(r.chinese_remainder([2,3,2],[3,5,7])[0],23)
 def test_complex(self):self.assertAlmostEqual(r.complex_gaussian_solve([[1+1j]],[2+2j]).solution[0].real,2)
 def test_inverse(self):self.assertAlmostEqual(r.tikhonov_least_squares([[1],[1]],[2,2],1e-8).solution[0],2,places=5)
 def test_info(self):self.assertAlmostEqual(r.entropy([.5,.5]),1)
if __name__=='__main__':unittest.main()
