import math
import unittest

class IndustrialTests(unittest.TestCase):
    def setUp(self):
        try:
            import numpy, scipy
        except ImportError:
            self.skipTest("NumPy/SciPy not installed")
        from ramanujan import (solve_linear_system, solve_lp, solve_milp, solve_nonlinear_root,
                               integrate_quad, solve_ode, tikhonov, Ramanujan)
        self.api=(solve_linear_system, solve_lp, solve_milp, solve_nonlinear_root, integrate_quad, solve_ode, tikhonov, Ramanujan)
    def test_dense_linear_residual(self):
        r=self.api[0]([[3.,1.],[1.,2.]],[9.,8.]); self.assertTrue(r.success); self.assertLess(r.residual,1e-10)
    def test_lp(self):
        r=self.api[1]([ -3., -2.], A_ub=[[1.,1.],[1.,0.],[0.,1.]], b_ub=[4.,2.,3.], bounds=[(0,None),(0,None)])
        self.assertEqual(r.status,"optimal"); self.assertAlmostEqual(r.objective,-10.,places=6)
    def test_milp(self):
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint
        c=[-3.,-2.]
        cons=LinearConstraint([[1.,1.]],[0.],[3.])
        r=self.api[2](c, integrality=[1,1], bounds=Bounds([0,0],[3,3]), constraints=cons)
        self.assertEqual(r.status,"optimal"); self.assertTrue(r.success)
    def test_nonlinear_root(self):
        r=self.api[3](lambda x:[x[0]**2-2.],[1.])
        self.assertTrue(r.success); self.assertLess(r.residual,1e-7)
    def test_quad(self):
        r=self.api[4](lambda x:x*x,0.,1.); self.assertAlmostEqual(r.solution,1/3,places=8)
    def test_ode(self):
        r=self.api[5](lambda t,y:-y,(0.,1.),[1.],t_eval=[0.,1.]); self.assertTrue(r.success); self.assertAlmostEqual(r.solution['y'][0][-1],math.exp(-1),places=4)
    def test_tikhonov(self):
        r=self.api[6]([[1.,0.],[0.,1.],[1.,1.]],[1.,2.,3.],regularization=1e-8); self.assertTrue(r.success); self.assertLess(r.residual,1e-3)
    def test_lp_infeasible_status(self):
        r=self.api[1]([1.], A_ub=[[1.],[-1.]], b_ub=[0.,-1.], bounds=[(None,None)])
        self.assertEqual(r.status,"infeasible"); self.assertFalse(r.success)
    def test_sparse_solver(self):
        import scipy.sparse as ss
        from ramanujan import solve_sparse_linear_system
        r=solve_sparse_linear_system(ss.eye(3,format="csr"),[2.,3.,4.])
        self.assertTrue(r.success); self.assertEqual(r.solution,[2.,3.,4.])
    def test_complex_linear_system(self):
        r=self.api[0]([[1+1j,0],[0,2-1j]],[2+2j,2-1j])
        self.assertTrue(r.success); self.assertAlmostEqual(abs(r.solution[0]-2),0.,places=8)
    def test_facade(self):
        api=self.api[7](); self.assertIn("optimisation.lp",api.available_solvers())

if __name__ == "__main__": unittest.main()
