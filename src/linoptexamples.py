"""
Test examples for the linopt module.
"""

import numpy as np

import linopt

# Example 1: a state-constrained integrator with maximization of integral
A = np.array([[0]])
B = np.array([[1]])
x0 = np.array([1])
o = linopt.LinOpt(A,B,x0)

C = -np.array([1])
o.set_objective(C,0,0,0)

o.add_path_constraint(np.array([[-1]]), np.array([[1]]), np.array([0]))

o.set_parameters(timesteps=10, finaltime=1.0)
o.set_parameters(collocation=linopt.LRCollocation(2))

c,b,d,e,Me,Mi = o.discretize(o.params['collocation'])

obj1, sol1 = o.solve(solver='cvxopt')

# Example 2: Minimal-time control of the rocket car to the origin.
A = np.array([[0, 1],[0, 0]])
B = np.array([[0],[1]])
x0 = np.array([1, 0])
o = linopt.LinOpt(A,B,x0)
o.set_objective(0,0,1.0,0)

# -1 <= u <= 1
o.add_path_constraint(np.array([[0, 0]]), np.array([[1]]), np.array([-1]))
o.add_path_constraint(np.array([[0, 0]]), np.array([[-1]]), np.array([-1]))

# x(t_f) = 0
o.add_terminal_constraint(np.eye(2), np.array([0, 0]))
o.add_terminal_constraint(-np.eye(2), np.array([0, 0]))

o.set_parameters(timesteps=15, finaltime=3.0, fixedterm=False)
o.set_parameters(collocation=linopt.LRCollocation(3))
tf2, obj2, sol2 = o.solve(solver='cvxopt')

# Example 3: Minimal-time control of the rocket car to the origin, with an objective of small position.
A = np.array([[0, 1],[0, 0]])
B = np.array([[0],[1]])
x0 = np.array([1, 0])
o = linopt.LinOpt(A,B,x0)
o.set_objective(np.array([1, 0]),0,1.0,0)
o.set_parameters(collocation=linopt.LRCollocation(3))

# -1 <= u <= 1
o.add_path_constraint(np.array([[0, 0]]), np.array([[1]]), np.array([-1]))
o.add_path_constraint(np.array([[0, 0]]), np.array([[-1]]), np.array([-1]))

# x1 >= 0
o.add_path_constraint(np.array([[-1, 0]]), np.array([[0]]), np.array([0]))

# x(t_f) = 0
o.add_terminal_constraint(np.eye(2), np.array([0, 0]))
o.add_terminal_constraint(-np.eye(2), np.array([0, 0]))

# o.set_parameters(timesteps=20, finaltime=3.0, fixedterm=False)
# tf3, obj3, sol3 = o.solve(solver='cvxopt')

o.set_parameters(timesteps=20, finaltime=3.0)
obj3, sol3 = o.solve(solver='cvxopt')

# Example 4: Mass transfer
A = np.array([[0, 0],[0, 0]])
B = np.array([[-1],[1]])
x0 = np.array([1, 0])
o4 = linopt.LinOpt(A,B,x0)

o4.set_objective(0,0,0,np.array([0, -1]))

o4.add_path_constraint(np.array([[-1, 0],[0, -1]]), 0, np.array([0, 0]))
o4.add_path_constraint(0, np.array([[1],[-1]]), np.array([-1, -1]))

o4.set_parameters(timesteps=20, finaltime=5.0)
o4.set_parameters(collocation=linopt.LRCollocation(2))

c,b,d,e,Me,Mi = o4.discretize(o4.params['collocation'])

obj4, sol4 = o4.solve(solver='cvxopt')

# Example 4: Time-variable matrices in the objective
A = np.array([[0, 0],[0, 0]])
B = np.array([[-1],[1]])
x0 = np.array([1, 0])
o5 = linopt.LinOpt(A,B,x0)

C = lambda t: -np.array([0, 1]) * np.exp(-t)
o5.set_objective(C,0,0,0)

o5.add_path_constraint(np.array([[-1, 0],[0, -1]]), 0, np.array([0, 0]))
o5.add_path_constraint(0, np.array([[1],[-1]]), np.array([-1, -1]))

o5.set_parameters(timesteps=16, finaltime=5.0)
o5.set_parameters(collocation=linopt.LRCollocation(2))

c,b,d,e,Me,Mi = o5.discretize(o5.params['collocation'])

obj5, sol5 = o5.solve(solver='cvxopt')
