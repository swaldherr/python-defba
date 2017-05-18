"""
template task for scripttool
"""
# Time-stamp: <Last change 2013-04-18 21:40:33 by Steffen Waldherr>

import numpy as np

import scripttool

from src import linopt

class LinoptIntegrator(scripttool.Task):
    """
    The constrained integrator.
    """
    customize = {"finaltime":1.0}

    def run(self):
        A = np.array([[0]])
        B = np.array([[1]])
        x0 = np.array([1])
        self.o = linopt.LinOpt(A,B,x0)
        C = -np.array([1])
        self.o.set_objective(C,0,0,0)
        self.o.add_path_constraint(np.array([[-1]]), np.array([[1]]), np.array([0]))
        self.o.set_parameters(timesteps=10, finaltime=self.finaltime)
        self.o.set_parameters(collocation=linopt.LRCollocation(2))
        c,b,d,e,Me,Mi = self.o.discretize(self.o.params['collocation'])
        obj1, sol1 = self.o.solve(solver='cvxopt')
        t = np.arange(0, self.finaltime, step=self.finaltime/200)

        # get collocation points
        h = self.finaltime / self.o.params['timesteps']
        r = np.concatenate([h * i + 0.5 * h * (sol1.collocation.points() + 1) for i in range(self.o.params['timesteps'])])
        
        fig, ax = self.make_ax(name="opt-state",
                               xlabel="t",
                               ylabel="x",
                               title="Optimal state trajectory")
        ax.plot(t, sol1.get_state(t))
        ax.plot(r, sol1.get_state(r), 'r.')
        fig, ax = self.make_ax(name="opt-control",
                               xlabel="t",
                               ylabel="u",
                               title="Optimal control trajectory")
        ax.plot(t, sol1.get_control(t))
        ax.plot(r, sol1.get_control(r), 'r.')

class LinoptIntegratorVar(scripttool.Task):
    """
    The constrained integrator with variability
    """
    customize = {"finaltime":1.0}

    def run(self):
        A = np.array([[0]])
        B = np.array([[1]])
        x0 = np.array([1])
        self.o = linopt.LinOpt(A,B,x0)
        self.o.set_objective(0,0,0,0)
        self.o.add_path_constraint(np.array([[-1]]), np.array([[1]]), np.array([0]))
        self.o.add_path_constraint(np.array([[0]]), np.array([[-1]]), np.array([0]))
        self.o.add_terminal_constraint(np.array([[-1]]), np.array([2.5]))
        self.o.set_parameters(timesteps=10, finaltime=self.finaltime)
        self.o.set_parameters(collocation=linopt.LLCollocation(3))
        self.o.set_solver('cvxopt')
        c,b,d,e,Me,Mi = self.o.discretize(self.o.params['collocation'])
        obj1, sol1 = self.o.solve(params={'solver':'glpk'})
        t = np.arange(0, self.finaltime, step=self.finaltime/200)

        tvar = np.linspace(0, self.finaltime, num=40)
        umin, umax = self.o.control_variability([1], tvar, optval=obj1, solver='glpk')
        xmin, xmax = self.o.state_variability([1], tvar, optval=obj1, solver='glpk')
        
        # get collocation points
        h = self.finaltime / self.o.params['timesteps']
        r = np.concatenate([h * i + 0.5 * h * (sol1.collocation.points() + 1) for i in range(self.o.params['timesteps'])])
        
        fig, ax = self.make_ax(name="opt-state",
                               xlabel="t",
                               ylabel="x",
                               title="Optimal state trajectory")
        ax.plot(t, sol1.get_state(t))
        ax.plot(r, sol1.get_state(r), 'r.')
        ax.fill_between(tvar, xmin, xmax, alpha=0.2, color='b')
        fig, ax = self.make_ax(name="opt-control",
                               xlabel="t",
                               ylabel="u",
                               title="Optimal control trajectory")
        ax.plot(t, sol1.get_control(t))
        ax.plot(r, sol1.get_control(r), 'r.')
        ax.fill_between(tvar, umin, umax, alpha=0.2, color='b')

class LinoptRocketcar(scripttool.Task):
    """
    Minimum time control of the rocket car.
    """
    customize = {"finaltime":3.0}

    def run(self):
        A = np.array([[0, 1],[0, 0]])
        B = np.array([[0],[1]])
        x0 = np.array([1, 0])
        self.o = linopt.LinOpt(A,B,x0)
        self.o.set_objective(0,0,1.0,0)

        # -1 <= u <= 1
        self.o.add_path_constraint(np.array([[0, 0]]), np.array([[1]]), np.array([-1]))
        self.o.add_path_constraint(np.array([[0, 0]]), np.array([[-1]]), np.array([-1]))

        # x(t_f) = 0
        self.o.add_terminal_constraint(np.eye(2), np.array([0, 0]))
        self.o.add_terminal_constraint(-np.eye(2), np.array([0, 0]))

        self.o.set_parameters(timesteps=15, finaltime=self.finaltime, fixedterm=False)
        self.o.set_parameters(collocation=linopt.LRrevCollocation(3))
        tf1, obj1, sol1 = self.o.solve(solver='cvxopt')
        self.printf("Found optimal terminal time: %g" % tf1)
        
        t = np.arange(0, tf1, step=tf1/200)

        # get collocation points
        h = tf1 / self.o.params['timesteps']
        r = np.concatenate([h * i + 0.5 * h * (sol1.collocation.points() + 1) for i in range(self.o.params['timesteps'])])
        
        fig, ax = self.make_ax(name="opt-state",
                               xlabel="t",
                               ylabel="x",
                               title="Optimal state trajectory")
        ax.plot(t, sol1.get_state(t))
        ax.plot(r, sol1.get_state(r), 'r.')
        fig, ax = self.make_ax(name="opt-control",
                               xlabel="t",
                               ylabel="x",
                               title="Optimal control trajectory")
        ax.plot(t, sol1.get_control(t))
        ax.plot(r, sol1.get_control(r), 'r.')

class LinoptRocketcarVar(scripttool.Task):
    """
    Optimal control of the rocket car with state and control variability.
    """
    customize = {"finaltime":3.0}

    def run(self):
        A = np.array([[0, 1],[0, 0]])
        B = np.array([[0],[1]])
        x0 = np.array([1, 0])
        self.o = linopt.LinOpt(A,B,x0)
        self.o.solver = 'cvxopt'
        self.o.set_objective(0, 0, 0, 0)

        # -1 <= u <= 1
        self.o.add_path_constraint(np.array([[0, 0]]), np.array([[1]]), np.array([-1]))
        self.o.add_path_constraint(np.array([[0, 0]]), np.array([[-1]]), np.array([-1]))

        # x(t_f) = 0
        self.o.add_terminal_constraint(np.eye(2), np.array([0, 0]))
        self.o.add_terminal_constraint(-np.eye(2), np.array([0, 0]))

        self.o.set_parameters(timesteps=15, finaltime=self.finaltime)
        self.o.set_parameters(collocation=linopt.LRCollocation(3))
        obj1, sol1 = self.o.solve()
        tf1 = self.o.params['finaltime']

        tvar = np.linspace(0, tf1, num=10)
        umin, umax = self.o.control_variability([1], tvar, optval=obj1)
        x1min, x1max = self.o.state_variability([1, 0], tvar, optval=obj1)
        
        t = np.arange(0, tf1, step=tf1/200)

        # get collocation points
        h = tf1 / self.o.params['timesteps']
        r = np.concatenate([h * i + 0.5 * h * (sol1.collocation.points() + 1) for i in range(self.o.params['timesteps'])])
        
        fig, ax = self.make_ax(name="opt-state",
                               xlabel="t",
                               ylabel="x",
                               title="Optimal state trajectory")
        ax.plot(t, sol1.get_state(t))
        ax.plot(r, sol1.get_state(r), 'r.')
        ax.plot(tvar, x1min, '--b')
        ax.plot(tvar, x1max, '--b')
        fig, ax = self.make_ax(name="opt-control",
                               xlabel="t",
                               ylabel="x",
                               title="Optimal control trajectory")
        ax.plot(t, sol1.get_control(t))
        ax.plot(r, sol1.get_control(r), 'r.')
        ax.plot(tvar, umin, '--r')
        ax.plot(tvar, umax, '--r')

# creation of my experiments
scripttool.register_task(LinoptIntegrator(), ident="linopt-integrator")
scripttool.register_task(LinoptIntegratorVar(), ident="linopt-integrator-var")
scripttool.register_task(LinoptRocketcar(), ident="linopt-rocketcar")
scripttool.register_task(LinoptRocketcarVar(), ident="linopt-rocketcar-var")
