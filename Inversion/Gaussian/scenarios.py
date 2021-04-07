import fenics
import numpy as np
import warnings

class Scenario(object):
    """
    Class that holds the important information about the system being solved
    
    Properties:
        source_function: a fenics.Expression that describes the sources and sinks in the problem domain
        dirichlet_bc: a fenics.Expression that describes the Dirichlet boundary conditions
        gamma_dirichlet: tells which parts of the boundary are subject to the Dirichlet BCs.
        integral_constraint: if True, we need to constrain the average pressure to be zero.
        
        Neumann BCs are currently flux = 0 and are defined wherever the Dirichlet BCs aren't
    """

    def __init__(self):
        self.source_function = fenics.Constant(0.0)
        self.dirichlet_bc = fenics.Constant(0.0)
        self.g = fenics.Constant(0.0)
        self.problem_number = 0
        self.integral_constraint = False
        self.ak = None  # Use this for analytical property field


    @staticmethod
    def gamma_dirichlet(x, on_boundary):
        return on_boundary



def darcy_problem_1():
    s = Scenario()
    s.problem_number = 1
    s.source_function = fenics.Constant(3.0)
    s.dirichlet_bc = fenics.Expression("0.0", degree=1)

    def boundary(x, on_boundary):
        return x[0] < fenics.DOLFIN_EPS or x[0] > 1.0 - fenics.DOLFIN_EPS
    s.gamma_dirichlet = boundary
    s.g = fenics.Constant(0.0)
    return s
