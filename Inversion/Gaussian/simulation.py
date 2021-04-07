from dolfin import *
import matplotlib.pyplot as plt
import fenics
import numpy as np
import os
import property_field as property_field
import scipy.misc as misc
import scipy.io
import scenarios as scenarios
from matplotlib import ticker,cm
import time


class simulation:
    def __init__(self):
        self.a =1


    def demo16(self,permeability,obs_case = 1):
        """This demo program solves the mixed formulation of Poisson's
        equation:

            sigma + grad(u) = 0    in Omega
                div(sigma) = f    in Omega
                    du/dn = g    on Gamma_N
                        u = u_D  on Gamma_D
        
        The corresponding weak (variational problem)
        
            <sigma, tau> + <grad(u), tau>   = 0
                                                    for all tau
                        - <sigma, grad(v)> = <f, v> + <g, v>
                                                    for all v
        
        is solved using DRT (Discontinuous Raviart-Thomas) elements
        of degree k for (sigma, tau) and CG (Lagrange) elements
        of degree k + 1 for (u, v) for k >= 1.
        """

        mesh = UnitSquareMesh(15, 15)
        ak_values = permeability
        flux_order=3
        s = scenarios.darcy_problem_1()
        DRT = fenics.FiniteElement("DRT", mesh.ufl_cell(), flux_order)
        # Lagrange
        CG = fenics.FiniteElement("CG", mesh.ufl_cell(), flux_order + 1)
        if s.integral_constraint:
            # From https://fenicsproject.org/qa/14184/how-to-solve-linear-system-with-constaint
            R = fenics.FiniteElement("R", mesh.ufl_cell(), 0)
            W = fenics.FunctionSpace(mesh, fenics.MixedElement([DRT, CG, R]))
            # Define trial and test functions
            (sigma, u ,r) = fenics.TrialFunctions(W)
            (tau, v , r_ ) = fenics.TestFunctions(W)
        else:
            W = fenics.FunctionSpace(mesh, DRT * CG)
            # Define trial and test functions
            (sigma, u) = fenics.TrialFunctions(W)
            (tau, v) = fenics.TestFunctions(W)
        f = s.source_function
        g = s.g

        # Define property field function
        W_CG = fenics.FunctionSpace(mesh, "Lagrange", 1)
        if s.ak is None:
            ak = property_field.get_conductivity(W_CG, ak_values)
        else:
            ak = s.ak

        # Define variational form
        a = (fenics.dot(sigma, tau) + fenics.dot(ak * fenics.grad(u), tau) +
            fenics.dot(sigma, fenics.grad(v))) * fenics.dx
        L = - f * v * fenics.dx + g * v * fenics.ds
        # L = 0
        if s.integral_constraint:
            # Lagrange multiplier?  See above link.
            a += r_ * u * fenics.dx + v * r * fenics.dx
        # Define Dirichlet BC
        bc = fenics.DirichletBC(W.sub(1), s.dirichlet_bc, s.gamma_dirichlet)
        # Compute solution
        w = fenics.Function(W)
        fenics.solve(a == L, w,bc)
        # fenics.solve(a == L, w)
        if s.integral_constraint:
            (sigma, u, r) = w.split()
        else:
            (sigma, u) = w.split()
        x=u.compute_vertex_values(mesh)
        x2 =  sigma.compute_vertex_values(mesh)
        p =x 
        pre = p.reshape((16,16))
        vx = x2[:256].reshape((16,16))
        vy = x2[256:].reshape((16,16))

        if obs_case == 1:
            dd = np.zeros([8,8])
            pos = np.full((8*8,2),0)
            col = [1,3,5,7,9,11,13,15]
            position = [1,3,5,7,9,11,13,15]
            for i in range(8):
                for j in range(8):
                    row = position
                    pos[8*i+j,:] = [col[i],row[j]]
                    dd[i,j] = pre[col[i],row[j]]    
            like = dd.reshape(8*8,)
        return like,pre,vx,vy,ak_values,pos

    def demo32(self,permeability,obs_case = 1):
        mesh = UnitSquareMesh(31, 31)
        ak_values = permeability
        flux_order=3
        s = scenarios.darcy_problem_1()
        DRT = fenics.FiniteElement("DRT", mesh.ufl_cell(), flux_order)
        # Lagrange
        CG = fenics.FiniteElement("CG", mesh.ufl_cell(), flux_order + 1)
        if s.integral_constraint:
            # From https://fenicsproject.org/qa/14184/how-to-solve-linear-system-with-constaint
            R = fenics.FiniteElement("R", mesh.ufl_cell(), 0)
            W = fenics.FunctionSpace(mesh, fenics.MixedElement([DRT, CG, R]))
            # Define trial and test functions
            (sigma, u ,r) = fenics.TrialFunctions(W)
            (tau, v , r_ ) = fenics.TestFunctions(W)
        else:
            W = fenics.FunctionSpace(mesh, DRT * CG)
            # Define trial and test functions
            (sigma, u) = fenics.TrialFunctions(W)
            (tau, v) = fenics.TestFunctions(W)
        f = s.source_function
        g = s.g

        # Define property field function
        W_CG = fenics.FunctionSpace(mesh, "Lagrange", 1)
        if s.ak is None:
            ak = property_field.get_conductivity(W_CG, ak_values)
        else:
            ak = s.ak

        # Define variational form
        a = (fenics.dot(sigma, tau) + fenics.dot(ak * fenics.grad(u), tau) +
            fenics.dot(sigma, fenics.grad(v))) * fenics.dx
        L = - f * v * fenics.dx + g * v * fenics.ds
        # L = 0
        if s.integral_constraint:
            # Lagrange multiplier?  See above link.
            a += r_ * u * fenics.dx + v * r * fenics.dx
        # Define Dirichlet BC
        bc = fenics.DirichletBC(W.sub(1), s.dirichlet_bc, s.gamma_dirichlet)
        # Compute solution

        w = fenics.Function(W)
        fenics.solve(a == L, w,bc)
        # fenics.solve(a == L, w)
        if s.integral_constraint:
            (sigma, u, r) = w.split()
        else:
            (sigma, u) = w.split()
        x=u.compute_vertex_values(mesh)
        x2 =  sigma.compute_vertex_values(mesh)
        p =x 
        pre = p.reshape((32,32))
        vx = x2[:1024].reshape((32,32))
        vy = x2[1024:].reshape((32,32))

        if obs_case == 1:
            dd = np.zeros([8,8])
            pos = np.full((8*8,2),0)
            col = [2,6,10,14,18,22,26,30]
            position = [2,6,10,14,18,22,26,30]
            for i in range(8):
                for j in range(8):
                    row = position
                    pos[8*i+j,:] = [col[i],row[j]]
                    dd[i,j] = pre[col[i],row[j]]    
            like = dd.reshape(8*8,)
        return like,pre,vx,vy,ak_values,pos

    def demo64(self,permeability,obs_case = 1):
        mesh = UnitSquareMesh(63, 63)
        ak_values = permeability
        flux_order=3
        s = scenarios.darcy_problem_1()
        DRT = fenics.FiniteElement("DRT", mesh.ufl_cell(), flux_order)
        # Lagrange
        CG = fenics.FiniteElement("CG", mesh.ufl_cell(), flux_order + 1)
        if s.integral_constraint:
            # From https://fenicsproject.org/qa/14184/how-to-solve-linear-system-with-constaint
            R = fenics.FiniteElement("R", mesh.ufl_cell(), 0)
            W = fenics.FunctionSpace(mesh, fenics.MixedElement([DRT, CG, R]))
            # Define trial and test functions
            (sigma, u ,r) = fenics.TrialFunctions(W)
            (tau, v , r_ ) = fenics.TestFunctions(W)
        else:
            W = fenics.FunctionSpace(mesh, DRT * CG)
            # Define trial and test functions
            (sigma, u) = fenics.TrialFunctions(W)
            (tau, v) = fenics.TestFunctions(W)
        f = s.source_function
        g = s.g

        # Define property field function
        W_CG = fenics.FunctionSpace(mesh, "Lagrange", 1)
        if s.ak is None:
            ak = property_field.get_conductivity(W_CG, ak_values)
        else:
            ak = s.ak

        # Define variational form
        a = (fenics.dot(sigma, tau) + fenics.dot(ak * fenics.grad(u), tau) +
            fenics.dot(sigma, fenics.grad(v))) * fenics.dx
        L = - f * v * fenics.dx + g * v * fenics.ds
        # L = 0
        if s.integral_constraint:
            # Lagrange multiplier?  See above link.
            a += r_ * u * fenics.dx + v * r * fenics.dx
        # Define Dirichlet BC
        bc = fenics.DirichletBC(W.sub(1), s.dirichlet_bc, s.gamma_dirichlet)
        # Compute solution

        w = fenics.Function(W)
        fenics.solve(a == L, w,bc)
        # fenics.solve(a == L, w)
        if s.integral_constraint:
            (sigma, u, r) = w.split()
        else:
            (sigma, u) = w.split()
        x=u.compute_vertex_values(mesh)
        x2 =  sigma.compute_vertex_values(mesh)
        p =x 
        pre = p.reshape((64,64))
        vx = x2[:4096].reshape((64,64))
        vy = x2[4096:].reshape((64,64))

        if obs_case == 1:
            dd = np.zeros([8,8])
            pos = np.full((8*8,2),0)
            col = [4,12,20,28,36,44,52,60]
            position = [4,12,20,28,36,44,52,60]
            for i in range(8):
                for j in range(8):
                    row = position
                    pos[8*i+j,:] = [col[i],row[j]]
                    dd[i,j] = pre[col[i],row[j]]    
            like = dd.reshape(8*8,)
        return like,pre,vx,vy,ak_values,pos

if __name__ == "__main__":
    dir = os.getcwd()
    plot=simulation()
    K = np.loadtxt(dir+f"/Gaussian/test_data/true_permeability_0.05.dat")
    t1 = time.time()
    u,pre,vx,vy,ak_values,pos = plot.demo64(K, obs_case = 1)
    u1=np.copy(u)
    t2 = time.time()
    print("time:",t2 - t1)
    u1=np.copy(u)
    samples = [np.log(K),pre,vx,vy]
    like_size = u.shape[0]
    sigma = np.full_like(u,0)
    # for i in range(like_size):
    #   sigma[i] = 0.05*np.abs(u[i])
    # #   print(sigma[i])
    #   u[i] = u[i] + np.random.normal(0,sigma[i])
    # np.savetxt(dir+f'/Gaussian/test_data/obs_0.05.dat',u)
    # np.savetxt(dir+f'/Gaussian/test_data/sigma_0.05.dat',sigma)
    # np.savetxt(dir+f'/Gaussian/test_data/true_pressure_0.05.dat',pre)
    # np.savetxt(dir+f'/Gaussian/test_data/vx_0.05.dat',vx)
    # np.savetxt(dir+f'/Gaussian/test_data/vy_0.05.dat',vy)
    # np.savetxt(dir+f'/Gaussian/test_data/obs_position_0.05.dat',pos)
    fig, _ = plt.subplots(1,4, figsize=(12, 3))
    vmin1 = [np.amin(samples[0]), np.amin(samples[1]),np.amin(samples[2]), np.amin(samples[3])]
    vmax1 = [np.amax(samples[0]), np.amax(samples[1]),np.amax(samples[2]), np.amax(samples[3])]
    for j, ax in enumerate(fig.axes):
        ax.set_aspect('equal')
        ax.set_axis_off()
        cax = ax.imshow(samples[j],  cmap='jet', origin='upper',vmin=vmin1[j],vmax=vmax1[j])
        cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04,
                            format=ticker.ScalarFormatter(useMathText=True))
        if j == 0:
            ax.set_title('log-permeability $X$',fontsize=12)
        if j == 1:
            ax.set_title('presure $p$',fontsize=12)
        if j == 2:
            ax.set_title('velocity $v_x$',fontsize=12)
        if j == 3:
            ax.set_title('velocity $v_y$',fontsize=12)
        if j == 1:
           ax.scatter(pos[:,0],pos[:,1],c='k',s=5)
    plt.savefig(dir+f'/Gaussian/test_data/truth.pdf',bbox_inches = 'tight', dpi=1000,pad_inches=0.0)
    plt.close()

