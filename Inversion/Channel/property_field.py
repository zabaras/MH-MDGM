import fenics


def get_conductivity(v, values, mesh=None, dimension=2, num_subspaces=0):
    """
    Take a conductivity field that was read in and translate it to a fenics mesh
    :param mesh: the fenics mesh onto which you want to put the conductivity field
    :param v: a function space
    :param values: function values at the nodes
    :param dimension: the problem spatial dimension
    :return: a fenics.Function describing the property field
    """
    c = fenics.Function(v)
    # ordering = dolfin.dof_to_vertex_map(v)
    ordering = fenics.dof_to_vertex_map(v)
    #c.vector()[:] = values.flatten(order='C')[ordering]
    values.astype(float) 
    c.vector()[:] = values.flatten()[ordering]
    return c

