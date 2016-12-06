"""
Copyright (C) 2013-2016 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

optional.py
~~~~~~~~~~~

Optionally loaded constraints.

"""

import pyomo.core as po
from .. import utils
from .. import exceptions
import time

def ramping_rate(model):
    """
    Ramping rate constraints.

    Depends on: node_energy_balance, node_constraints_build

    """
    m = model.m
    time_res = model.data['_time_res'].to_series()

    # Constraint rules
    def _ramping_rule(m, y, x, t, direction):
        # e_ramping: Ramping rate [fraction of installed capacity per hour]
        ramping_rate_value = model.get_option(y + '.constraints.e_ramping')
        if ramping_rate_value is False:
            # If the technology defines no `e_ramping`, we don't build a
            # ramping constraint for it!
            return po.Constraint.NoConstraint
        else:
            # No constraint for first timestep
            # NB: From Pyomo 3.5 to 3.6, order_dict became zero-indexed
            if m.t.order_dict[t] == 0:
                return po.Constraint.NoConstraint
            else:
                carrier = model.get_option(y + '.carrier')
                diff = ((m.es_prod[carrier, y, x, t]
                         + m.es_con[carrier, y, x, t]) / time_res.at[t]
                        - (m.es_prod[carrier, y, x, model.prev_t(t)]
                           + m.es_con[carrier, y, x, model.prev_t(t)])
                        / time_res.at[model.prev_t(t)])
                max_ramping_rate = ramping_rate_value * m.e_cap[y, x]
                if direction == 'up':
                    return diff <= max_ramping_rate
                else:
                    return -1 * max_ramping_rate <= diff

    def c_ramping_up_rule(m, y, x, t):
        return _ramping_rule(m, y, x, t, direction='up')

    def c_ramping_down_rule(m, y, x, t):
        return _ramping_rule(m, y, x, t, direction='down')

    # Constraints
    m.c_ramping_up = po.Constraint(m.y, m.x, m.t, rule=c_ramping_up_rule)
    m.c_ramping_down = po.Constraint(m.y, m.x, m.t, rule=c_ramping_down_rule)


def group_fraction(model):
    """
    Constrain groups of technologies to reach given fractions of e_prod.

    """
    m = model.m

    def sign_fraction(group, group_type):
        o = model.config_model
        sign, fraction = o.group_fraction[group_type].get_key(group)
        return sign, fraction

    def group_set(group_type):
        try:
            group = model.config_model.group_fraction[group_type].keys()
        except (TypeError, KeyError):
            group = []
        return po.Set(initialize=group)

    def techs_to_consider(supply_techs, group_type):
        # Remove ignored techs if any defined
        gfc = model.config_model.group_fraction
        if 'ignored_techs' in gfc and group_type in gfc.ignored_techs:
            return [i for i in supply_techs
                    if i not in gfc.ignored_techs[group_type]]
        else:
            return supply_techs

    def equalizer(lhs, rhs, sign):
        if sign == '<=':
            return lhs <= rhs
        elif sign == '>=':
            return lhs >= rhs
        elif sign == '==':
            return lhs == rhs
        else:
            raise ValueError('Invalid sign: {}'.format(sign))

    supply_techs = (model.get_group_members('supply') +
                    model.get_group_members('conversion'))

    # Sets
    m.output_group = group_set('output')
    m.capacity_group = group_set('capacity')
    m.demand_power_peak_group = group_set('demand_power_peak')

    # Constraint rules
    def c_group_fraction_output_rule(m, c, output_group):
        sign, fraction = sign_fraction(output_group, 'output')
        techs = techs_to_consider(supply_techs, 'output')
        rhs = (fraction
               * sum(m.es_prod[c, y, x, t] for y in techs
                     for x in m.x for t in m.t))
        lhs = sum(m.es_prod[c, y, x, t]
                  for y in model.get_group_members(output_group) for x in m.x
                  for t in m.t)
        return equalizer(lhs, rhs, sign)

    def c_group_fraction_capacity_rule(m, c, capacity_group):  # pylint: disable=unused-argument
        sign, fraction = sign_fraction(capacity_group, 'capacity')
        techs = techs_to_consider(supply_techs, 'capacity')
        rhs = (fraction
               * sum(m.e_cap[y, x] for y in techs for x in m.x))
        lhs = sum(m.e_cap[y, x] for y in model.get_group_members(capacity_group)
                  for x in m.x)
        return equalizer(lhs, rhs, sign)

    def c_group_fraction_demand_power_peak_rule(m, c, demand_power_peak_group):
        sign, fraction = sign_fraction(demand_power_peak_group,
                                       'demand_power_peak')
        margin = model.config_model.system_margin.get_key(c, default=0)
        peak_timestep = model.t_max_demand.power
        y = 'demand_power'
        # Calculate demand peak taking into account both r_scale and time_res
        peak = (float(sum(model.m.r[y, x, peak_timestep]
                      * model.get_option(y + '.constraints.r_scale', x=x)
                      for x in model.m.x))
                / model.data.time_res_series.at[peak_timestep])
        rhs = fraction * (-1 - margin) * peak
        lhs = sum(m.e_cap[y, x]
                  for y in model.get_group_members(demand_power_peak_group)
                  for x in m.x)
        return equalizer(lhs, rhs, sign)

    # Constraints
    m.c_group_fraction_output = \
        po.Constraint(m.c, m.output_group, rule=c_group_fraction_output_rule)
    m.c_group_fraction_capacity = \
        po.Constraint(m.c, m.capacity_group, rule=c_group_fraction_capacity_rule)
    grp = m.demand_power_peak_group
    m.c_group_fraction_demand_power_peak = \
        po.Constraint(m.c, grp, rule=c_group_fraction_demand_power_peak_rule)

def piecewise(model):
    """
    piecewise constraint creates 2-dimensional piecewise constraint to connect consumption and
    production of conversion technologies

    uses BuildPiecewiseND from
    https://projects.coin-or.org/Pyomo/browser/pyomo.data/trunk/pyomo/data/pyomobook/scripts/advanced?rev=10872&order=name
    
    """
    import numpy as np
    from . import base
    m = model.m
    # remove constraint referring to energy conversion
    del(m.c_s_balance_conversion)
    del(m.c_s_balance_conversion_index)
    del(m.c_s_balance_conversion_index_index_0)

    # pieces are in yaml file 'piecewise' and loaded into config_model
    piece_dict = model.config_model.pieces

    def set_piecewise_constraints(y, x, t, piece_dict):
        c_prod = model.get_option(y + '.carrier', x=x)
        c_source = model.get_option(y + '.source_carrier', x=x)

        num = 10 #number of pieces to split capacity variable

        def generate_delaunay(prod, y, x=None, num=10):
            """
            Generate a Delaunay triangulation of the 2-dimensional
            bounded variable domain given the array of Pyomo
            variables [x, y]. x = load rate piecewise, y = maximum capacity.
            The number of grid points to generate for
            each variable is set by the optional keyword argument
            'num' (default=10).
            Requires both numpy and scipy.spatial be available.
            """
            import scipy.spatial
        
            linegrids = []
            e_cap = model.get_option(y + '.constraints.e_cap.max', x=x)
            if e_cap == 0:
                return None
            cap = np.linspace(0, e_cap, num)

            for c in cap:
                for i in prod:
                    linegrids.append([c*i,c])
            # generates a meshgrid and then flattens and transposes
            # the meshgrid into an (npoints, D) shaped array of
            # coordinates
            points = np.vstack(linegrids)
            return scipy.spatial.Delaunay(points)

        def BuildPiecewiseND(xvars, zvar, tri, zvals):
            """
            Builds constraints defining a D-dimensional
            piecewise representation of the given triangulation.
            Args:
                xvars: A (D, 1) array of Pyomo variable objects
                       representing the inputs of the piecewise
                       function.
                zvar: A Pyomo variable object set equal to the
                      output of the piecewise function.
                tri: A triangulation over the discretized
                     variable domain. Required attributes:
                   - points: An (npoints, D) shaped array listing the
                             D-dimensional coordinates of the
                             discretization points.
                   - simplices: An (nsimplices, D+1) shaped array of
                                integers specifying the D+1 indices
                                of the points vector that define
                                each simplex of the triangulation.
                zvals: An (npoints, 1) shaped array listing the
                       value of the piecewise function at each of
                       coordinates in the triangulation points
                       array.
            Returns:
                A Pyomo Block object containing variables and
                constraints that define the piecewise function.
            """
        
            b = po.Block(concrete=True)
            ndim = len(xvars)
            nsimplices = len(tri.simplices)
            npoints = len(tri.points)
            pointsT = list(zip(*tri.points))
        
            # create index objects
            b.dimensions =  po.RangeSet(0, ndim-1)
            b.simplices = po.RangeSet(0, nsimplices-1)
            b.vertices = po.RangeSet(0, npoints-1)
        
            # create variables
            b.lmda = po.Var(b.vertices, within=po.NonNegativeReals)
            b.y = po.Var(b.simplices, within=po.Binary)
        
            # create constraints
            def input_c_rule(b, d):
                pointsTd = pointsT[d]
                return xvars[d] == sum(pointsTd[v]*b.lmda[v]
                                       for v in b.vertices)
            b.input_c = po.Constraint(b.dimensions, rule=input_c_rule)
        
            b.output_c = po.Constraint(expr=\
                zvar == sum(zvals[v]*b.lmda[v] for v in b.vertices))
            
            b.convex_c = po.Constraint(expr=\
                sum(b.lmda[v] for v in b.vertices) == 1)
            
            # generate a map from vertex index to simplex index,
            # which avoids an n^2 lookup when generating the
            # constraint
            vertex_to_simplex = [[] for v in b.vertices]
            for s, simplex in enumerate(tri.simplices):
                for v in simplex:
                    vertex_to_simplex[v].append(s)
            def vertex_regions_rule(b, v):
                return b.lmda[v] <= \
                    sum(b.y[s] for s in vertex_to_simplex[v])
            b.vertex_regions_c = \
                po.Constraint(b.vertices, rule=vertex_regions_rule)
            
            b.single_region_c = po.Constraint(expr=\
                sum(b.y[s] for s in b.simplices) == 1)
            
            return b

        tri = generate_delaunay(piece_dict[y][c_source]['prod'], y, x=x, num=num)
        
        if not tri:
            return po.Constraint(po.Set(y), po.Set(x), po.Set(t), rule=conversion_rule)
        def _get_con_vals(tri, piece_dict, c_source):
            prod_array, cap_array = np.transpose(tri.points)
            cap_array_T = np.reshape(cap_array,(num,len(piece_dict[y][c_source]['prod'])))
            con_vals = []
            for i in range(len(cap_array_T)):
                con_vals.append(np.multiply(cap_array_T[i], piece_dict[y][c_source]['con']))
            con_vals = [item for sublist in con_vals for item in sublist]
            return con_vals
        
        con_vals = _get_con_vals(tri, piece_dict, c_source)

        return BuildPiecewiseND([m.es_prod[c_prod,y,x,t], m.e_cap[y,x]],
         -1 * m.es_con[c_source,y,x,t],
         tri,
         con_vals)
        
    def conversion_rule(m, y, x, t):
                c_prod = model.get_option(y + '.carrier')
                c_source = model.get_option(y + '.source_carrier')
                e_eff = base.get_constraint_param(model, 'e_eff', y, x, t)
                return (m.es_prod[c_prod, y, x, t]
                      == -1 * m.es_con[c_source, y, x, t] * e_eff)

    y_conv_non_piecewise = []

    for y in m.y_conv:
        if 'piecewise' in model.get_option(y):
            if 'source_carrier' in model.get_option(y + '.piecewise'):
                for x in m.x:
                    for t in m.t:
                        timestamp = int(time.mktime(t.timetuple()))
                        setattr(m,"pc_{}_{}_{}".format(y,x,timestamp),set_piecewise_constraints(y, x, t, piece_dict))
            else:
                y_conv_non_piecewise.append(y)
        else:
            y_conv_non_piecewise.append(y)

    m.c_s_balance_conversion = po.Constraint(po.Set(initialize=y_conv_non_piecewise, ordered=True), m.x, m.t,
                                             rule=conversion_rule)

def secondary_carrier(model):
    """
    Temporary optional constraint to provide a secondary carrier for a given
    technology. 
    E.g. combined heat and power (CHP) consumes gas to produce heat and electricity.
    
    Requires definition of heat to power ratio (htp) in constraints
    """
    m = model.m
    
    def htp_rule(m, y, x, t):
        try:
            c_1 = model.get_option(y + '.carrier')
            c_2 = model.get_option(y + '.carrier_2')
            htp = model.get_option(y + '.constraints.htp')
            return (m.es_prod[c_2, y, x, t]
                == m.es_prod[c_1, y, x, t] * htp)
        except exceptions.OptionNotSetError:
            return po.Constraint.Skip

    def set_piecewise_constraints(y, x, t, piece_dict):
        c_1 = model.get_option(y + '.carrier')
        c_2 = model.get_option(y + '.carrier_2')
        num = 10 #number of pieces to split capacity variable

        def generate_delaunay(prod, y, x=None, num=10):
            """
            Generate a Delaunay triangulation of the 2-dimensional
            bounded variable domain given the array of Pyomo
            variables [x, y]. x = load rate piecewise, y = maximum capacity.
            The number of grid points to generate for
            each variable is set by the optional keyword argument
            'num' (default=10).
            Requires both numpy and scipy.spatial be available.
            """
            import scipy.spatial
        
            linegrids = []
            e_cap = model.get_option(y + '.constraints.e_cap.max', x=x)
            cap = np.linspace(0, e_cap, num)
            if e_cap == 0:
                return None
            for c in cap:
                for i in prod:
                    linegrids.append([c*i,c])
            # generates a meshgrid and then flattens and transposes
            # the meshgrid into an (npoints, D) shaped array of
            # coordinates
            points = np.vstack(linegrids)
            return scipy.spatial.Delaunay(points)

        def BuildPiecewiseND(xvars, zvar, tri, zvals):
            """
            Builds constraints defining a D-dimensional
            piecewise representation of the given triangulation.
            Args:
                xvars: A (D, 1) array of Pyomo variable objects
                       representing the inputs of the piecewise
                       function.
                zvar: A Pyomo variable object set equal to the
                      output of the piecewise function.
                tri: A triangulation over the discretized
                     variable domain. Required attributes:
                   - points: An (npoints, D) shaped array listing the
                             D-dimensional coordinates of the
                             discretization points.
                   - simplices: An (nsimplices, D+1) shaped array of
                                integers specifying the D+1 indices
                                of the points vector that define
                                each simplex of the triangulation.
                zvals: An (npoints, 1) shaped array listing the
                       value of the piecewise function at each of
                       coordinates in the triangulation points
                       array.
            Returns:
                A Pyomo Block object containing variables and
                constraints that define the piecewise function.
            """
        
            b = po.Block(concrete=True)
            ndim = len(xvars)
            nsimplices = len(tri.simplices)
            npoints = len(tri.points)
            pointsT = list(zip(*tri.points))
        
            # create index objects
            b.dimensions =  po.RangeSet(0, ndim-1)
            b.simplices = po.RangeSet(0, nsimplices-1)
            b.vertices = po.RangeSet(0, npoints-1)
        
            # create variables
            b.lmda = po.Var(b.vertices, within=po.NonNegativeReals)
            b.y = po.Var(b.simplices, within=po.Binary)
        
            # create constraints
            def input_c_rule(b, d):
                pointsTd = pointsT[d]
                return xvars[d] == sum(pointsTd[v]*b.lmda[v]
                                       for v in b.vertices)
            b.input_c = po.Constraint(b.dimensions, rule=input_c_rule)
        
            b.output_c = po.Constraint(expr=\
                zvar == sum(zvals[v]*b.lmda[v] for v in b.vertices))
        
            b.convex_c = po.Constraint(expr=\
                sum(b.lmda[v] for v in b.vertices) == 1)
        
            # generate a map from vertex index to simplex index,
            # which avoids an n^2 lookup when generating the
            # constraint
            vertex_to_simplex = [[] for v in b.vertices]
            for s, simplex in enumerate(tri.simplices):
                for v in simplex:
                    vertex_to_simplex[v].append(s)
            def vertex_regions_rule(b, v):
                return b.lmda[v] <= \
                    sum(b.y[s] for s in vertex_to_simplex[v])
            b.vertex_regions_c = \
                po.Constraint(b.vertices, rule=vertex_regions_rule)
        
            b.single_region_c = po.Constraint(expr=\
                sum(b.y[s] for s in b.simplices) == 1)
        
            return b
        
        tri = generate_delaunay(piece_dict[y]['htp']['c_1'], y, x=x, num=num)
        if not tri: # case where e_cap.max = 0
            return None
        def _get_c2_vals(tri, piece_dict):
            prod_array, cap_array = np.transpose(tri.points)
            cap_array_T = np.reshape(cap_array,(num,len(piece_dict[y]['htp']['c_1'])))
            c2_vals = []
            for i in range(len(cap_array_T)):
                c2_vals.append(np.multiply(cap_array_T[i], piece_dict[y]['htp']['c_2']))
            c2_vals = [item for sublist in c2_vals for item in sublist]
            return c2_vals

        c2_vals = _get_con_vals(tri, piece_dict)

        return BuildPiecewiseND([m.es_prod[c_1,y,x,t], m.e_cap[y,x]],
         -1 * m.es_prod[c_2,y,x,t],
         tri,
         c2_vals)
    
    #if 'constraints.optional.piecewise' in model.config_model.constraints:
    #    piece_dict = model.config_model.pieces
    #    y_htp_non_piecewise=[]
    #    for y in m.y_conv:
    #        if 'piecewise' in model.get_option(y):
    #            if 'carrier_2' in model.get_option(y + '.piecewise'):
    #                for x in m.x:
    #                        for t in m.t:
    #                            timestamp = int(time.mktime(t.timetuple()))
    #                            setattr(m,"pc_{}_{}_{}_htp".format(y,x,timestamp),set_piecewise_constraints(y, x, t, piece_dict))
    #            else:
    #                y_htp_non_piecewise.append(y)
    #        else:
    #            y_htp_non_piecewise.append(y)
    #    m.c_s_htp = po.Constraint(po.Set(initialize=y_htp_non_piecewise, ordered=True),
    #                          m.x, m.t, rule=htp_rule)
    #else:
        # TODO: set of techs given by whether they define a secondary carrier
    m.c_s_htp = po.Constraint(m.y_conv, m.x, m.t, rule=htp_rule)
    

def secondary_source_carrier(model):
    """
    Temporary optional constraint to provide a secondary source carrier for a given
    technology. 
    E.g. Boiler consumes gas and electricity to produce heat.
    The electricity consumption is for pumps in this case.

    Requires the technologies to be subscripted with their share of the energy (e.g.
    chp_heat & chp_power).

    Constraint simply forces "_heat" technology to operate at the same time as the "_power"
    technology.
    """
    m = model.m
    def conversion_rule(m, y, x, t):
        c_prod = model.get_option(y + '.carrier')
        c_source = model.get_option(y + '.source_carrier_2')
        try:
            e_eff = get_constraint_param(model, 'e_eff_2', y, x, t)
        except:
            e_eff = 1
        return (m.es_prod[c_prod, y, x, t]
                == -1 * m.es_con[c_source, y, x, t] * e_eff)

    m.c_s_balance_conversion_source_2 = po.Constraint(m.y_conv, m.x, m.t,
                                             rule=conversion_rule)

def fixed_cost(model):
    
    m = model.m
    m.del_component(m.c_cost) #remove this (created initially in base.py)so we can recreate it
    m.del_component(m.c_cost_index)
    m.del_component(m.c_cost_index_index_0)

    cost_getter = utils.cost_getter(model.get_option)
    @utils.memoize
    def _cost(cost, y, k, x=None):
        return cost_getter(cost, y, k, x=x)
    
    m.purchased = po.Var(m.y, m.x, within=po.Binary)
    m.cost_con_fixed = po.Var(m.y, m.x, m.kc, within=po.NonNegativeReals)

    def purchased_rule(m, y, x): #Binary result of whether a tech has non-zero production at any point in time horizon
        prod = sum(m.es_prod[c,y,x,t] for c in m.c for t in m.t)
        return (m.purchased[y,x] >= prod / 1e10)

    def c_cost_rule(m, y, x, k): #re-create this rule with cost_con_fixed
        return (
            m.cost[y, x, k] ==
            m.cost_con[y, x, k] +
            m.cost_op_fixed[y, x, k] +
            m.cost_op_variable[y, x, k] +
            m.cost_con_fixed[y, x, k] 
        )

    def cost_con_fixed_rule(m, y, x, k): # Cost incurred as fixed value irrespective of technology size
        cap_fixed = _cost('cap_fixed', y, k, x)
        if y in m.y_trans:
            # Divided by 2 for transmission techs because construction costs
            # are counted at both ends
            cap_fixed = cap_fixed/2 
        return (m.cost_con_fixed[y,x,k] == m.purchased[y,x] * cap_fixed)

    m.c_purchased = po.Constraint(m.y, m.x, rule=purchased_rule)
    m.c_cost = po.Constraint(m.y, m.x, m.kc, rule=c_cost_rule)
    m.c_cost_con_fixed = po.Constraint(m.y, m.x, m.kc, rule=cost_con_fixed_rule)