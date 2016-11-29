"""
Copyright (C) 2013-2016 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

base.py
~~~~~~~

Basic model constraints.

"""

import pyomo.core as po
import numpy as np
import time

from .. import exceptions
from .. import transmission
from .. import utils

def get_constraint_param(model, param_string, y, x, t):
    """
    Function to get values for constraints which can optionally be
    loaded from file (so may have time dependency).

    model = calliope model
    param_string = constraint as string
    y = technology
    x = location
    t = timestep
    """
    get_any_option = utils.any_option_getter(model)
    if param_string in model.data:
        return getattr(model.m, param_string)[y, x, t]
    else:
        return get_any_option(y + '.constraints.' + param_string, x=x)

def get_cost_param(model, cost, k, y, x, t):
    """
    Function to get values for constraints which can optionally be
    loaded from file (so may have time dependency).

    model = calliope model
    cost = cost name, e.g. 'om_fuel'
    k = cost type, e.g. 'monetary'
    y = technology
    x = location
    t = timestep
    """
    get_any_option = utils.any_option_getter(model)

    param_string = 'costs_' + k + '_' + cost #format stored in model.data
    
    if param_string in model.data:
        return getattr(model.m, param_string)[y, x, t]
    else: #turn e.g. costs_monetary_om_var to costs.monetary.om_var, then search in DataArray
        return get_any_option(y + '.' + param_string.replace('_','.',2), x=x) 

def get_revenue_param(model, rev, k, y, x, t):
    """
    Function to get values for constraints which can optionally be
    loaded from file (so may have time dependency).

    model = calliope model
    rev = revenue name, e.g. 'sub_var'
    k = revenue type, e.g. 'monetary'
    y = technology
    x = location
    t = timestep
    """
    get_any_option = utils.any_option_getter(model)

    param_string = 'revenue_' + k + '_' + rev #format stored in model.data
    
    if param_string in model.data:
        return getattr(model.m, param_string)[y, x, t]
    else: #turn e.g. revenue_monetary_om_var to revenue.monetary.om_var, then search in DataArray
        return get_any_option(y + '.' + param_string.replace('_','.',2), x=x) 

def node_resource(model):
    """
    Defines variables:

    * rs: resource <-> storage (+ production, - consumption)
    * r_area: resource collector area
    * rbs: secondary resource -> storage (+ production)

    """
    m = model.m
    get_any_option = utils.any_option_getter(model)
    # Variables
    m.rs = po.Var(m.y, m.x, m.t, within=po.Reals)
    m.r_area = po.Var(m.y_def_r, m.x, within=po.NonNegativeReals)
    m.rbs = po.Var(m.y_rb, m.x, m.t, within=po.NonNegativeReals)

    # Constraint rules
    def c_rs_rule(m, y, x, t):
        r_scale = get_any_option(y + '.constraints.r_scale', x=x)
        r_eff = get_constraint_param(model, 'r_eff', y, x, t)
        force_r = get_constraint_param(model, 'force_r', y, x, t)
        r_avail = (m.r[y, x, t]
                   * r_scale
                   * m.r_area[y, x]
                   * r_eff)
        if force_r:
            return m.rs[y, x, t] == r_avail
        # TODO reformulate conditionally once Pyomo supports that:
        # had to remove the following formulation because it is not
        # re-evaluated on model re-construction -- we now check for
        # demand/supply tech instead, which means that `r` can only
        # be ALL negative or ALL positive for a given tech!
        # elif po.value(m.r[y, x, t]) > 0:
        elif (y in model.get_group_members('supply') or
              y in model.get_group_members('unmet_demand')):
            return m.rs[y, x, t] <= r_avail
        elif y in model.get_group_members('demand'):
            return m.rs[y, x, t] >= r_avail

    # Constraints
    m.c_rs = po.Constraint(m.y_def_r, m.x, m.t, rule=c_rs_rule)


def node_energy_balance(model):
    """
    Defines variables:

    * s: storage level
    * es_prod: storage -> carrier (+ production)
    * es_con: storage <- carrier (- consumption)

    """
    m = model.m
    get_any_option = utils.any_option_getter(model)
    d = model.data
    time_res = model.data['_time_res'].to_series()

    ## FIXME this is very inefficient for y not in y_def_e_eff
    #def get_e_eff(m, y, x, t):
    #    if y in m.y_def_e_eff:
    #        e_eff = m.e_eff[y, x, t]
    #    else:  # This includes transmission technologies
    #        e_eff = d.e_eff.loc[dict(y=y, x=x)][0]  # Just get first entry
    #    return e_eff

    def get_e_eff_per_distance(model, y, x):
        try:
            e_loss = get_any_option(y + '.constraints_per_distance.e_loss', x=x)
            per_distance = model.get_option(y + '.per_distance')
            distance = model.get_option(y + '.distance')
            return 1 - (e_loss * (distance / per_distance))
        except exceptions.OptionNotSetError:
            return 1.0

    # Variables
    m.s = po.Var(m.y_pc, m.x, m.t, within=po.NonNegativeReals)
    m.es_prod = po.Var(m.c, m.y, m.x, m.t, within=po.NonNegativeReals)
    m.es_con = po.Var(m.c, m.y, m.x, m.t, within=po.NegativeReals)
    # used to be defined in node_constraints_build, moved to work with piecewise:
    
    m.e_cap = po.Var(m.y, m.x, within=po.NonNegativeReals)
    # Constraint rules
    def transmission_rule(m, y, x, t):
        y_remote, x_remote = transmission.get_remotes(y, x)
        e_eff = get_constraint_param(model, 'e_eff', y, x, t)
        if y_remote in m.y_trans:
            c = model.get_option(y + '.carrier')
            return (m.es_prod[c, y, x, t]
                    == -1 * m.es_con[c, y_remote, x_remote, t]
                    * e_eff
                    * get_e_eff_per_distance(model, y, x))
        else:
            return po.Constraint.NoConstraint

    def conversion_rule(m, y, x, t):
        c_prod = model.get_option(y + '.carrier')
        c_source = model.get_option(y + '.source_carrier')
        e_eff = get_constraint_param(model, 'e_eff', y, x, t)
        if 'piecewise.source_carrier' in model.get_option(y).as_dict_flat():
                # in case source_carrier is a piecewise function
                timestamp = int(time.mktime(t.timetuple()))
                x_pieces = model.config_model.pieces[y][c_source].prod
                z_pieces = model.config_model.pieces[y][c_source].con
                p_z = -1 * m.es_con[c_source,y,x,t]
                piecewise_constraints = set_piecewise_constraints(model,y,x,t,
                    m.es_prod[c_prod,y,x,t],m.e_cap[y,x],p_z,
                    x_pieces,z_pieces)
                # create the Pyomo Block of variables and constraints
                setattr(m,"pc_{}_{}_{}".format(y,x,timestamp), piecewise_constraints)
                # skip creating c_es_prod_max for this technology
                return po.Constraint.Skip
        else:
            return (m.es_prod[c_prod, y, x, t]
            == -1 * m.es_con[c_source, y, x, t] * e_eff)

    def pc_rule(m, y, x, t):
        e_eff = get_constraint_param(model, 'e_eff', y, x, t)
        c = model.get_option(y + '.carrier')
        # TODO once Pyomo supports it,
        # let this update conditionally on param update!
        if po.value(e_eff) == 0:
            e_prod = 0
        else:
            e_prod = m.es_prod[c, y, x, t]/ e_eff
        e_con = m.es_con[c, y, x, t] * e_eff

        # If this tech is in the set of techs allowing rb, include it
        if y in m.y_rb:
            rbs = m.rbs[y, x, t]
        else:
            rbs = 0

        # A) Case where no storage allowed
        s_cap_max = model.get_option(y + '.constraints.s_cap.max', x=x)
        use_s_time = get_constraint_param(model, 'use_s_time', y, x, t)
        if ( s_cap_max == 0 and
                not use_s_time):
            return m.rs[y, x, t] == e_prod + e_con - rbs

        # B) Case where storage is allowed
        else:
            # Ensure that storage-only techs have no rs
            if y in model.get_group_members('storage'):
                rs = 0
            else:
                rs = m.rs[y, x, t]
            m.rs[y, x, t]
            # set up s_minus_one
            # NB: From Pyomo 3.5 to 3.6, order_dict became zero-indexed
            if m.t.order_dict[t] == 0:
                s_minus_one = m.s_init[y, x]
            else:
                s_loss = get_constraint_param(model, 's_loss', y, x, t)
                s_minus_one = (((1 - s_loss)
                                ** time_res.at[model.prev_t(t)])
                               * m.s[y, x, model.prev_t(t)])
            return (m.s[y, x, t] == s_minus_one + rs
                    + rbs - e_prod - e_con)

    # Constraints
    m.c_s_balance_transmission = po.Constraint(m.y_trans, m.x, m.t,
                                               rule=transmission_rule)
    m.c_s_balance_conversion = po.Constraint(m.y_conv, m.x, m.t,
                                             rule=conversion_rule)
    m.c_s_balance_pc = po.Constraint(m.y_pc, m.x, m.t, rule=pc_rule)


def node_constraints_build(model):
    """
    Defines variables:

    * s_cap: installed storage capacity
    * r_cap: installed resource <-> storage conversion capacity
    * e_cap: installed storage <-> grid conversion capacity (gross)
    * e_cap_net: installed storage <-> grid conversion capacity (net)
    * rb_cap: installed secondary resource conversion capacity

    """
    m = model.m
    get_any_option = utils.any_option_getter(model)
    d = model.data

    def get_var_constraint(model_var, y, var, x,
                           _equals=None, _max=None, _min=None,
                           scale=None):

        if not _equals:
            _equals = model.get_option(y + '.constraints.'
                                       + var + '.equals', x=x)
        if not _max:
            _max = model.get_option(y + '.constraints.' + var + '.max', x=x)
        if not _min:
            _min = model.get_option(y + '.constraints.' + var + '.min', x=x)
        if scale:
            _equals = scale * _equals
            _min = scale * _min
            _max = scale * _max
        if _equals:
            if np.isinf(_equals):
                e = exceptions.ModelError
                raise e('Cannot use inf in operational mode, for value of '
                        '{}.{}.equals.{}'.format(y, var, x))
            return model_var == _equals
        elif model.mode == 'operate':
            # Operational mode but 'equals' constraint not set, we use 'max'
            # instead
            # FIXME this should be logged
            if np.isinf(_max):
                return po.Constraint.NoConstraint
            else:
                return model_var == _max
        else:
            if np.isinf(_max):
                _max = None  # to disable upper bound
            if _min == 0 and _max is None:
                return po.Constraint.NoConstraint
            else:
                return (_min, model_var, _max)

    # Variables
    m.s_cap = po.Var(m.y_pc, m.x, within=po.NonNegativeReals)
    m.r_cap = po.Var(m.y_def_r, m.x, within=po.NonNegativeReals)
    # m.e_cap = po.Var(m.y, m.x, within=po.NonNegativeReals) # defined earlier to work with piecewise definition
    m.e_cap_net = po.Var(m.y, m.x, within=po.NonNegativeReals)
    m.rb_cap = po.Var(m.y_rb, m.x, within=po.NonNegativeReals)

    # Constraint rules
    def c_s_cap_rule(m, y, x):
        if model.get_option(y + '.constraints.use_s_time', x=x):
            scale = model.get_option(y + '.constraints.e_cap_scale', x=x)
            s_time_max = model.get_option(y + '.constraints.s_time.max', x=x)
            e_cap = model.get_option(y + '.constraints.e_cap.equals', x=x)
            if not e_cap:
                e_cap = model.get_option(y + '.constraints.e_cap.max', x=x)
            e_eff_ref = model.get_eff_ref('e', y) #look into updating this to account for time dependancy (defined in core.py)
            s_cap_max = s_time_max * e_cap * scale / e_eff_ref
        else:
            s_cap_max = None

        return get_var_constraint(m.s_cap[y, x], y, 's_cap', x, _max=s_cap_max)

    def c_r_cap_rule(m, y, x):
        if get_any_option(y + '.constraints.r_cap_equals_e_cap', x=x):
            return m.r_cap[y, x] == m.e_cap[y, x]
        else:
            return get_var_constraint(m.r_cap[y, x], y, 'r_cap', x)

    def c_r_area_rule(m, y, x):
        area_per_cap = get_any_option(y + '.constraints.r_area_per_e_cap', x=x)
        if area_per_cap:
            return m.r_area[y, x] == m.e_cap[y, x] * area_per_cap
        else:
            e_cap_max = get_any_option(y + '.constraints.e_cap.max', x=x)
            if e_cap_max == 0:
                # If a technology has no e_cap here, we force r_area to zero,
                # so as not to accrue spurious costs
                return m.r_area[y, x] == 0
            elif model.get_option(y + '.constraints.r_area.max', x=x) is False:
                return m.r_area[y, x] == 1
            else:
                return get_var_constraint(m.r_area[y, x], y, 'r_area', x)

    def c_e_cap_rule(m, y, x):
        # First check whether this tech is allowed at this location
        if not model._locations.at[x, y] == 1:
            return m.e_cap[y, x] == 0
        else:
            e_cap_scale = get_any_option(y + '.constraints.e_cap_scale', x=x)
            return get_var_constraint(m.e_cap[y, x], y, 'e_cap', x,
                                      scale=e_cap_scale)

    def c_e_cap_gross_net_rule(m, y, x):
        # Existence of this rule currently means that you can't load c_eff from file.
        # Changed for the time being to raise error if someone tries loading c_eff from file
        # to remind me that it is an issue - might need c_eff_ref like e_eff_ref
        c_eff = get_any_option(y + '.constraints.c_eff', x=x)
        if isinstance(c_eff,str):
            e = exceptions.ModelError
            raise e('can\'t load c_eff from file (for '
                        '{} at {}) until '
                        'c_e_cap_gross_net_rule '
                        'is updated'.format(y, x))
        return m.e_cap[y, x] * c_eff == m.e_cap_net[y, x]

    def c_rb_cap_rule(m, y, x):
        follow = get_any_option(y + '.constraints.rb_cap_follow', x=x)
        mode = get_any_option(y + '.constraints.rb_cap_follow_mode', x=x)

        # First deal with the special case of ``rb_cap_follow`` being set
        if follow:
            if follow == 'r_cap':
                rb_cap_val = m.r_cap[y, x]
            elif follow == 'e_cap':
                rb_cap_val = m.e_cap[y, x]
            elif follow is not False:
                # Raise an error to make sure follows isn't accidentally set to
                # something invalid
                e = exceptions.ModelError
                raise e('rb_cab_follow set to invalid value at '
                        '({}, {}): {}'.format(y, x, follow))

            if mode == 'max':
                return m.rb_cap[y, x] <= rb_cap_val
            elif mode == 'equals':
                return m.rb_cap[y, x] == rb_cap_val

        else:  # If ``rb_cap_follow`` not set, set up standard constraints
            return get_var_constraint(m.rb_cap[y, x], y, 'rb_cap', x)

    # Constraints
    m.c_s_cap = po.Constraint(m.y_pc, m.x, rule=c_s_cap_rule)
    m.c_r_cap = po.Constraint(m.y_def_r, m.x, rule=c_r_cap_rule)
    m.c_r_area = po.Constraint(m.y_def_r, m.x, rule=c_r_area_rule)
    m.c_e_cap = po.Constraint(m.y, m.x, rule=c_e_cap_rule)
    m.c_e_cap_gross_net = po.Constraint(m.y, m.x, rule=c_e_cap_gross_net_rule)
    m.c_rb_cap = po.Constraint(m.y_rb, m.x, rule=c_rb_cap_rule)


def node_constraints_operational(model):
    m = model.m
    time_res = model.data['_time_res'].to_series()

    # Constraint rules
    def c_rs_max_upper_rule(m, y, x, t):
        return m.rs[y, x, t] <= time_res.at[t] * m.r_cap[y, x]

    def c_rs_max_lower_rule(m, y, x, t):
        return m.rs[y, x, t] >= -1 * time_res.at[t] * m.r_cap[y, x]

    def c_es_prod_max_rule(m, c, y, x, t):
        e_prod = get_constraint_param(model, 'e_prod', y, x, t)
        if (e_prod is True and
                c == model.get_option(y + '.carrier')):
            return m.es_prod[c, y, x, t] <= time_res.at[t] * m.e_cap[y, x]
        elif (e_prod is True and
                c == model.get_option(y + '.carrier_2')):
            #case where a second carrier is defined, e.g. for a combined heat and power station
            #would be better to have a set of techs with a secondary carrier at start
            c_1 = model.get_option(y + '.carrier')
            c_2 = c
            htp = model.get_option(y + '.constraints.htp')
            if 'piecewise.htp' in model.get_option(y).as_dict_flat():
                # in case htp is a piecewise function
                timestamp = int(time.mktime(t.timetuple()))
                x_pieces = model.config_model.pieces[y].htp.c_1
                z_pieces = model.config_model.pieces[y].htp.c_2
                piecewise_constraints = set_piecewise_constraints(model,y,x,t,
                    m.es_prod[c_1,y,x,t],m.e_cap[y,x],m.es_prod[c_2,y,x,t],
                    x_pieces,z_pieces)
                # create the Pyomo Block of variables and constraints
                setattr(m,"pc_{}_{}_{}_htp".format(y,x,timestamp), piecewise_constraints)
                # skip creating c_es_prod_max for this technology
                return po.Constraint.Skip
            else:
                return (m.es_prod[c_2, y, x, t]
                == m.es_prod[c_1, y, x, t] * htp)
        else:
            return m.es_prod[c, y, x, t] == 0

    def c_es_prod_min_rule(m, c, y, x, t):
        min_use = get_constraint_param(model, 'e_cap_min_use', y, x, t)
        if (min_use and c == model.get_option(y + '.carrier')):
            return (m.es_prod[c, y, x, t]
                    >= time_res.at[t] * m.e_cap[y, x] * min_use)
        else:
            return po.Constraint.NoConstraint

    def c_es_con_max_rule(m, c, y, x, t):
        e_con = get_constraint_param(model, 'e_con', y, x, t)
        if y in m.y_conv:
            return po.Constraint.Skip
        else:
            carrier = '.carrier'
        if (e_con is True and
                c == model.get_option(y + carrier)):
            return m.es_con[c, y, x, t] >= (-1 * time_res.at[t]
                                            * m.e_cap[y, x])
        else:
            return m.es_con[c, y, x, t] == 0

    def c_s_max_rule(m, y, x, t):
        return m.s[y, x, t] <= m.s_cap[y, x]

    def c_rbs_max_rule(m, y, x, t):
        rb_startup = get_constraint_param(model, 'rb_startup_only', y, x, t)
        if (rb_startup
                and t >= model.data.startup_time_bounds):
            return m.rbs[y, x, t] == 0
        else:
            return m.rbs[y, x, t] <= time_res.at[t] * m.rb_cap[y, x]

    # Constraints
    m.c_rs_max_upper = po.Constraint(m.y_def_r, m.x, m.t,
                                     rule=c_rs_max_upper_rule)
    m.c_rs_max_lower = po.Constraint(m.y_def_r, m.x, m.t,
                                     rule=c_rs_max_lower_rule)
    m.c_es_prod_max = po.Constraint(m.c, m.y, m.x, m.t,
                                    rule=c_es_prod_max_rule)
    m.c_es_prod_min = po.Constraint(m.c, m.y, m.x, m.t,
                                    rule=c_es_prod_min_rule)
    m.c_es_con_max = po.Constraint(m.c, m.y, m.x, m.t,
                                   rule=c_es_con_max_rule)
    m.c_s_max = po.Constraint(m.y_pc, m.x, m.t,
                              rule=c_s_max_rule)
    m.c_rbs_max = po.Constraint(m.y_rb, m.x, m.t,
                                rule=c_rbs_max_rule)


def node_constraints_transmission(model):
    """
    Constrains e_cap symmetrically for transmission nodes.

    """
    m = model.m

    # Constraint rules
    def c_trans_rule(m, y, x):
        y_remote, x_remote = transmission.get_remotes(y, x)
        if y_remote in m.y_trans:
            return m.e_cap[y, x] == m.e_cap[y_remote, x_remote]
        else:
            return po.Constraint.NoConstraint

    # Constraints
    m.c_transmission_capacity = po.Constraint(m.y_trans, m.x,
                                              rule=c_trans_rule)


def node_parasitics(model):
    """
    Additional variables and constraints for plants with internal parasitics.

    Defines variables:

    * ec_prod: storage -> carrier after parasitics (+ production)
    * ec_con: storage <- carrier after parasitics (- consumption)

    """
    m = model.m
    get_any_option = utils.any_option_getter(model)

    # Variables
    m.ec_prod = po.Var(m.c, m.y_p, m.x, m.t, within=po.NonNegativeReals)
    m.ec_con = po.Var(m.c, m.y_p, m.x, m.t, within=po.NegativeReals)

    # Constraint rules
    def c_ec_prod_rule(m, c, y, x, t):
        return (m.ec_prod[c, y, x, t]
                == m.es_prod[c, y, x, t]
                * get_any_option(y + '.constraints.c_eff', x=x))

    def c_ec_con_rule(m, c, y, x, t):                                       
        if y in m.y_trans or y in m.y_conv:                                 
            # Ensure that transmission and conversion technologies          
            # do not double count c_eff                                     
            c_eff = 1.0                                                     
        else:                                                               
            c_eff = get_any_option(y + '.constraints.c_eff', x=x)      
        if c_eff > 0:                                                       
            return (m.ec_con[c, y, x, t]                                    
                    == m.es_con[c, y, x, t]                                 
                    / c_eff)                                                
        else:                                                               
            return (m.ec_con[c, y, x, t] == 0)   

    # Constraints
    m.c_ec_prod = po.Constraint(m.c, m.y_p, m.x, m.t, rule=c_ec_prod_rule)
    m.c_ec_con = po.Constraint(m.c, m.y_p, m.x, m.t, rule=c_ec_con_rule)


def node_costs(model):
    """
    Defines variables:

    * cost: total costs
    * cost_con: construction costs
    * cost_op_fixed: fixed operation costs
    * cost_op_var: variable operation costs
    * cost_op_fuel: primary resource fuel costs
    * cost_op_rb: secondary resource fuel costs
    * revenue_var: variable revenue (operation + fuel)
    * revenue_fixed: fixed revenue
    * revenue: total revenue

    """
    m = model.m
    time_res = model.data['_time_res'].to_series()
    weights = model.data['_weights'].to_series()

    cost_getter = utils.cost_getter(model.get_option)
    depreciation_getter = utils.depreciation_getter(model.get_option)
    cost_per_distance_getter = utils.cost_per_distance_getter(model.get_option)

    @utils.memoize
    def _depreciation_rate(y, k):
        return depreciation_getter(y, k)

    @utils.memoize
    def _cost(cost, y, k, x=None):
        return cost_getter(cost, y, k, x=x)

    def _revenue(cost, y, k, x=None):
        return cost_getter(cost, y, k, x=x, costs_type='revenue')

    @utils.memoize
    def _cost_per_distance(cost, y, k, x):
        return cost_per_distance_getter(cost, y, k, x)

    # Variables
    m.cost = po.Var(m.y, m.x, m.kc, within=po.NonNegativeReals)
    m.cost_con = po.Var(m.y, m.x, m.kc, within=po.NonNegativeReals)
    m.cost_con_fixed = po.Var(m.y, m.x, m.kc, within=po.NonNegativeReals)
    m.cost_op_fixed = po.Var(m.y, m.x, m.kc, within=po.NonNegativeReals)
    m.cost_op_variable = po.Var(m.y, m.x, m.kc, within=po.NonNegativeReals)
    m.cost_op_var = po.Var(m.y, m.x, m.t, m.kc, within=po.NonNegativeReals)
    m.cost_op_fuel = po.Var(m.y, m.x, m.t, m.kc, within=po.NonNegativeReals)
    m.cost_op_rb = po.Var(m.y, m.x, m.t, m.kc, within=po.NonNegativeReals)
    m.revenue_var = po.Var(m.y, m.x, m.t, m.kr, within=po.NonNegativeReals)
    m.revenue_fixed = po.Var(m.y, m.x, m.kr, within=po.NonNegativeReals)
    m.revenue = po.Var(m.y, m.x, m.kr, within=po.NonNegativeReals)
    m.purchased = po.Var(m.y, m.x, within=po.Binary)

    # Constraint rules
    def c_cost_rule(m, y, x, k):
        return (
            m.cost[y, x, k] ==
            m.cost_con[y, x, k] +
            m.cost_op_fixed[y, x, k] +
            m.cost_op_variable[y, x, k]            
        )

    def c_cost_con_rule(m, y, x, k):
        if y in m.y_pc:
            cost_s_cap = _cost('s_cap', y, k, x) * m.s_cap[y, x]
        else:
            cost_s_cap = 0

        if y in m.y_def_r:
            cost_r_cap = _cost('r_cap', y, k, x) * m.r_cap[y, x]
            cost_r_area = _cost('r_area', y, k, x) * m.r_area[y, x]
        else:
            cost_r_cap = 0
            cost_r_area = 0

        if y in m.y_trans:
            # Divided by 2 for transmission techs because construction costs
            # are counted at both ends
            cost_e_cap = (_cost('e_cap', y, k, x)
                          + _cost_per_distance('e_cap', y, k, x)) / 2
        else:
            cost_e_cap = _cost('e_cap', y, k, x)

        if y in m.y_rb:
            cost_rb_cap = _cost('rb_cap', y, k, x) * m.rb_cap[y, x]
        else:
            cost_rb_cap = 0

        return (
            m.cost_con[y, x, k] == _depreciation_rate(y, k) *
            (sum(time_res * weights) / 8760) *
            (cost_s_cap + cost_r_cap + cost_r_area + cost_rb_cap +
             cost_e_cap * m.e_cap[y, x] + m.cost_con_fixed[y, x, k])
        )
    
    def purchased_rule(m, y, x): #Binary result of whether a tech has non-zero production at any point in time horizon
        prod = sum(m.es_prod[c,y,x,t] for c in m.c for t in m.t) - sum(m.es_con[c,y,x,t] for c in m.c for t in m.t)
        return (m.purchased[y,x] >= prod / 1e10)

    def cost_con_fixed_rule(m, y, x, k): # Cost incurred as fixed value irrespective of technology size
        cap_fixed = _cost('cap_fixed', y, k, x)
        if y in m.y_trans:
            # Divided by 2 for transmission techs because construction costs
            # are counted at both ends
            cap_fixed = cap_fixed/2 
        return (m.cost_con_fixed[y,x,k] == m.purchased[y,x] * cap_fixed)

    def c_cost_op_fixed_rule(m, y, x, k):
        if y in m.y:
            return (m.cost_op_fixed[y, x, k] ==
                    _cost('om_frac', y, k, x) * m.cost_con[y, x, k]
                    + (_cost('om_fixed', y, k, x) * m.e_cap[y, x] *
                       (sum(time_res * weights) / 8760)))
        else:
            return m.cost_op_fixed[y, x, k] == 0

    def c_cost_op_variable_rule(m, y, x, k):
        return (
            m.cost_op_variable[y, x, k] ==
            sum(
                m.cost_op_var[y, x, t, k] +
                m.cost_op_fuel[y, x, t, k] +
                m.cost_op_rb[y, x, t, k]
                for t in m.t
            )
        )

    def c_cost_op_var_rule(m, y, x, t, k):
        # Note: only counting es_prod for operational costs.
        # This should generally be a reasonable assumption to make.
        if y in m.y:
            carrier = model.get_option(y + '.carrier')
            return (
                m.cost_op_var[y, x, t, k] ==
                get_cost_param(model,'om_var',k,y,x,t) *
                weights.loc[t] *
                m.es_prod[carrier, y, x, t]
            )
        else:
            return m.cost_op_var[y, x, t, k] == 0

    def c_cost_op_fuel_rule(m, y, x, t, k):
        r_eff = get_constraint_param(model, 'r_eff', y, x, t)
        om_fuel = get_cost_param(model,'om_fuel',k,y,x,t)
        if po.value(r_eff) > 0:
            # Dividing by r_eff here so we get the actual r used, not the rs
            # moved into storage...
            return (
                m.cost_op_fuel[y, x, t, k] ==
                om_fuel *
                weights.loc[t] *
                (m.rs[y, x, t] / r_eff)
            )
        else: #in case r_eff is zero, to avoid an infinite value for cost_op_fuel
            return m.cost_op_fuel[y, x, t, k] == 0

    def c_cost_op_rb_rule(m, y, x, t, k):
        rb_eff = get_constraint_param(model, 'rb_eff', y, x, t)
        if y in m.y_rb and po.value(rb_eff) > 0:
            return (
                m.cost_op_rb[y, x, t, k] ==
                get_cost_param(model,'om_rb',k,y,x,t) *
                weights.loc[t] *
                (m.rbs[y, x, t] / rb_eff)
            )
        else:
            return m.cost_op_rb[y, x, t, k] == 0

    def c_revenue_var_rule(m, y, x, t, k):
        carrier = model.get_option(y + '.carrier')
        sub_var = get_revenue_param(model,'sub_var',k,y,x,t)
        if y in m.y_demand:
            return (m.revenue_var[y, x, t, k] ==
                sub_var * weights.loc[t]
                * -m.es_con[carrier, y, x, t])
        else:
            return (m.revenue_var[y, x, t, k] ==
                sub_var * weights.loc[t]
                * m.es_prod[carrier, y, x, t])

    def c_revenue_fixed_rule(m, y, x, k):
        revenue = (sum(time_res * weights) / 8760 * 
            (_revenue('sub_cap', y, k, x) 
            * _depreciation_rate(y, k)
            + _revenue('sub_annual', y, k, x)))
        if y in m.y_demand and revenue > 0:
            e = exceptions.ModelError
            raise e('Cannot receive fixed revenue at a demand node, i.e. '
                    '{}'.format(y))
        else:
            return (m.revenue_fixed[y, x, k] == 
             revenue * m.e_cap[y, x])


    def c_revenue_rule(m, y, x, k):
        return (m.revenue[y, x, k] == m.revenue_fixed[y, x, k] +
            sum(m.revenue_var[y, x, t, k] for t in m.t))


    # Constraints
    m.c_cost = po.Constraint(m.y, m.x, m.kc, rule=c_cost_rule)
    m.c_cost_con = po.Constraint(m.y, m.x, m.kc, rule=c_cost_con_rule)
    m.c_cost_con_fixed = po.Constraint(m.y, m.x, m.kc, rule=cost_con_fixed_rule)
    m.c_cost_op_fixed = po.Constraint(m.y, m.x, m.kc, rule=c_cost_op_fixed_rule)
    m.c_cost_op_variable = po.Constraint(m.y, m.x, m.kc, rule=c_cost_op_variable_rule)
    m.c_cost_op_var = po.Constraint(m.y, m.x, m.t, m.kc, rule=c_cost_op_var_rule)
    m.c_cost_op_fuel = po.Constraint(m.y, m.x, m.t, m.kc, rule=c_cost_op_fuel_rule)
    m.c_cost_op_rb = po.Constraint(m.y, m.x, m.t, m.kc, rule=c_cost_op_rb_rule)
    m.c_revenue_var = po.Constraint(m.y, m.x, m.t, m.kr, rule=c_revenue_var_rule)
    m.c_revenue_fixed = po.Constraint(m.y, m.x, m.kr, rule=c_revenue_fixed_rule)
    m.c_revenue = po.Constraint(m.y, m.x, m.kr, rule=c_revenue_rule)
    m.c_purchased = po.Constraint(m.y, m.x, rule=purchased_rule)
    

def model_constraints(model):
    m = model.m

    @utils.memoize
    def get_parents(level):
        return list(model._locations[model._locations._level == level].index)

    @utils.memoize
    def get_children(parent, childless_only=True):
        """
        If childless_only is True, only children that have no children
        themselves are returned.

        """
        locations = model._locations
        children = list(locations[locations._within == parent].index)
        if childless_only:  # FIXME childless_only param needs tests
            children = [i for i in children if len(get_children(i)) == 0]
        return children

    # Constraint rules
    def c_system_balance_rule(m, c, x, t):
        # Balacing takes place at top-most (level 0) locations, as well
        # as within any lower-level locations that contain children
        if (model._locations.at[x, '_level'] == 0
                or len(get_children(x)) > 0):
            family = get_children(x) + [x]  # list of children + parent
            balance = (sum(m.es_prod[c, y, xs, t]
                           for xs in family for y in m.y_np)
                       + sum(m.ec_prod[c, y, xs, t]
                             for xs in family for y in m.y_p)
                       + sum(m.es_con[c, y, xs, t]
                             for xs in family for y in m.y_np)
                       + sum(m.ec_con[c, y, xs, t]
                             for xs in family for y in m.y_p))
            if c == 'power':
                return balance == 0
            else:  # e.g. for heat, 5% additional allowed
                return (0, balance,50)
        else:
            return po.Constraint.NoConstraint

    # Constraints
    m.c_system_balance = po.Constraint(m.c, m.x, m.t,
                                       rule=c_system_balance_rule)

def set_piecewise_constraints(model,y, x, t, p_x, p_y, p_z, x_pieces, z_pieces, num=2):
    """
    Generate a Pyomo block containing required variables and constraints for 
    2D piecewise function.

    Args:
        y: tech
        x: location
        t: timestep
        p_x: Pyomo variable as input to piecewise function, pieces taken from piece_dict
            Tested only with es_prod normalised against e_cap
        p_y:  Pyomo variable as input to piecewise function, pieces created in this function
            Only works with e_cap, which is cut into 5 pieces (can change number of pieces in `generate delaunay`)
        p_z: Pyomo variable to take value of output, pieces taken from piece_dict
            Tested only with es_con & es_prod where a secondary carrier is defined by a heat to power ratio
        x_pieces: list of values describing curve p_x follows, found at e.g. model.config_model.pieces.y.power.prod
            Normalised w.r.t. y-axis 
        z_pieces: list of values describing curve p_z follows
            Normalised w.r.t. y-axis 
        num: number of pieces to create a Delaunay triangulation from p_y maximum and minimum values
    
    Based on:
    https://projects.coin-or.org/Pyomo/browser/pyomo.data/trunk/pyomo/data/pyomobook/scripts/advanced?rev=10872&order=name
    """
    def generate_delaunay(x_pieces, y, num, x=None):
        """
        Generate a Delaunay triangulation of the 2-dimensional
        bounded variable domain given the array of Pyomo
        variables [x, y]. 
        **Currently only works with e_cap**
        Args:
            x_pieces: points of the piecewise curve associated with p_x, 
                taken from piece_dict 
            y: tech
            x: location
            num: The number of grid points to generate for each variable

        Requires both numpy and scipy.spatial be available.
        """
        import scipy.spatial
    
        linegrids = []
        e_cap = model.get_option(y + '.constraints.e_cap.max', x=x)
        cap = np.linspace(0, e_cap, num)
        if e_cap == 0:
            return None
        for c in cap:
            for i in x_pieces:
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
    

    def _get_z_vals(tri, x_pieces, z_pieces):
        """
        Creates a matrix of values defining what z would be at each [x,y] point
        """
        x_array, y_array = np.transpose(tri.points)
        y_array_T = np.reshape(y_array,(num,len(x_pieces)))
        z_vals = []
        for i in range(len(y_array_T)):
            # need to multiply the value given in piecewise dictionary by y axis (e_cap)
            # as pieces in dictionary are normalised w.r.t. y axis
            z_vals.append(np.multiply(y_array_T[i], z_pieces))
        z_vals = [item for sublist in z_vals for item in sublist]

        return z_vals
    
    tri = generate_delaunay(x_pieces, y, x=x, num=num)

    if not tri: # case where e_cap.max = 0
        return None
    z_vals = _get_z_vals(tri, x_pieces, z_pieces)
    return BuildPiecewiseND([p_x, p_y], p_z,
     tri, z_vals)
    