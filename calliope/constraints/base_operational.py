"""
Copyright (C) 2013-2016 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

base_operational.py
~~~~~~~

Basic model constraints when working in operational mode.

"""

import pyomo.core as po

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

    if param_string in model.data:
        return getattr(model.m, param_string)[y, x, t]
    else:
        return model.get_option(y + '.constraints.' + param_string, x=x)

def get_cost_param(model, cost, k, y, x, t, cost_type='cost'):
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

    cost_getter = utils.cost_getter(model.get_option)

    @utils.memoize
    def _cost(cost, y, k, x=None):
        return cost_getter(cost, y, k, x=x)

    @utils.memoize
    def _revenue(cost, y, k, x=None):
        return cost_getter(cost, y, k, x=x, costs_type='revenue')

    if cost_type == 'cost':
        param_string = 'costs_' + k + '_' + cost #format stored in model.data
        if param_string in model.data:
            return getattr(model.m, param_string)[y, x, t]
        else:
            return _cost(cost, y, k, x)
    elif cost_type == 'revenue':
        param_string = 'revenue_' + k + '_' + cost #format stored in model.data
        if param_string in model.data:
            return getattr(model.m, param_string)[y, x, t]
        else:
            return _revenue(cost, y, k, x)

def generate_variables(model):
    """
    Defines variables:

    * rs: resource <-> storage (+ production, - consumption)
    * r_area: resource collector area
    * rbs: secondary resource -> storage (+ production)
    * es_prod: storage -> carrier (+ production)
    * es_con: storage <- carrier (- consumption)

    * cost: total costs
    * cost_fixed: all fixed costs
    * cost_var: all operational costs
    * revenue_var: variable revenue (operation + fuel)
    * revenue_fixed: fixed revenue
    * revenue: total revenue

    """

    m = model.m

    # Unit commitment
    m.rs = po.Var(m.y, m.x, m.t, within=po.Reals)
    m.rbs = po.Var(m.y_rb, m.x, m.t, within=po.NonNegativeReals)
    m.s = po.Var(m.y_pc, m.x, m.t, within=po.NonNegativeReals)
    m.c_prod = po.Var(m.c, m.y, m.x, m.t, within=po.NonNegativeReals)
    m.c_con = po.Var(m.c, m.y, m.x, m.t, within=po.NegativeReals)

    # Costs/revenue
    m.cost_var = po.Var(m.y, m.x, m.t, m.kc, within=po.NonNegativeReals)
    m.cost_fixed = po.Var(m.y, m.x, m.kc, within=po.NonNegativeReals)
    m.cost = po.Var(m.y, m.x, m.kc, within=po.NonNegativeReals)
    m.revenue_var = po.Var(m.y, m.x, m.t, m.kr, within=po.NonNegativeReals)
    m.revenue_fixed = po.Var(m.y, m.x, m.kr, within=po.NonNegativeReals)
    m.revenue = po.Var(m.y, m.x, m.kr, within=po.NonNegativeReals)

def node_resource(model):

    m = model.m
    get_any_option = utils.any_option_getter(model)

    # Constraint rules
    def c_rs_rule(m, y, x, t):
        r_scale = get_any_option(y + '.constraints.r_scale', x=x)
        r_eff = get_constraint_param(model, 'r_eff', y, x, t)
        force_r = get_constraint_param(model, 'force_r', y, x, t)
        r_area = model.get_cap_constraint('r_area', y, x)
        r_avail = (m.r[y, x, t]
                   * r_scale
                   * r_area
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

    # Constraint rules
    def transmission_rule(m, y, x, t):
        y_remote, x_remote = transmission.get_remotes(y, x)
        e_eff = get_constraint_param(model, 'e_eff', y, x, t)
        if y_remote in m.y_trans:
            c = model.get_option(y + '.carrier')
            return (m.c_prod[c, y, x, t]
                    == -1 * m.c_con[c, y_remote, x_remote, t]
                    * e_eff
                    * get_e_eff_per_distance(model, y, x))
        else:
            return po.Constraint.NoConstraint

    def conversion_rule(m, y, x, t):
        c_out = model.get_option(y + '.carrier')
        c_in = model.get_option(y + '.source_carrier')
        e_eff = get_constraint_param(model, 'e_eff', y, x, t)
        return (m.c_prod[c_out, y, x, t]
                == -1 * m.c_con[c_in, y, x, t] * e_eff)

    def pc_rule(m, y, x, t):
        e_eff = get_constraint_param(model, 'e_eff', y, x, t)
        p_eff = model.get_option(y + '.constraints.p_eff', x=x)
        total_eff = e_eff * p_eff
        # TODO once Pyomo supports it,
        # let this update conditionally on param update!
        if po.value(e_eff) == 0:
            c_prod = 0
        else:
            c_prod = sum(m.c_prod[c, y, x, t] for c in m.c) / total_eff
        c_con = sum(m.c_con[c, y, x, t] for c in m.c) * total_eff

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
            return m.rs[y, x, t] == c_prod + c_con - rbs

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
                    + rbs - c_prod - c_con)

    # Constraints
    m.c_s_balance_transmission = po.Constraint(m.y_trans, m.x, m.t,
                                               rule=transmission_rule)
    m.c_s_balance_conversion = po.Constraint(m.y_conv, m.x, m.t,
                                             rule=conversion_rule)
    m.c_s_balance_pc = po.Constraint(m.y_pc, m.x, m.t, rule=pc_rule)

def node_constraints_operational(model):
    m = model.m
    time_res = model.data['_time_res'].to_series()

    # Constraint rules
    def c_rs_max_upper_rule(m, y, x, t):
        r_cap = model.get_cap_constraint('r_cap', y, x)
        if not r_cap == None:
            return (m.rs[y, x, t] <= time_res.at[t] * r_cap)
        else:
            return po.Constraint.Skip

    def c_rs_max_lower_rule(m, y, x, t):
        r_cap = model.get_cap_constraint('r_cap', y, x)
        if not r_cap == None:
            return (m.rs[y, x, t] >= -1 * time_res.at[t]
                                * r_cap)
        else:
            return po.Constraint.Skip

    def c_prod_max_rule(m, c, y, x, t):
        c_prod = get_constraint_param(model, 'e_prod', y, x, t)
        p_eff = model.get_option(y + '.constraints.p_eff', x=x)
        if (c_prod is True and
                c == model.get_option(y + '.carrier')):
            e_cap = model.get_cap_constraint('e_cap', y, x)
            if not e_cap == None:
                return m.c_prod[c, y, x, t] <= time_res.at[t] * e_cap * p_eff
            else:
                return po.Constraint.Skip
        else:
            return m.c_prod[c, y, x, t] == 0

    def c_prod_min_rule(m, c, y, x, t):
        min_use = get_constraint_param(model, 'e_cap_min_use', y, x, t)
        if (min_use and c == model.get_option(y + '.carrier')):
            e_cap = model.get_cap_constraint('e_cap', y, x, sense='min')
            return (m.c_prod[c, y, x, t]
                    >= time_res.at[t] * e_cap * min_use)
        else:
            return po.Constraint.Skip

    def c_con_max_rule(m, c, y, x, t):
        c_con = get_constraint_param(model, 'e_con', y, x, t)
        p_eff = model.get_option(y + '.constraints.p_eff', x=x)
        if y in m.y_conv:
            return po.Constraint.Skip
        else:
            carrier = '.carrier'
        if (c_con is True and
                c == model.get_option(y + carrier)):
            e_cap = model.get_cap_constraint('e_cap', y, x)
            if not e_cap == None:
                return m.c_con[c, y, x, t] >= (-1 * time_res.at[t]
                                            * e_cap * p_eff)
            else:
                return po.Constraint.Skip
        else:
            return m.c_con[c, y, x, t] == 0

    def c_s_max_rule(m, y, x, t):
        s_cap = model.get_cap_constraint('s_cap', y, x)
        if not s_cap == None:
            return m.s[y, x, t] <= s_cap
        else:
            return po.Constraint.Skip

    def c_rbs_max_rule(m, y, x, t):
        rb_startup = get_constraint_param(model, 'rb_startup_only', y, x, t)
        if (rb_startup
                and t >= model.data.startup_time_bounds):
            return m.rbs[y, x, t] == 0
        else:
            rb_cap = model.get_cap_constraint('rb_cap', y, x)
            if not rb_cap == None:
                return m.rbs[y, x, t] <= time_res.at[t] * rb_cap
            else:
                return po.Constraint.Skip

    # Constraints
    m.c_rs_max_upper = po.Constraint(m.y_def_r, m.x, m.t,
                                     rule=c_rs_max_upper_rule)
    m.c_rs_max_lower = po.Constraint(m.y_def_r, m.x, m.t,
                                     rule=c_rs_max_lower_rule)
    m.c_prod_max = po.Constraint(m.c, m.y, m.x, m.t,
                                    rule=c_prod_max_rule)
    m.c_prod_min = po.Constraint(m.c, m.y, m.x, m.t,
                                    rule=c_prod_min_rule)
    m.c_con_max = po.Constraint(m.c, m.y, m.x, m.t,
                                   rule=c_con_max_rule)
    m.c_s_max = po.Constraint(m.y_pc, m.x, m.t,
                              rule=c_s_max_rule)
    m.c_rbs_max = po.Constraint(m.y_rb, m.x, m.t,
                                rule=c_rbs_max_rule)

def node_costs(model):

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

    @utils.memoize
    def _revenue(cost, y, k, x=None):
        return cost_getter(cost, y, k, x=x, costs_type='revenue')

    @utils.memoize
    def _cost_per_distance(cost, y, k, x):
        return cost_per_distance_getter(cost, y, k, x)

    # Constraint rules

    def c_cost_rule(m, y, x, k):
        return (
            m.cost[y, x, k] ==
            m.cost_fixed[y, x, k] +
            sum(m.cost_var[y, x, t, k] for t in m.t)
        )

    def c_cost_fixed_rule(m, y, x, k):
        if y in m.y_pc:
            s_cap = model.get_cap_constraint('s_cap', y, x)
            cost_s_cap = _cost('s_cap', y, k, x) * s_cap
        else:
            cost_s_cap = 0

        if y in m.y_def_r:
            r_area = model.get_cap_constraint('r_area', y, x)
            r_cap = model.get_cap_constraint('r_cap', y, x)
            r_area_cost = _cost('r_cap', y, k, x)
            r_cap_cost = _cost('r_area', y, k, x)
            if r_cap == None:
                r_cap = 0
                if r_cap_cost > 0:
                    print('Warning: r_cap cost defined without setting r_cap '
                            'capacity constraint for {}:{}. Cost set to zero.'
                            .format(y, x))
            if r_area == None:
                r_area = 0
                if r_area_cost > 0:
                    print('Warning: r_area cost defined without setting r_area '
                            'capacity constraint for {}:{}. Cost set to zero.'
                            .format(y, x))
            cost_r_cap = r_area_cost * r_cap
            cost_r_area = r_cap_cost * r_area
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
            rb_cap = model.get_cap_constraint('rb_cap', y, x)
            rb_cap_cost = _cost('rb_cap', y, k, x)
            if rb_cap == None:
                rb_cap = 0
                if rb_cap_cost > 0:
                    print('Warning: rb_cap cost defined without setting rb_cap '
                            'capacity constraint for {}:{}. Cost set to zero.'
                            .format(y, x))
            cost_rb_cap = rb_cap_cost * rb_cap
        else:
            cost_rb_cap = 0
        e_cap = model.get_cap_constraint('e_cap', y, x)
        if e_cap == None:
            e_cap = 0
            if cost_e_cap > 0:
                print('Warning: e_cap cost defined without setting e_cap '
                        'capacity constraint for {}:{}. Cost set to zero.'
                        .format(y, x))
        cost_con = (_depreciation_rate(y, k) *
            (sum(time_res * weights) / 8760) *
            (cost_s_cap + cost_r_cap + cost_r_area + cost_rb_cap +
             cost_e_cap * e_cap))

        return (m.cost_fixed[y, x, k] ==
                    _cost('om_frac', y, k, x) * cost_con
                    + (_cost('om_fixed', y, k, x) * e_cap *
                       (sum(time_res * weights) / 8760)) + cost_con)

    def c_cost_var_rule(m, y, x, t, k):
        om_var = get_cost_param(model,'om_var', k, y, x, t)
        carrier = model.get_option(y + '.carrier')
        # Note: only counting c_prod for operational costs.
        # This should generally be a reasonable assumption to make.
        # It might be necessary to remove parasitic losses for this
        # i.e. c_prod --> es_prod.

        cost_op_var = om_var * weights.loc[t] * m.c_prod[carrier, y, x, t]

        # in case r_eff is zero, to avoid an infinite value
        if y in m.y_pc:
            r_eff = get_constraint_param(model, 'r_eff', y, x, t)
            om_fuel = get_cost_param(model,'om_fuel', k, y, x, t)
            if po.value(r_eff) > 0:
                # Dividing by r_eff here so we get the actual r used, not the rs
                # moved into storage...
                cost_op_fuel = (om_fuel * weights.loc[t] * (m.rs[y, x, t] / r_eff))
            else:
                cost_op_fuel = 0
        else:
            cost_op_fuel = 0

        # in case rb_eff is zero, to avoid an infinite value
        if y in m.y_rb:
            rb_eff = get_constraint_param(model, 'rb_eff', y, x, t)
            if po.value(rb_eff) > 0:
                om_rb = get_cost_param(model, 'om_rb', k, y, x, t)
                cost_op_rb = (om_rb * weights.loc[t] * (m.rbs[y, x, t] / rb_eff))
            else:
                cost_op_rb = 0
        else: #in case rb_eff is zero, to avoid an infinite value
            cost_op_rb = 0

        return (m.cost_var[y, x, t, k] == cost_op_var + cost_op_fuel
                                                    + cost_op_rb)

    def c_revenue_var_rule(m, y, x, t, k):
        carrier = model.get_option(y + '.carrier')
        sub_var = get_cost_param(model, 'sub_var', k, y, x, t,
                                 cost_type='revenue')
        if y in m.y_demand:
            return (m.revenue_var[y, x, t, k] ==
                sub_var * weights.loc[t]
                * -m.c_con[carrier, y, x, t])
        else:
            return (m.revenue_var[y, x, t, k] ==
                sub_var * weights.loc[t]
                * m.c_prod[carrier, y, x, t])

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
            e_cap = model.get_cap_constraint('e_cap', y, x)
            if e_cap == None:
                if revenue > 0:
                    print('Warning: sub_cap/sub_annual cost defined without '
                        'setting e_cap capacity constraint for {}:{}. '
                        'Cost set to zero.'.format(y, x))
            return (m.revenue_fixed[y, x, k] ==
             revenue * e_cap)

    def c_revenue_rule(m, y, x, k):
        return (m.revenue[y, x, k] == m.revenue_fixed[y, x, k] +
            sum(m.revenue_var[y, x, t, k] for t in m.t))


    # Constraints
    m.c_cost_var = po.Constraint(m.y, m.x, m.t, m.kc, rule=c_cost_var_rule)
    m.c_cost_fixed = po.Constraint(m.y, m.x, m.kc, rule=c_cost_fixed_rule)
    m.c_cost = po.Constraint(m.y, m.x, m.kc, rule=c_cost_rule)

    m.c_revenue_var = po.Constraint(m.y, m.x, m.t, m.kr, rule=c_revenue_var_rule)
    m.c_revenue_fixed = po.Constraint(m.y, m.x, m.kr, rule=c_revenue_fixed_rule)
    m.c_revenue = po.Constraint(m.y, m.x, m.kr, rule=c_revenue_rule)


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
            balance = (sum(m.c_prod[c, y, xs, t]
                           for xs in family for y in m.y)
                       + sum(m.c_con[c, y, xs, t]
                             for xs in family for y in m.y))
            if c == 'power':
                return balance == 0
            else:
                # e.g. for heat. should probably limit the maximum
                # inbalance allowed for these energy types.
                return balance >= 0
        else:
            return po.Constraint.NoConstraint

    # Constraints
    m.c_system_balance = po.Constraint(m.c, m.x, m.t,
                                       rule=c_system_balance_rule)
