"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

objective.py
~~~~~~~~~~~~

Objective functions.

"""

import pyomo.core as po  # pylint: disable=import-error


def get_y(loc_tech):
    return loc_tech.split(":", 1)[1]

def objective_cost_minimization(model):
    """
    Minimizes total system monetary cost.
    Used as a default if a model does not specify another objective.

    """
    m = model.m

    def obj_rule(m):
        return (sum(model.get_option(get_y(loc_tech) + '.weight') *
                       m.cost[loc_tech, 'monetary', s]
                       for loc_tech in m.loc_tech for s in m.scenarios
                   )
               )

    m.obj = po.Objective(sense=po.minimize, rule=obj_rule)
    m.obj.domain = po.Reals

def objective_robust_cost_minimization(model):
    """
    Minimizes total system monetary cost.
    Used as a default if a model does not specify another objective.

    """
    m = model.m
    # beta is the degree of risk aversion, which can be anything from 0 (no
    # risk aversion) to infinity (infinite risk aversion)
    beta = model.config_run.robust_optimisation.cvar_parameters.get('beta', 5)
    # alpha is the percentile of the cost distribution associated with Value at
    # Risk (VaR)
    alpha = model.config_run.robust_optimisation.cvar_parameters.get('alpha', 0.95)

    def cost_equation(m, s):
        cost_sum = sum(model.get_option(get_y(loc_tech) + '.weight') *
                       m.cost[loc_tech, 'monetary', s] for loc_tech in m.loc_tech)
        return cost_sum

    def obj_rule(m):
        return (sum(m.probability[s] * cost_equation(m, s) for s in m.scenarios)
                + beta * (m.xi + 1/(1 - alpha) *
                          sum(m.probability[s] *
                              m.eta[s] for s in m.scenarios)
                         )
                )

    m.obj = po.Objective(sense=po.minimize, rule=obj_rule)
    m.obj.domain = po.Reals
