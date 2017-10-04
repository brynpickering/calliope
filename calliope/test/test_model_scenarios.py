import tempfile

from calliope.utils import AttrDict
from . import common
from .common import solver, solver_io, _add_test_path

def create_and_run_model(config_run):
    locations = """
        locations:
            1:
                techs: ['ccgt', 'demand_power']
                override:
                    ccgt:
                        constraints:
                            e_cap.max: 150
                    demand_power:
                        constraints:
                            r: file=demand-static_r.csv
                            r_area.equals: 2000
        links:
    """
    override = AttrDict("""override:""")
    override.set_key('solver', solver)
    override.set_key('solver_io', solver_io)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(locations.encode('utf-8'))
        f.read()
        model = common.simple_model(config_run=config_run,
                                    config_locations=f.name,
                                    override=override,
                                    path=_add_test_path('common/t_scenarios'))
    model.run()
    return model

class TestModel:

    # all constraints are fixed values
    def test_model_fixed(self):
        config_run = """
            mode: plan
            model: ['{techs}', '{locations}']
            subset_t: ['2005-01-01', '2005-01-02']
        """
        model = create_and_run_model(config_run)
        assert str(model.results.solver.termination_condition) == 'optimal'

    def test_model_multivariate_raw_data_hierarchical(self):
        config_run = """
            mode: plan
            model: ['{techs}', '{locations}']
            subset_t: ['2005-01-01', '2005-01-02']
            robust_optimisation:
                scenarios: 5
                probability: equal
                cvar_parameters:
                    alpha: 0.9
                    beta: 5
                uncertain_parameters:
                    r:
                        techs: ['demand_power']
                        demand_power:
                            method:
                                distribution: multivariate_normal
                                raw_data:
                                    folder: raw_data
                                    x_map:
                                        1: electricity_raw_data
                                cluster:
                                    func: hierarchical
                                    arguments:
                                        k: 8"""
        model = create_and_run_model(config_run)
        assert str(model.results.solver.termination_condition) == 'optimal'

    def test_model_multivariate_raw_data_kmeans(self):
        config_run = """
            mode: plan
            model: ['{techs}', '{locations}']
            subset_t: ['2005-01-01', '2005-01-02']
            robust_optimisation:
                scenarios: 5
                probability: equal
                cvar_parameters:
                    alpha: 0.9
                    beta: 5
                uncertain_parameters:
                    r:
                        techs: ['demand_power']
                        demand_power:
                            method:
                                distribution: multivariate_normal
                                raw_data:
                                    folder: raw_data
                                    x_map:
                                        1: electricity_raw_data
                                cluster:
                                    func: kmeans
                                    arguments:
                                        k: 8"""
        model = create_and_run_model(config_run)
        assert str(model.results.solver.termination_condition) == 'optimal'
