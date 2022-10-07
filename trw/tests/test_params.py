from unittest import TestCase
import trw
import numpy as np
import math


class TestParams(TestCase):
    def test_power_initial_value(self):
        """
        Make sure we apply the power of the initial value
        """

        p = trw.hparams.ContinuousPower('hp1', 2, 0, 3)
        value = p.get_value()
        expected = 2
        assert abs(value - expected) < 1e-5
        
    def test_power_range(self):
        """
        Test the range of generated values is as expected
        """

        values_to_generate = 10000
        values = []
        min_power = 2
        max_power = 5
        for n in range(values_to_generate):
            p = trw.hparams.ContinuousPower('hp1', 101, min_power, max_power)
            p.randomize()
            value = p.get_value()
            assert value >= 10 ** min_power
            assert value <= 10 ** max_power
            values.append(value)
            
        # check the min/max domain
        exponent_domain = np.log(np.asarray(values)) / math.log(10)
        assert np.min(exponent_domain) >= min_power
        assert np.max(exponent_domain) <= max_power
        
        # make sure we have uniform distribution of the exponent
        histogram_bins = 10
        expected_bin_count = values_to_generate / histogram_bins
        counts, intervals = np.histogram(exponent_domain, range=(min_power, max_power), bins=10)
        error_rate = np.max(np.abs(counts - expected_bin_count)) / expected_bin_count
        assert error_rate < 0.1

    def test_hparams_discrete_mapping(self):
        mapping = {
            'key1': 'value1',
            'key2': 'value2',
            'key3': 'value3',
            'key4': 'value4',
        }
        h = trw.hparams.DiscreteMapping('hp1', default_value='key1', mapping=mapping)
        assert h.get_value() == 'value1'

        h.randomize()
        assert h.get_value() in mapping.values()

    def test_discrete_value(self):
        values = ['value0', 'value1', 'value2']
        h = trw.hparams.DiscreteValue('hp1', default_value='value0', values=values)
        assert h.get_value() == 'value0'

        h.randomize()
        assert h.get_value() in values

    def test_discrete_integer(self):
        h = trw.hparams.DiscreteInteger('hp1', default_value=42, min_range=-10, max_range=42)
        assert h.get_value() == 42

        for i in range(50):
            h.randomize()
            assert -10 <= h.get_value() <= 42

    def test_discrete_boolean(self):
        h = trw.hparams.DiscreteBoolean('hp1', default_value=True)
        assert h.get_value()
        h.randomize()
        assert isinstance(h.get_value(), bool)

    def test_continuous_uniform(self):
        h = trw.hparams.ContinuousUniform('hp1', default_value=42, min_range=-10, max_range=42)
        assert h.get_value() == 42

        for i in range(50):
            h.randomize()
            assert -10 <= h.get_value() <= 42

    def test_hparams_optimize_all(self):
        hparams = trw.hparams.HyperParameters()
        hp1 = trw.hparams.DiscreteBoolean('hp1', default_value=True)
        hp2 = trw.hparams.DiscreteInteger('hp2', default_value=0, min_range=0, max_range=10)
        v1 = hparams.create(hp1)
        v2 = hparams.create(hp2)
        assert v1
        assert v2 == 0

        v1_all = set()
        v2_all = set()
        for i in range(1000):
            hparams.randomize()
            v1_all.add(hp1.get_value())
            v2_all.add(hp2.get_value())
        assert len(v1_all) == 2
        assert len(v2_all) == 11

        assert hparams['hp1'] == hp1

    def test_hparams_repo_recreate_with_same_value(self):
        """
        Repo MUST recreate the same hyper parameter with the same value!
        """
        trw.hparams.HyperParameterRepository.reset()  # new session
        v1 = trw.hparams.create_discrete_value('hp1', 2, [0, 1, 2])
        assert v1 == 2

        v1_b = trw.hparams.create_discrete_value('hp1', 0, [0, 1, 2])
        assert v1 == v1_b

    def test_hparams_repo_initial_random(self):
        """
        Repo MUST recreate the same hyper parameter with the same value!
        """
        np.random.seed(0)

        hparams = trw.hparams.HyperParameters(randomize_at_creation=True)
        hp = trw.hparams.DiscreteInteger('hp', default_value=0, min_range=0, max_range=10000000)
        v = hparams.create(hp)
        assert v != 0

    def test_hparams_partial_randomization(self):
        """
        Repo MUST recreate the same hyper parameter with the same value!
        """
        np.random.seed(0)

        hparams = trw.hparams.HyperParameters(randomize_at_creation=True, hparams_to_randomize=['hp2'])
        hp1 = trw.hparams.DiscreteInteger('hp1', default_value=0, min_range=0, max_range=10000000)
        hp2 = trw.hparams.DiscreteInteger('hp2', default_value=0, min_range=0, max_range=10000000)
        hparams.create(hp1)
        hparams.create(hp2)

        v1_all = set()
        v2_all = set()
        for i in range(1000):
            hparams.randomize()
            v1_all.add(hp1.get_value())
            v2_all.add(hp2.get_value())
        assert len(v1_all) == 1
        assert len(v2_all) > 20

    def test_regex(self):
        """
        Handle regular expression (e.g., to handle hyper-parameters hierarchically)
        """
        hparams = trw.hparams.HyperParameters(randomize_at_creation=True, hparams_to_randomize=['hp.test1.*'])
        assert hparams.hparam_to_be_randomized('hp.test1.name1')
        assert not hparams.hparam_to_be_randomized('hp.test2.name1')

    def test_non_optimized_params_created_with_default_value(self):
        hparams = trw.hparams.HyperParameters(randomize_at_creation=True, hparams_to_randomize=['hp2'])
        hp1 = trw.hparams.DiscreteInteger('hp1', default_value=42, min_range=0, max_range=100)
        assert hp1.get_value() == 42

        hparams.randomize()
        assert hp1.get_value() == 42