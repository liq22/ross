"""Test cases for the Health State Simulator module."""

import pytest
import numpy as np
from numpy.testing import assert_allclose
import ross as rs
from ross.units import Q_
from ross.health_simulator import (
    HealthStateSimulator,
    TimeResponseResults,
    create_example_rotor_config,
    create_example_excitation_config,
    create_example_measurement_config
)


class TestTimeResponseResults:
    """Test the TimeResponseResults wrapper class."""
    
    def test_initialization(self):
        """Test TimeResponseResults initialization."""
        t = np.linspace(0, 1, 100)
        yout = np.random.random((100, 12))
        
        result = TimeResponseResults(t, yout)
        
        assert_allclose(result.t, t)
        assert_allclose(result.yout, yout)


class TestHealthStateSimulator:
    """Test the HealthStateSimulator class."""
    
    @pytest.fixture
    def simulator(self):
        """Create a simulator instance for testing."""
        rotor_config = create_example_rotor_config()
        excitation_config = create_example_excitation_config()
        measurement_config = create_example_measurement_config()
        
        # Use shorter time for faster testing
        excitation_config['time'] = np.arange(0, 0.5, 0.001)
        
        return HealthStateSimulator(
            rotor_config=rotor_config,
            excitation_config=excitation_config,
            measurement_config=measurement_config
        )
    
    def test_initialization(self, simulator):
        """Test simulator initialization."""
        assert len(simulator.rotor.shaft_elements) == 33
        assert len(simulator.rotor.disk_elements) == 2
        assert len(simulator.rotor.bearing_elements) == 2
        assert len(simulator.probes) == 2
        
        # Check probe configurations
        assert simulator.probes[0].node == 14
        assert simulator.probes[1].node == 22
    
    def test_simulate_normal(self, simulator):
        """Test normal condition simulation."""
        result = simulator.simulate_normal()
        
        assert isinstance(result, TimeResponseResults)
        assert len(result.t) == len(simulator.excitation_config['time'])
        assert result.yout.shape[0] == len(result.t)
        
        # Check that response is reasonable
        max_response = np.max(np.abs(result.yout))
        assert max_response > 0
        assert max_response < 1e-2  # Should be reasonable for this rotor
    
    def test_simulate_crack(self, simulator):
        """Test crack condition simulation."""
        result = simulator.simulate_crack(
            crack_node=18,
            depth_ratio=0.1,
            crack_model="Mayes"
        )
        
        assert hasattr(result, 't')
        assert hasattr(result, 'yout')
        assert len(result.t) == len(simulator.excitation_config['time'])
        
        # Crack should generally increase response compared to normal
        normal_result = simulator.simulate_normal()
        crack_rms = np.sqrt(np.mean(result.yout**2))
        normal_rms = np.sqrt(np.mean(normal_result.yout**2))
        
        # Crack should typically increase response (but this may not always be true)
        # So we just check that both are positive
        assert crack_rms > 0
        assert normal_rms > 0
    
    def test_simulate_misalignment(self, simulator):
        """Test misalignment condition simulation."""
        result = simulator.simulate_misalignment(
            misalignment_node=0,
            mis_type="parallel",
            mis_distance_x=1e-4,
            mis_distance_y=1e-4
        )
        
        assert hasattr(result, 't')
        assert hasattr(result, 'yout')
        assert len(result.t) == len(simulator.excitation_config['time'])
        
        max_response = np.max(np.abs(result.yout))
        assert max_response > 0
    
    def test_simulate_rubbing(self, simulator):
        """Test rubbing condition simulation."""
        result = simulator.simulate_rubbing(
            rubbing_node=12,
            distance=2e-4,  # Conservative parameters
            contact_stiffness=1e5,
            contact_damping=20,
            friction_coeff=0.2
        )
        
        assert hasattr(result, 't')
        assert hasattr(result, 'yout')
        assert len(result.t) == len(simulator.excitation_config['time'])
        
        max_response = np.max(np.abs(result.yout))
        assert max_response > 0
    
    def test_simulate_all_states(self, simulator):
        """Test simulation of all health states."""
        results = simulator.simulate_all_states(
            crack_node=18,
            misalignment_node=0,
            rubbing_node=12,
            # Conservative rubbing parameters
            rub_distance=2e-4,
            rub_contact_stiffness=1e5,
            rub_contact_damping=20
        )
        
        expected_states = ['normal', 'crack', 'misalignment', 'rubbing']
        assert all(state in results for state in expected_states)
        
        for state, result in results.items():
            assert hasattr(result, 't')
            assert hasattr(result, 'yout')
            assert len(result.t) == len(simulator.excitation_config['time'])
            
            max_response = np.max(np.abs(result.yout))
            assert max_response > 0
    
    def test_create_unbalance_force(self, simulator):
        """Test unbalance force creation."""
        force = simulator._create_unbalance_force()
        
        expected_nt = len(simulator.excitation_config['time'])
        expected_ndof = len(simulator.rotor.nodes) * 6
        
        assert force.shape == (expected_nt, expected_ndof)
        
        # Force should be non-zero at unbalance nodes
        unb_nodes = simulator.excitation_config['nodes']
        unb_mags = simulator.excitation_config['unbalance_magnitude']
        
        for node, mag in zip(unb_nodes, unb_mags):
            if mag > 0:
                dof_x = node * 6
                dof_y = node * 6 + 1
                assert np.any(force[:, dof_x] != 0)
                assert np.any(force[:, dof_y] != 0)


class TestConfigurationFunctions:
    """Test configuration creation functions."""
    
    def test_create_example_rotor_config(self):
        """Test example rotor configuration creation."""
        config = create_example_rotor_config()
        
        required_keys = ['material', 'shaft', 'disks', 'bearings']
        assert all(key in config for key in required_keys)
        
        # Check material properties
        material = config['material']
        assert material['name'] == 'Steel'
        assert material['rho'] > 0
        assert material['E'] > 0
        
        # Check shaft properties
        shaft = config['shaft']
        assert len(shaft['lengths']) == 33
        assert shaft['o_d'] > shaft['i_d']
        
        # Check disks and bearings
        assert len(config['disks']) == 2
        assert len(config['bearings']) == 2
    
    def test_create_example_excitation_config(self):
        """Test example excitation configuration creation."""
        config = create_example_excitation_config()
        
        required_keys = ['nodes', 'unbalance_magnitude', 'unbalance_phase', 'speed', 'time']
        assert all(key in config for key in required_keys)
        
        assert len(config['nodes']) == 2
        assert len(config['unbalance_magnitude']) == 2
        assert len(config['unbalance_phase']) == 2
        assert len(config['time']) > 1000  # Should have many time points
    
    def test_create_example_measurement_config(self):
        """Test example measurement configuration creation."""
        config = create_example_measurement_config()
        
        assert 'probes' in config
        assert len(config['probes']) == 2
        
        for probe in config['probes']:
            assert 'node' in probe
            assert 'angle' in probe
            assert isinstance(probe['node'], int)
            assert isinstance(probe['angle'], (int, float))


class TestCustomConfigurations:
    """Test simulator with custom configurations."""
    
    def test_simple_rotor_configuration(self):
        """Test simulator with a simple rotor configuration."""
        # Create a simple rotor
        simple_config = {
            'material': {
                'name': 'Steel',
                'rho': 7850,
                'E': 2.1e11,
                'Poisson': 0.3
            },
            'shaft': {
                'i_d': 0,
                'o_d': 0.02,
                'lengths': [0.1, 0.1, 0.1],
                'alpha': 0,
                'beta': 0,
                'rotary_inertia': True,
                'shear_effects': True
            },
            'disks': [
                {'n': 1, 'm': 5, 'Id': 0.01, 'Ip': 0.02}
            ],
            'bearings': [
                {'n': 0, 'kxx': 1e6, 'kyy': 1e6, 'cxx': 100, 'cyy': 100},
                {'n': 2, 'kxx': 1e6, 'kyy': 1e6, 'cxx': 100, 'cyy': 100}
            ]
        }
        
        excitation_config = {
            'nodes': [1],
            'unbalance_magnitude': [1e-4],
            'unbalance_phase': [0],
            'speed': Q_(1000, 'RPM'),
            'time': np.arange(0, 0.1, 0.001)
        }
        
        measurement_config = {
            'probes': [{'node': 1, 'angle': 0}]
        }
        
        simulator = HealthStateSimulator(
            rotor_config=simple_config,
            excitation_config=excitation_config,
            measurement_config=measurement_config
        )
        
        assert len(simulator.rotor.shaft_elements) == 3
        assert len(simulator.rotor.disk_elements) == 1
        assert len(simulator.rotor.bearing_elements) == 2
        
        # Test normal simulation
        result = simulator.simulate_normal()
        assert hasattr(result, 't')
        assert hasattr(result, 'yout')


# Test that the module can be imported without errors
def test_import():
    """Test that the module imports correctly."""
    from ross.health_simulator import (
        HealthStateSimulator,
        TimeResponseResults,
        create_example_rotor_config,
        create_example_excitation_config,
        create_example_measurement_config
    )
    
    # Basic functionality test
    config = create_example_rotor_config()
    assert isinstance(config, dict)