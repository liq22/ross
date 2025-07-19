"""Health State Simulator Module.

This module provides a unified interface for simulating different rotor health states:
- Normal (healthy)
- Crack
- Misalignment
- Rubbing

The module allows configuring rotor parameters, excitation parameters, fault parameters,
and measurement points to generate data for all four health states.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import ross as rs
from ross.units import Q_
from ross.probe import Probe
from ross.materials import Material


class TimeResponseResults:
    """Simple wrapper for time response results to maintain consistent interface."""
    
    def __init__(self, t: np.ndarray, yout: np.ndarray):
        """Initialize time response results.
        
        Parameters
        ----------
        t : np.ndarray
            Time array.
        yout : np.ndarray
            Response array.
        """
        self.t = t
        self.yout = yout


class HealthStateSimulator:
    """Unified interface for simulating different rotor health states.
    
    This class provides a consistent interface to simulate normal, crack,
    misalignment, and rubbing conditions in rotor systems using the same
    rotor configuration and excitation parameters.
    
    Parameters
    ----------
    rotor_config : dict
        Configuration parameters for the rotor including shaft elements,
        disks, bearings, and material properties.
    excitation_config : dict
        Configuration for excitation including unbalance magnitude, phase,
        speed, and time array.
    measurement_config : dict
        Configuration for measurement points (probe locations).
    
    Examples
    --------
    >>> # Create rotor configuration
    >>> rotor_config = {
    ...     'material': {'name': 'Steel', 'rho': 7850, 'E': 2.17e11, 'Poisson': 0.299},
    ...     'shaft': {'i_d': 0, 'o_d': 0.019, 'lengths': [0.025, 0.039, ...]},
    ...     'disks': [{'n': 12, 'm': 2.6375, 'Id': 0.003845, 'Ip': 0.007513}],
    ...     'bearings': [{'n': 4, 'kxx': 4.4e5, 'kyy': 4.61e5, 'cxx': 27.4, 'cyy': 2.505}]
    ... }
    >>> 
    >>> # Create excitation configuration
    >>> excitation_config = {
    ...     'nodes': [12, 24],
    ...     'unbalance_magnitude': [5e-4, 0],
    ...     'unbalance_phase': [-np.pi/2, 0],
    ...     'speed': Q_(1200, 'RPM'),
    ...     'time': np.arange(0, 5, 0.0001)
    ... }
    >>> 
    >>> # Create measurement configuration
    >>> measurement_config = {
    ...     'probes': [{'node': 14, 'angle': 0}, {'node': 22, 'angle': 0}]
    ... }
    >>> 
    >>> # Initialize simulator
    >>> simulator = HealthStateSimulator(rotor_config, excitation_config, measurement_config)
    >>> 
    >>> # Simulate all health states
    >>> results = simulator.simulate_all_states()
    """
    
    def __init__(
        self,
        rotor_config: Dict,
        excitation_config: Dict,
        measurement_config: Dict
    ):
        """Initialize the health state simulator."""
        self.rotor_config = rotor_config
        self.excitation_config = excitation_config
        self.measurement_config = measurement_config
        
        # Build the rotor from configuration
        self.rotor = self._build_rotor()
        
        # Create probes from configuration
        self.probes = self._create_probes()
        
    def _build_rotor(self) -> rs.Rotor:
        """Build rotor from configuration parameters."""
        # Create material
        mat_config = self.rotor_config['material']
        material = Material(
            name=mat_config['name'],
            rho=mat_config['rho'],
            E=mat_config['E'],
            Poisson=mat_config['Poisson']
        )
        
        # Create shaft elements
        shaft_config = self.rotor_config['shaft']
        shaft_elements = []
        
        for length in shaft_config['lengths']:
            shaft_elem = rs.ShaftElement(
                material=material,
                L=length,
                idl=shaft_config['i_d'],
                odl=shaft_config['o_d'],
                idr=shaft_config['i_d'],
                odr=shaft_config['o_d'],
                alpha=shaft_config.get('alpha', 8.0501),
                beta=shaft_config.get('beta', 1.0e-5),
                rotary_inertia=shaft_config.get('rotary_inertia', True),
                shear_effects=shaft_config.get('shear_effects', True),
            )
            shaft_elements.append(shaft_elem)
        
        # Create disks
        disks = []
        for disk_config in self.rotor_config.get('disks', []):
            disk = rs.DiskElement(
                n=disk_config['n'],
                m=disk_config['m'],
                Id=disk_config['Id'],
                Ip=disk_config['Ip']
            )
            disks.append(disk)
        
        # Create bearings
        bearings = []
        for bearing_config in self.rotor_config.get('bearings', []):
            bearing = rs.BearingElement(
                n=bearing_config['n'],
                kxx=bearing_config['kxx'],
                kyy=bearing_config['kyy'],
                cxx=bearing_config['cxx'],
                cyy=bearing_config['cyy'],
                kzz=bearing_config.get('kzz', 0),
                czz=bearing_config.get('czz', 0)
            )
            bearings.append(bearing)
        
        return rs.Rotor(shaft_elements, disks, bearings)
    
    def _create_probes(self) -> List[Probe]:
        """Create probe objects from configuration."""
        probes = []
        for probe_config in self.measurement_config['probes']:
            probe = Probe(
                node=probe_config['node'],
                angle=probe_config['angle']
            )
            probes.append(probe)
        return probes
    
    def simulate_normal(self, **kwargs) -> TimeResponseResults:
        """Simulate normal/healthy rotor response.
        
        Parameters
        ----------
        **kwargs : dict
            Additional parameters for simulation.
            
        Returns
        -------
        TimeResponseResults
            Time response results for normal condition.
        """
        # Create unbalance force for normal condition
        force = self._create_unbalance_force()
        
        # Convert speed to float value in rad/s
        speed = self.excitation_config['speed']
        if hasattr(speed, 'magnitude'):
            speed_rad_s = float(speed.to('rad/s').magnitude)
        else:
            speed_rad_s = speed * 2 * np.pi / 60  # Convert RPM to rad/s
        
        # Run time response analysis
        t, yout, xout = self.rotor.time_response(
            speed=speed_rad_s,
            F=force,
            t=self.excitation_config['time'],
            **kwargs
        )
        
        return TimeResponseResults(t, yout)
    
    def simulate_crack(
        self,
        crack_node: int,
        depth_ratio: float = 0.2,
        crack_model: str = "Mayes",
        **kwargs
    ) -> rs.TimeResponseResults:
        """Simulate rotor response with crack.
        
        Parameters
        ----------
        crack_node : int
            Node where crack is located.
        depth_ratio : float, optional
            Crack depth ratio (0.1 = 10%, 0.2 = 20%, etc.). Default is 0.2.
        crack_model : str, optional
            Crack model type ("Mayes" or "Gasch"). Default is "Mayes".
        **kwargs : dict
            Additional parameters for simulation.
            
        Returns
        -------
        ross.TimeResponseResults
            Time response results for crack condition.
        """
        results = self.rotor.run_crack(
            n=crack_node,
            depth_ratio=depth_ratio,
            node=self.excitation_config['nodes'],
            unbalance_magnitude=self.excitation_config['unbalance_magnitude'],
            unbalance_phase=self.excitation_config['unbalance_phase'],
            speed=self.excitation_config['speed'],
            t=self.excitation_config['time'],
            crack_model=crack_model,
            **kwargs
        )
        
        return results
    
    def simulate_misalignment(
        self,
        misalignment_node: int,
        coupling_type: str = "flex",
        mis_type: str = "combined",
        **mis_kwargs
    ) -> rs.TimeResponseResults:
        """Simulate rotor response with misalignment.
        
        Parameters
        ----------
        misalignment_node : int
            Node where misalignment occurs.
        coupling_type : str, optional
            Type of coupling ("flex" or "rigid"). Default is "flex".
        mis_type : str, optional
            Type of misalignment ("parallel", "angular", "combined"). Default is "combined".
        **mis_kwargs : dict
            Misalignment-specific parameters such as:
            - radial_stiffness : float
            - bending_stiffness : float (for angular/combined)
            - mis_distance_x : float
            - mis_distance_y : float
            - mis_angle : float (for angular/combined)
            
        Returns
        -------
        ross.TimeResponseResults
            Time response results for misalignment condition.
        """
        # Set default misalignment parameters if not provided
        default_params = {
            'n': misalignment_node,
            'radial_stiffness': 40e3,
            'bending_stiffness': 38e3,
            'mis_distance_x': 2e-4,
            'mis_distance_y': 2e-4,
            'mis_angle': 5 * np.pi / 180,
            'input_torque': 0,
            'load_torque': 0,
        }
        
        # Update with user-provided parameters
        default_params.update(mis_kwargs)
        
        results = self.rotor.run_misalignment(
            node=self.excitation_config['nodes'],
            unbalance_magnitude=self.excitation_config['unbalance_magnitude'],
            unbalance_phase=self.excitation_config['unbalance_phase'],
            speed=self.excitation_config['speed'],
            t=self.excitation_config['time'],
            coupling=coupling_type,
            mis_type=mis_type,
            **default_params
        )
        
        return results
    
    def simulate_rubbing(
        self,
        rubbing_node: int,
        distance: float = 7.95e-5,
        contact_stiffness: float = 1.1e6,
        contact_damping: float = 40,
        friction_coeff: float = 0.3,
        **kwargs
    ) -> rs.TimeResponseResults:
        """Simulate rotor response with rubbing.
        
        Parameters
        ----------
        rubbing_node : int
            Node where rubbing occurs.
        distance : float, optional
            Distance between housing and shaft surface. Default is 7.95e-5.
        contact_stiffness : float, optional
            Contact stiffness. Default is 1.1e6.
        contact_damping : float, optional
            Contact damping. Default is 40.
        friction_coeff : float, optional
            Friction coefficient. Default is 0.3.
        **kwargs : dict
            Additional parameters for simulation.
            
        Returns
        -------
        ross.TimeResponseResults
            Time response results for rubbing condition.
        """
        results = self.rotor.run_rubbing(
            n=rubbing_node,
            distance=distance,
            contact_stiffness=contact_stiffness,
            contact_damping=contact_damping,
            friction_coeff=friction_coeff,
            node=self.excitation_config['nodes'],
            unbalance_magnitude=self.excitation_config['unbalance_magnitude'],
            unbalance_phase=self.excitation_config['unbalance_phase'],
            speed=self.excitation_config['speed'],
            t=self.excitation_config['time'],
            **kwargs
        )
        
        return results
    
    def simulate_all_states(
        self,
        crack_node: Optional[int] = None,
        misalignment_node: Optional[int] = None,
        rubbing_node: Optional[int] = None,
        **fault_params
    ) -> Dict[str, rs.TimeResponseResults]:
        """Simulate all four health states.
        
        Parameters
        ----------
        crack_node : int, optional
            Node for crack simulation. If None, uses middle shaft element.
        misalignment_node : int, optional
            Node for misalignment simulation. If None, uses first node.
        rubbing_node : int, optional
            Node for rubbing simulation. If None, uses first disk node.
        **fault_params : dict
            Additional parameters for fault simulations.
            
        Returns
        -------
        dict
            Dictionary containing results for all health states:
            {'normal': results, 'crack': results, 'misalignment': results, 'rubbing': results}
        """
        # Determine default nodes if not specified
        if crack_node is None:
            crack_node = len(self.rotor.shaft_elements) // 2
        
        if misalignment_node is None:
            misalignment_node = 0
            
        if rubbing_node is None:
            if self.rotor.disk_elements:
                rubbing_node = self.rotor.disk_elements[0].n
            else:
                rubbing_node = len(self.rotor.shaft_elements) // 2
        
        results = {}
        
        # Simulate normal condition
        print("Simulating normal condition...")
        results['normal'] = self.simulate_normal()
        
        # Simulate crack condition
        print(f"Simulating crack condition at node {crack_node}...")
        crack_params = {k: v for k, v in fault_params.items() if k.startswith('crack_')}
        crack_params = {k.replace('crack_', ''): v for k, v in crack_params.items()}
        results['crack'] = self.simulate_crack(crack_node, **crack_params)
        
        # Simulate misalignment condition
        print(f"Simulating misalignment condition at node {misalignment_node}...")
        mis_params = {k: v for k, v in fault_params.items() if k.startswith('mis_')}
        mis_params = {k.replace('mis_', ''): v for k, v in mis_params.items()}
        results['misalignment'] = self.simulate_misalignment(misalignment_node, **mis_params)
        
        # Simulate rubbing condition
        print(f"Simulating rubbing condition at node {rubbing_node}...")
        rub_params = {k: v for k, v in fault_params.items() if k.startswith('rub_')}
        rub_params = {k.replace('rub_', ''): v for k, v in rub_params.items()}
        results['rubbing'] = self.simulate_rubbing(rubbing_node, **rub_params)
        
        return results
    
    def _create_unbalance_force(self) -> np.ndarray:
        """Create unbalance force array for time response analysis."""
        # Get rotor DOF information
        ndof = len(self.rotor.nodes) * 6
        nt = len(self.excitation_config['time'])
        
        # Initialize force array
        force = np.zeros((nt, ndof))
        
        # Convert speed to rad/s if needed
        speed = self.excitation_config['speed']
        if hasattr(speed, 'magnitude'):
            speed_rad_s = float(speed.to('rad/s').magnitude)
        else:
            speed_rad_s = speed * 2 * np.pi / 60  # Convert RPM to rad/s
        
        # Apply unbalance forces
        nodes = self.excitation_config['nodes']
        magnitudes = self.excitation_config['unbalance_magnitude']
        phases = self.excitation_config['unbalance_phase']
        time = self.excitation_config['time']
        
        for i, (node, mag, phase) in enumerate(zip(nodes, magnitudes, phases)):
            if mag > 0:  # Only apply force if magnitude is non-zero
                # Calculate force components
                fx = mag * speed_rad_s**2 * np.cos(speed_rad_s * time + phase)
                fy = mag * speed_rad_s**2 * np.sin(speed_rad_s * time + phase)
                
                # Apply to corresponding DOFs
                dof_x = node * 6  # X displacement DOF
                dof_y = node * 6 + 1  # Y displacement DOF
                
                if dof_x < ndof and dof_y < ndof:
                    force[:, dof_x] += fx
                    force[:, dof_y] += fy
        
        return force
    
    def plot_results(
        self,
        results: Dict[str, rs.TimeResponseResults],
        probe_index: int = 0,
        show: bool = True
    ):
        """Plot comparison of all health states.
        
        Parameters
        ----------
        results : dict
            Dictionary of results from simulate_all_states().
        probe_index : int, optional
            Index of probe to plot. Default is 0.
        show : bool, optional
            Whether to show the plot. Default is True.
            
        Returns
        -------
        plotly.graph_objects.Figure
            The plot figure.
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        if probe_index >= len(self.probes):
            raise ValueError(f"Probe index {probe_index} out of range. Available probes: {len(self.probes)}")
        
        probe = self.probes[probe_index]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Normal", "Crack", "Misalignment", "Rubbing"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        conditions = ['normal', 'crack', 'misalignment', 'rubbing']
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for condition, (row, col) in zip(conditions, positions):
            if condition in results:
                result = results[condition]
                # Get time series data from the probe
                time = result.t
                response = result.yout[probe.node * 6]  # X displacement
                
                fig.add_trace(
                    go.Scatter(
                        x=time,
                        y=response,
                        mode='lines',
                        name=f'{condition.capitalize()} - Node {probe.node}',
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title=f"Rotor Health States Comparison - Probe at Node {probe.node}",
            height=800
        )
        
        if show:
            fig.show()
        
        return fig


def create_example_rotor_config() -> Dict:
    """Create an example rotor configuration.
    
    Returns
    -------
    dict
        Example rotor configuration dictionary.
    """
    # Length distribution from the problem statement
    L_positions = np.array([
        0, 25, 64, 104, 124, 143, 175, 207, 239, 271,
        303, 335, 345, 355, 380, 408, 436, 466, 496, 526,
        556, 586, 614, 647, 657, 667, 702, 737, 772, 807,
        842, 862, 881, 914
    ]) / 1000
    
    lengths = [L_positions[i] - L_positions[i-1] for i in range(1, len(L_positions))]
    
    rotor_config = {
        'material': {
            'name': 'Steel',
            'rho': 7850,
            'E': 2.17e11,
            'Poisson': 0.2992610837438423
        },
        'shaft': {
            'i_d': 0,
            'o_d': 0.019,
            'lengths': lengths,
            'alpha': 8.0501,
            'beta': 1.0e-5,
            'rotary_inertia': True,
            'shear_effects': True
        },
        'disks': [
            {'n': 12, 'm': 2.6375, 'Id': 0.003844540885417, 'Ip': 0.007513248437500},
            {'n': 24, 'm': 2.6375, 'Id': 0.003844540885417, 'Ip': 0.007513248437500}
        ],
        'bearings': [
            {
                'n': 4,
                'kxx': 4.40e5,
                'kyy': 4.6114e5,
                'cxx': 27.4,
                'cyy': 2.505,
                'kzz': 0,
                'czz': 0
            },
            {
                'n': 31,
                'kxx': 9.50e5,
                'kyy': 1.09e8,
                'cxx': 50.4,
                'cyy': 100.4553,
                'kzz': 0,
                'czz': 0
            }
        ]
    }
    
    return rotor_config


def create_example_excitation_config() -> Dict:
    """Create an example excitation configuration.
    
    Returns
    -------
    dict
        Example excitation configuration dictionary.
    """
    excitation_config = {
        'nodes': [12, 24],
        'unbalance_magnitude': [5e-4, 0],
        'unbalance_phase': [-np.pi/2, 0],
        'speed': Q_(1200, 'RPM'),
        'time': np.arange(0, 5, 0.0001)
    }
    
    return excitation_config


def create_example_measurement_config() -> Dict:
    """Create an example measurement configuration.
    
    Returns
    -------
    dict
        Example measurement configuration dictionary.
    """
    measurement_config = {
        'probes': [
            {'node': 14, 'angle': 0},
            {'node': 22, 'angle': 0}
        ]
    }
    
    return measurement_config