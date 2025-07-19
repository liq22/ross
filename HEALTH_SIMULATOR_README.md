# Health State Simulator

A unified interface for simulating different rotor health states in the ROSS (Rotordynamics Open Source Software) library.

## Overview

The Health State Simulator provides a consistent interface to simulate four different rotor health conditions:

1. **Normal** (healthy condition)
2. **Crack** (shaft crack)
3. **Misalignment** (coupling misalignment)
4. **Rubbing** (rotor-stator contact)

## Features

- **Unified Interface**: Single class to configure and simulate all health states
- **Configurable Parameters**: Flexible rotor geometry, material properties, excitation, and measurement setup
- **Consistent Results**: All simulations return results with the same interface for easy comparison
- **Multiple Fault Types**: Support for different crack models, misalignment types, and rubbing parameters
- **Measurement Points**: Configure multiple probe locations for data collection

## Quick Start

### Basic Usage

```python
import numpy as np
from ross.units import Q_
from ross.health_simulator import (
    HealthStateSimulator,
    create_example_rotor_config,
    create_example_excitation_config,
    create_example_measurement_config
)

# Create configurations
rotor_config = create_example_rotor_config()
excitation_config = create_example_excitation_config()
measurement_config = create_example_measurement_config()

# Initialize simulator
simulator = HealthStateSimulator(
    rotor_config=rotor_config,
    excitation_config=excitation_config,
    measurement_config=measurement_config
)

# Simulate all health states
results = simulator.simulate_all_states(
    crack_node=18,
    misalignment_node=0,
    rubbing_node=12
)

# Access results
for state, result in results.items():
    print(f"{state}: Max response = {np.max(np.abs(result.yout)):.2e} m")
```

### Individual Simulations

```python
# Normal condition
normal_result = simulator.simulate_normal()

# Crack with custom parameters
crack_result = simulator.simulate_crack(
    crack_node=18,
    depth_ratio=0.2,  # 20% crack depth
    crack_model="Mayes"
)

# Misalignment with custom parameters
misalign_result = simulator.simulate_misalignment(
    misalignment_node=0,
    mis_type="combined",
    mis_distance_x=2e-4,
    mis_distance_y=2e-4,
    mis_angle=5 * np.pi / 180  # 5 degrees
)

# Rubbing with custom parameters
rubbing_result = simulator.simulate_rubbing(
    rubbing_node=12,
    distance=7.95e-5,
    contact_stiffness=1.1e6,
    contact_damping=40,
    friction_coeff=0.3
)
```

## Configuration

### Rotor Configuration

```python
rotor_config = {
    'material': {
        'name': 'Steel',
        'rho': 7850,           # Density (kg/m³)
        'E': 2.17e11,          # Young's modulus (Pa)
        'Poisson': 0.299       # Poisson's ratio
    },
    'shaft': {
        'i_d': 0,              # Inner diameter (m)
        'o_d': 0.019,          # Outer diameter (m)
        'lengths': [0.025, 0.039, ...],  # Element lengths (m)
        'alpha': 8.0501,       # Proportional damping
        'beta': 1.0e-5,        # Proportional damping
        'rotary_inertia': True,
        'shear_effects': True
    },
    'disks': [
        {
            'n': 12,           # Node number
            'm': 2.6375,       # Mass (kg)
            'Id': 0.003845,    # Diametral moment of inertia (kg⋅m²)
            'Ip': 0.007513     # Polar moment of inertia (kg⋅m²)
        }
    ],
    'bearings': [
        {
            'n': 4,            # Node number
            'kxx': 4.4e5,      # Stiffness (N/m)
            'kyy': 4.61e5,
            'cxx': 27.4,       # Damping (N⋅s/m)
            'cyy': 2.505
        }
    ]
}
```

### Excitation Configuration

```python
excitation_config = {
    'nodes': [12, 24],                    # Unbalance nodes
    'unbalance_magnitude': [5e-4, 0],     # Unbalance masses (kg⋅m)
    'unbalance_phase': [-np.pi/2, 0],     # Phase angles (rad)
    'speed': Q_(1200, 'RPM'),             # Rotor speed
    'time': np.arange(0, 5, 0.0001)       # Time array (s)
}
```

### Measurement Configuration

```python
measurement_config = {
    'probes': [
        {'node': 14, 'angle': 0},  # Probe at node 14, 0° angle
        {'node': 22, 'angle': 0}   # Probe at node 22, 0° angle
    ]
}
```

## Methods

### `simulate_normal(**kwargs)`
Simulates healthy rotor response using unbalance forces.

### `simulate_crack(crack_node, depth_ratio=0.2, crack_model="Mayes", **kwargs)`
Simulates rotor with crack fault.

**Parameters:**
- `crack_node`: Node where crack is located
- `depth_ratio`: Crack depth as fraction of diameter (0.1 = 10%)
- `crack_model`: "Mayes" or "Gasch"

### `simulate_misalignment(misalignment_node, coupling_type="flex", mis_type="combined", **kwargs)`
Simulates rotor with misalignment fault.

**Parameters:**
- `misalignment_node`: Node where misalignment occurs
- `coupling_type`: "flex" or "rigid"
- `mis_type`: "parallel", "angular", or "combined"
- `mis_distance_x`, `mis_distance_y`: Parallel misalignment distances
- `mis_angle`: Angular misalignment angle

### `simulate_rubbing(rubbing_node, distance=7.95e-5, contact_stiffness=1.1e6, **kwargs)`
Simulates rotor with rubbing fault.

**Parameters:**
- `rubbing_node`: Node where rubbing occurs
- `distance`: Clearance between rotor and stator
- `contact_stiffness`: Contact stiffness when rubbing
- `contact_damping`: Contact damping
- `friction_coeff`: Friction coefficient

### `simulate_all_states(**fault_params)`
Simulates all four health states with specified parameters.

## Results

All simulation methods return results with the following attributes:
- `t`: Time array
- `yout`: Response array (displacement, velocity, acceleration for each DOF)

## Examples

See the following example files:
- `examples/health_simulator_example.py`: Comprehensive usage example
- `original_problem_demo.py`: Recreation of the original problem statement
- `test_health_simulator.py`: Simple functionality test

## Requirements

- ROSS library
- NumPy
- SciPy (included with ROSS)
- Pint (for units, included with ROSS)

## Notes

- The simulator automatically handles unit conversions for speed and other parameters
- Rubbing simulations may have convergence issues with very stiff contact parameters
- For large systems or long time simulations, consider reducing the time step or simulation duration
- All fault parameters can be customized through keyword arguments to fine-tune the simulation