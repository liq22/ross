"""
Recreation of the original problem statement with the new Health State Simulator.

This script recreates the rotor configuration from the problem statement and 
demonstrates how to simulate all four health states using the unified interface.
"""

import numpy as np
import ross as rs
from ross.faults import *
from ross.units import Q_
from ross.probe import Probe
from ross.health_simulator import HealthStateSimulator

# Recreate the original rotor configuration from the problem statement
def create_original_rotor_config():
    """Create the exact rotor configuration from the problem statement."""
    steel2 = rs.Material(name="Steel", rho=7850, E=2.17e11, Poisson=0.2992610837438423)
    
    # Original rotor geometry
    i_d = 0
    o_d = 0.019
    n = 33

    # Original length distribution
    L = np.array(
        [0  ,  25,  64, 104, 124, 143, 175, 207, 239, 271,
         303, 335, 345, 355, 380, 408, 436, 466, 496, 526,
         556, 586, 614, 647, 657, 667, 702, 737, 772, 807,
         842, 862, 881, 914]
    ) / 1000

    L = [L[i] - L[i - 1] for i in range(1, len(L))]

    # Original disk properties
    Id = 0.003844540885417
    Ip = 0.007513248437500

    # Original bearing properties
    kxx1 = 4.40e5
    kyy1 = 4.6114e5
    kzz = 0
    cxx1 = 27.4
    cyy1 = 2.505
    czz = 0
    kxx2 = 9.50e5
    kyy2 = 1.09e8
    cxx2 = 50.4
    cyy2 = 100.4553

    rotor_config = {
        'material': {
            'name': 'Steel',
            'rho': 7850,
            'E': 2.17e11,
            'Poisson': 0.2992610837438423
        },
        'shaft': {
            'i_d': i_d,
            'o_d': o_d,
            'lengths': L,
            'alpha': 8.0501,
            'beta': 1.0e-5,
            'rotary_inertia': True,
            'shear_effects': True
        },
        'disks': [
            {'n': 12, 'm': 2.6375, 'Id': Id, 'Ip': Ip},
            {'n': 24, 'm': 2.6375, 'Id': Id, 'Ip': Ip}
        ],
        'bearings': [
            {'n': 4, 'kxx': kxx1, 'kyy': kyy1, 'cxx': cxx1, 'cyy': cyy1, 'kzz': kzz, 'czz': czz},
            {'n': 31, 'kxx': kxx2, 'kyy': kyy2, 'cxx': cxx2, 'cyy': cyy2, 'kzz': kzz, 'czz': czz}
        ]
    }
    
    return rotor_config


def main():
    """Main demonstration following the original problem statement."""
    print("Recreation of Original Problem Statement")
    print("=" * 45)
    
    # Create the original rotor configuration
    rotor_config = create_original_rotor_config()
    
    # Original excitation parameters
    massunbt = np.array([5e-4, 0])
    phaseunbt = np.array([-np.pi / 2, 0])
    
    excitation_config = {
        'nodes': [12, 24],
        'unbalance_magnitude': massunbt,
        'unbalance_phase': phaseunbt,
        'speed': Q_(1200, "RPM"),
        'time': np.arange(0, 5, 0.0001)
    }
    
    # Original measurement configuration
    probe1 = Probe(14, 0)
    probe2 = Probe(22, 0)
    
    measurement_config = {
        'probes': [
            {'node': 14, 'angle': 0},
            {'node': 22, 'angle': 0}
        ]
    }
    
    print("✓ Original rotor configuration recreated")
    print(f"  - Shaft elements: {len(rotor_config['shaft']['lengths'])}")
    print(f"  - Disks: {len(rotor_config['disks'])}")
    print(f"  - Bearings: {len(rotor_config['bearings'])}")
    print(f"  - Speed: {excitation_config['speed']}")
    print(f"  - Simulation time: {excitation_config['time'][-1]}s")
    
    # Initialize the health state simulator
    simulator = HealthStateSimulator(
        rotor_config=rotor_config,
        excitation_config=excitation_config,
        measurement_config=measurement_config
    )
    
    print("\n✓ Health State Simulator initialized")
    
    # Simulate all four health states as requested
    print("\nSimulating all four health states...")
    print("-" * 40)
    
    # Parameters matching the original problem statement
    results = simulator.simulate_all_states(
        crack_node=18,  # Middle of shaft
        misalignment_node=0,  # At coupling
        rubbing_node=12,  # At first disk (matching original)
        # Rubbing parameters from original problem
        rub_distance=7.95e-5,
        rub_contact_stiffness=1.1e6,
        rub_contact_damping=40,
        rub_friction_coeff=0.3
    )
    
    print("\n✓ All health states simulated successfully!")
    
    # Display results summary
    print("\nResults Summary:")
    print("=" * 30)
    
    for state, result in results.items():
        # Calculate some statistics
        time_span = result.t[-1] - result.t[0]
        max_response = np.max(np.abs(result.yout))
        rms_response = np.sqrt(np.mean(result.yout**2))
        
        print(f"{state.upper():>12}:")
        print(f"  Time span: {time_span:.3f}s")
        print(f"  Max response: {max_response:.2e}m")
        print(f"  RMS response: {rms_response:.2e}m")
        print()
    
    # Extract data at measurement points (probes)
    print("Data at Measurement Points:")
    print("=" * 35)
    
    for i, probe_config in enumerate(measurement_config['probes']):
        node = probe_config['node']
        print(f"\nProbe {i+1} at Node {node}:")
        
        for state, result in results.items():
            # Get displacement in X direction
            x_displacement = result.yout[node * 6]  # X DOF
            rms_value = np.sqrt(np.mean(x_displacement**2))
            peak_value = np.max(np.abs(x_displacement))
            
            print(f"  {state:>12}: RMS={rms_value:.2e}m, Peak={peak_value:.2e}m")
    
    # Demonstrate individual state simulation with custom parameters
    print(f"\nIndividual Simulations with Custom Parameters:")
    print("=" * 50)
    
    # Normal condition
    normal_custom = simulator.simulate_normal()
    print("✓ Normal state simulated")
    
    # Crack with different severity
    crack_custom = simulator.simulate_crack(
        crack_node=18,
        depth_ratio=0.3,  # 30% crack depth
        crack_model="Mayes"
    )
    print("✓ Crack state simulated (30% depth)")
    
    # Misalignment with different parameters  
    misalign_custom = simulator.simulate_misalignment(
        misalignment_node=0,
        mis_type="combined",
        mis_distance_x=3e-4,
        mis_distance_y=3e-4,
        mis_angle=8 * np.pi / 180,  # 8 degrees
        radial_stiffness=45e3,
        bending_stiffness=40e3
    )
    print("✓ Misalignment state simulated (combined, 8°)")
    
    # Rubbing with original parameters from problem statement
    rubbing_custom = simulator.simulate_rubbing(
        rubbing_node=12,
        distance=7.95e-5,
        contact_stiffness=1.1e6,
        contact_damping=40,
        friction_coeff=0.3
    )
    print("✓ Rubbing state simulated (original parameters)")
    
    print("\n" + "=" * 60)
    print("✓ SIMULATION COMPLETE - All four health states generated!")
    print("✓ Interface provides unified access to normal, crack,")
    print("  misalignment, and rubbing simulations with configurable")
    print("  rotor parameters, excitation, and measurement points.")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    results = main()