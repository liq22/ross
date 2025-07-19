"""Example usage of Health State Simulator.

This script demonstrates how to use the HealthStateSimulator to generate
data for all four rotor health states: normal, crack, misalignment, and rubbing.
"""

import numpy as np
import ross as rs
from ross.units import Q_
from ross.health_simulator import (
    HealthStateSimulator,
    create_example_rotor_config,
    create_example_excitation_config,
    create_example_measurement_config
)


def main():
    """Main example demonstrating health state simulation."""
    print("Health State Simulator Example")
    print("==============================")
    
    # Create example configurations
    print("1. Creating rotor configuration...")
    rotor_config = create_example_rotor_config()
    
    print("2. Creating excitation configuration...")
    excitation_config = create_example_excitation_config()
    
    print("3. Creating measurement configuration...")
    measurement_config = create_example_measurement_config()
    
    # Initialize simulator
    print("4. Initializing health state simulator...")
    simulator = HealthStateSimulator(
        rotor_config=rotor_config,
        excitation_config=excitation_config,
        measurement_config=measurement_config
    )
    
    print(f"   Rotor has {len(simulator.rotor.shaft_elements)} shaft elements")
    print(f"   Rotor has {len(simulator.rotor.disk_elements)} disks")
    print(f"   Rotor has {len(simulator.rotor.bearing_elements)} bearings")
    print(f"   Number of measurement probes: {len(simulator.probes)}")
    
    # Simulate all health states
    print("\n5. Simulating all health states...")
    results = simulator.simulate_all_states(
        crack_node=18,  # Middle of the shaft
        misalignment_node=0,  # At coupling
        rubbing_node=12,  # At first disk
        # Additional fault parameters
        crack_depth_ratio=0.2,
        crack_model="Mayes",
        mis_distance_x=2e-4,
        mis_distance_y=2e-4,
        mis_angle=5 * np.pi / 180,
        rub_distance=7.95e-5,
        rub_contact_stiffness=1.1e6,
        rub_contact_damping=40,
        rub_friction_coeff=0.3
    )
    
    # Display results summary
    print("\n6. Results Summary:")
    print("   " + "="*50)
    
    for condition, result in results.items():
        time_span = result.t[-1] - result.t[0]
        max_response = np.max(np.abs(result.yout))
        print(f"   {condition.capitalize():12}: Time span = {time_span:.3f}s, Max response = {max_response:.2e}m")
    
    # Demonstrate individual simulations with custom parameters
    print("\n7. Demonstrating individual simulations...")
    
    # Normal condition
    print("   Simulating normal condition...")
    normal_result = simulator.simulate_normal()
    
    # Crack with different parameters
    print("   Simulating crack with 30% depth...")
    crack_result = simulator.simulate_crack(
        crack_node=18,
        depth_ratio=0.3,
        crack_model="Gasch"
    )
    
    # Misalignment with angular misalignment only
    print("   Simulating angular misalignment...")
    misalignment_result = simulator.simulate_misalignment(
        misalignment_node=0,
        mis_type="angular",
        mis_angle=10 * np.pi / 180,
        radial_stiffness=50e3,
        bending_stiffness=45e3
    )
    
    # Rubbing with different contact parameters
    print("   Simulating rubbing with higher stiffness...")
    rubbing_result = simulator.simulate_rubbing(
        rubbing_node=12,
        distance=5e-5,
        contact_stiffness=2e6,
        contact_damping=60,
        friction_coeff=0.4
    )
    
    # Extract data for comparison
    print("\n8. Extracting measurement data...")
    
    probe_node = simulator.probes[0].node
    print(f"   Using probe at node {probe_node}")
    
    # Get time series data for each condition
    time = results['normal'].t
    normal_x = results['normal'].yout[probe_node * 6]  # X displacement
    crack_x = results['crack'].yout[probe_node * 6]
    misalign_x = results['misalignment'].yout[probe_node * 6]
    rubbing_x = results['rubbing'].yout[probe_node * 6]
    
    # Calculate some basic statistics
    print(f"   Normal RMS:       {np.sqrt(np.mean(normal_x**2)):.2e} m")
    print(f"   Crack RMS:        {np.sqrt(np.mean(crack_x**2)):.2e} m")
    print(f"   Misalignment RMS: {np.sqrt(np.mean(misalign_x**2)):.2e} m")
    print(f"   Rubbing RMS:      {np.sqrt(np.mean(rubbing_x**2)):.2e} m")
    
    # Demonstrate plotting (optional - requires plotly)
    try:
        print("\n9. Creating comparison plot...")
        fig = simulator.plot_results(results, probe_index=0, show=False)
        print("   Plot created successfully!")
        print("   To display plot, call fig.show() where fig is the returned figure.")
    except Exception as e:
        print(f"   Plotting not available: {e}")
    
    print("\n" + "="*60)
    print("Health State Simulator Example Completed Successfully!")
    print("="*60)
    
    return results


def demonstrate_custom_rotor():
    """Demonstrate with a custom rotor configuration."""
    print("\nCustom Rotor Configuration Example")
    print("==================================")
    
    # Create a simpler rotor configuration
    custom_rotor_config = {
        'material': {
            'name': 'Steel',
            'rho': 7850,
            'E': 2.1e11,
            'Poisson': 0.3
        },
        'shaft': {
            'i_d': 0,
            'o_d': 0.025,
            'lengths': [0.2, 0.2, 0.2, 0.2, 0.2],  # 5 elements of 0.2m each
            'alpha': 0,
            'beta': 0,
            'rotary_inertia': True,
            'shear_effects': True
        },
        'disks': [
            {'n': 2, 'm': 10, 'Id': 0.05, 'Ip': 0.1}
        ],
        'bearings': [
            {'n': 0, 'kxx': 1e6, 'kyy': 1e6, 'cxx': 100, 'cyy': 100},
            {'n': 4, 'kxx': 1e6, 'kyy': 1e6, 'cxx': 100, 'cyy': 100}
        ]
    }
    
    custom_excitation_config = {
        'nodes': [2],
        'unbalance_magnitude': [1e-3],
        'unbalance_phase': [0],
        'speed': Q_(1800, 'RPM'),
        'time': np.arange(0, 1, 0.002)
    }
    
    custom_measurement_config = {
        'probes': [
            {'node': 1, 'angle': 0},
            {'node': 2, 'angle': 0},
            {'node': 3, 'angle': 0}
        ]
    }
    
    # Initialize custom simulator
    custom_simulator = HealthStateSimulator(
        rotor_config=custom_rotor_config,
        excitation_config=custom_excitation_config,
        measurement_config=custom_measurement_config
    )
    
    print(f"Custom rotor has {len(custom_simulator.rotor.shaft_elements)} shaft elements")
    print(f"Custom rotor has {len(custom_simulator.rotor.disk_elements)} disks")
    print(f"Custom rotor has {len(custom_simulator.probes)} measurement probes")
    
    # Simulate all states for custom rotor
    custom_results = custom_simulator.simulate_all_states(
        crack_node=2,
        misalignment_node=0,
        rubbing_node=2
    )
    
    print("\nCustom rotor simulation completed!")
    
    return custom_results


if __name__ == "__main__":
    # Run main example
    main_results = main()
    
    # Run custom rotor example
    custom_results = demonstrate_custom_rotor()
    
    print("\nAll examples completed successfully!")