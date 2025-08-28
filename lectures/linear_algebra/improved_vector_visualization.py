"""
Enhanced Vector Span Visualization with Comprehensive Documentation

This module provides an improved visualization of vector spans in 3D space with
clear documentation and distinct colors for better understanding.

Author: Enhanced version for educational purposes
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def find_coefficients(vectors, target):
    """
    Find scalar coefficients to express target vector as linear combination of given vectors.
    
    This function solves the equation: c1*v1 + c2*v2 + ... + cn*vn = target
    where c1, c2, ..., cn are the coefficients we want to find.
    
    Parameters:
    -----------
    vectors : list of numpy arrays
        List of basis vectors (each should be a 1D numpy array)
    target : numpy array
        The target vector we want to express as a linear combination
        
    Returns:
    --------
    numpy array or None
        Array of coefficients if solution exists, None if no solution
        
    Example:
    --------
    >>> v1 = np.array([1, 0])
    >>> v2 = np.array([0, 1]) 
    >>> target = np.array([3, 4])
    >>> coeffs = find_coefficients([v1, v2], target)
    >>> print(coeffs)  # Should output [3, 4]
    """
    # Stack vectors as columns to form coefficient matrix A
    A = np.column_stack(vectors)
    
    try:
        # Solve the linear system A * coefficients = target
        coefficients = np.linalg.solve(A, target)
        
        # Verify the solution by checking if reconstruction matches target
        reconstruction = sum(coeff * vec for coeff, vec in zip(coefficients, vectors))
        
        # Check if solution is accurate within numerical tolerance
        if np.allclose(reconstruction, target, rtol=1e-10):
            return coefficients
        else:
            return None
            
    except np.linalg.LinAlgError:
        # Matrix is singular (vectors are linearly dependent)
        return None


def create_span_surface(v, w, grid_range=(-2, 3), grid_points=20):
    """
    Create a surface representing the span of two 3D vectors.
    
    The span of two vectors v and w is the set of all linear combinations:
    {a*v + b*w | a, b are real numbers}
    
    Parameters:
    -----------
    v, w : numpy arrays
        Two 3D vectors that define the span
    grid_range : tuple
        Range for the parameter grid (min, max)
    grid_points : int
        Number of grid points in each direction
        
    Returns:
    --------
    tuple
        (X, Y, Z) coordinates for the surface mesh
    """
    # Create parameter grid for linear combinations
    a_vals = np.linspace(grid_range[0], grid_range[1], grid_points)
    b_vals = np.linspace(grid_range[0], grid_range[1], grid_points)
    A, B = np.meshgrid(a_vals, b_vals)
    
    # Calculate surface points: each point is A[i,j]*v + B[i,j]*w
    X = A * v[0] + B * w[0]  # x-coordinates of span surface
    Y = A * v[1] + B * w[1]  # y-coordinates of span surface  
    Z = A * v[2] + B * w[2]  # z-coordinates of span surface
    
    return X, Y, Z


def visualize_vector_span():
    """
    Create an enhanced 3D visualization of vector spans with clear documentation.
    
    This function demonstrates:
    1. Individual vectors v, w, and target t
    2. The span (plane) formed by vectors v and w
    3. Whether target t lies in the span of v and w
    4. The linear combination that reconstructs t (if it exists)
    """
    
    # Define our vectors with clear, distinct values
    v = np.array([1, 2, 1])    # First basis vector (blue)
    w = np.array([3, 1, 2])    # Second basis vector (green)  
    t = np.array([5, 7, 5])    # Target vector (red)
    
    print("Vector Analysis:")
    print(f"v = {v}")
    print(f"w = {w}")  
    print(f"t = {t}")
    
    # Try to express t as linear combination of v and w
    coeffs = find_coefficients([v, w], t)
    
    if coeffs is not None:
        print(f"\nTarget t CAN be expressed as: {coeffs[0]:.3f}*v + {coeffs[1]:.3f}*w")
        reconstruction = coeffs[0] * v + coeffs[1] * w
        print(f"Verification: {coeffs[0]:.3f}*{v} + {coeffs[1]:.3f}*{w} = {reconstruction}")
    else:
        print(f"\nTarget t CANNOT be expressed as a linear combination of v and w")
        print("This means t is NOT in the span of {v, w}")
    
    # Create the 3D plot with enhanced styling
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define distinct, colorblind-friendly colors
    colors = {
        'v': '#1f77b4',        # Blue - first basis vector
        'w': '#2ca02c',        # Green - second basis vector  
        't': '#d62728',        # Red - target vector
        'reconstruction': '#ff7f0e',  # Orange - reconstructed vector
        'span': '#9467bd',     # Purple - span surface (semi-transparent)
        'origin': '#000000'    # Black - origin point
    }
    
    # Plot the origin point
    ax.scatter([0], [0], [0], color=colors['origin'], s=100, label='Origin')
    
    # Plot the original vectors as arrows from origin
    ax.quiver(0, 0, 0, v[0], v[1], v[2], 
              color=colors['v'], arrow_length_ratio=0.1, linewidth=3, 
              label=f'v = {v}')
    
    ax.quiver(0, 0, 0, w[0], w[1], w[2], 
              color=colors['w'], arrow_length_ratio=0.1, linewidth=3,
              label=f'w = {w}')
    
    ax.quiver(0, 0, 0, t[0], t[1], t[2], 
              color=colors['t'], arrow_length_ratio=0.1, linewidth=3,
              label=f't = {t}')
    
    # Plot the reconstruction if it exists
    if coeffs is not None:
        reconstruction = coeffs[0] * v + coeffs[1] * w
        ax.quiver(0, 0, 0, reconstruction[0], reconstruction[1], reconstruction[2], 
                  color=colors['reconstruction'], linestyle='--', linewidth=2,
                  arrow_length_ratio=0.1,
                  label=f'Reconstruction: {coeffs[0]:.2f}v + {coeffs[1]:.2f}w')
    
    # Create and plot the span surface (plane through origin)
    X, Y, Z = create_span_surface(v, w)
    ax.plot_surface(X, Y, Z, alpha=0.3, color=colors['span'])
    
    # Enhanced plot styling
    ax.set_xlabel('X Axis', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Axis', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z Axis', fontsize=12, fontweight='bold')
    
    # Set equal aspect ratio for better visualization
    max_range = max(np.max(np.abs([v, w, t])), 8)
    ax.set_xlim([-max_range/2, max_range])
    ax.set_ylim([-max_range/2, max_range]) 
    ax.set_zlim([-max_range/2, max_range])
    
    # Add legend (simple approach)
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
    
    title = 'Enhanced Vector Span Visualization\n'
    if coeffs is not None:
        title += f't ∈ span{{v, w}} ✓ (t = {coeffs[0]:.2f}v + {coeffs[1]:.2f}w)'
    else:
        title += 't ∉ span{v, w} ✗'
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add grid for better depth perception
    ax.grid(True, alpha=0.3)
    
    # Improve viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax


def demonstrate_different_cases():
    """
    Demonstrate different cases of vector relationships with clear examples.
    """
    print("=" * 60)
    print("DEMONSTRATION: Different Vector Span Cases")
    print("=" * 60)
    
    # Case 1: Target is in the span
    print("\nCASE 1: Target vector IS in the span")
    print("-" * 40)
    v1 = np.array([1, 0, 0])
    w1 = np.array([0, 1, 0]) 
    t1 = np.array([2, 3, 0])  # Clearly in xy-plane
    
    coeffs1 = find_coefficients([v1, w1], t1)
    print(f"v = {v1}, w = {w1}, t = {t1}")
    if coeffs1 is not None:
        print(f"✓ t = {coeffs1[0]}v + {coeffs1[1]}w")
    
    # Case 2: Target is NOT in the span  
    print("\nCASE 2: Target vector is NOT in the span")
    print("-" * 40)
    v2 = np.array([1, 0, 0])
    w2 = np.array([0, 1, 0])
    t2 = np.array([1, 1, 1])  # Has z-component, not in xy-plane
    
    coeffs2 = find_coefficients([v2, w2], t2)
    print(f"v = {v2}, w = {w2}, t = {t2}")
    if coeffs2 is None:
        print("✗ t cannot be expressed as av + bw (t has z-component)")


if __name__ == "__main__":
    """
    Main execution: Run the enhanced vector span visualization.
    """
    print("Starting Enhanced Vector Span Visualization...")
    print("This will show vectors v, w, target t, and their span relationship.\n")
    
    # Run the main visualization
    fig, ax = visualize_vector_span()
    
    # Show additional examples
    demonstrate_different_cases()
    
    print("\n" + "=" * 60)
    print("Visualization complete! The plot shows:")
    print("• Blue arrow: Vector v")
    print("• Green arrow: Vector w") 
    print("• Red arrow: Target vector t")
    print("• Orange dashed arrow: Reconstruction (if t is in span)")
    print("• Purple surface: The span of vectors v and w")
    print("=" * 60)
