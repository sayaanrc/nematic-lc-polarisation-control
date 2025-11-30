import numpy as np
import matplotlib.pyplot as plt

def jones_retarder(theta, delta):
    c = np.cos(delta / 2)
    s = np.sin(delta / 2)
    cos2t = np.cos(2 * theta)
    sin2t = np.sin(2 * theta)
    return np.array([
        [c - 1j * s * cos2t, -1j * s * sin2t],
        [-1j * s * sin2t, c + 1j * s * cos2t]
    ])

def simulate():
    #Default inputs
    psi = float(input('Orientation angle:'))  # Orientation angle (degrees)
    chi = float(input('Ellipticity angle:'))  # Ellipticity angle (degrees)
    
    # Convert angles to Jones vector
    psi_rad = np.radians(psi)
    chi_rad = np.radians(chi)
    cos_chi = np.cos(chi_rad)
    sin_chi = np.sin(chi_rad)
    cos_psi = np.cos(psi_rad)
    sin_psi = np.sin(psi_rad)
    Ex = cos_psi * cos_chi + 1j * sin_psi * sin_chi
    Ey = -sin_psi * cos_chi + 1j * cos_psi * sin_chi
    jones_target = np.array([Ex, Ey], dtype=complex)
    
    # Normalize target
    magnitude = np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2)
    if magnitude > 0:
        jones_target /= magnitude
    
    # Calculate retardances
    Ex, Ey = jones_target
    Ex_mag = np.abs(Ex)
    Ey_mag = np.abs(Ey)
    
    delta1 = 2 * np.arctan2(Ey_mag, Ex_mag)
    if Ex_mag > 1e-10 and Ey_mag > 1e-10:
        phase_ratio = np.angle(Ey / Ex)
        delta2 = (-phase_ratio - np.pi / 2) % (2 * np.pi)
    else:
        delta2 = 0
    
    # Simulate polarization
    jones_input = np.array([1, 0], dtype=complex)
    jones_matrix1 = jones_retarder(np.pi / 4, delta1)
    jones_matrix2 = jones_retarder(0, delta2)
    jones_intermediate = np.dot(jones_matrix1, jones_input)
    jones_output = np.dot(jones_matrix2, jones_intermediate)
    
    # Normalize output
    magnitude = np.sqrt(np.abs(jones_output[0])**2 + np.abs(jones_output[1])**2)
    if magnitude > 0:
        jones_output /= magnitude
    
    # Computing output psi and chi from Stokes parameters
    Ex, Ey = jones_output
    S0 = np.abs(Ex)**2 + np.abs(Ey)**2
    S1 = np.abs(Ex)**2 - np.abs(Ey)**2
    S2 = 2 * np.real(Ex * np.conj(Ey))
    S3 = 2 * np.imag(Ex * np.conj(Ey))
    psi_out = 0.5 * np.arctan2(S2, S1) * 180 / np.pi
    chi_out = 0.5 * np.arcsin(S3 / S0) * 180 / np.pi
    
    # Adjusting psi_out to [0, 180°] and chi_out to match input sign
    if psi_out < 0:
        psi_out += 180
    chi_out = abs(chi_out) if chi >= 0 else -abs(chi_out)
    
    return delta1, delta2, jones_output, psi, chi, psi_out, chi_out

# Run simulation
delta1, delta2, jones_output, psi, chi, psi_out, chi_out = simulate()

print("Results:")
print(f"LC1 Retardance (45°): {delta1:.3f} rad ({delta1 * 180 / np.pi:.1f}°)")
print(f"LC2 Retardance (0°): {delta2:.3f} rad ({delta2 * 180 / np.pi:.1f}°)")
print(f"Target Orientation Angle ψ: {psi:.1f}°")
print(f"Target Ellipticity Angle χ: {chi:.1f}°")
print(f"Output Orientation Angle ψ: {psi_out:.1f}°")
print(f"Output Ellipticity Angle χ: {chi_out:.1f}°")

# Plot
Ex, Ey = jones_output
t = np.linspace(0, 2 * np.pi, 100)
Ex_t = np.real(Ex * np.exp(1j * t))
Ey_t = np.real(Ey * np.exp(1j * t))
plt.figure(figsize=(6, 6))
plt.plot(Ex_t, Ey_t, 'b', label=f'ψ={psi:.1f}°, χ={chi:.1f}°')
plt.xlabel('Re[E_x]')
plt.ylabel('Re[E_y]')
plt.title('Output Polarization Ellipse')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.show()
