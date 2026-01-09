import numpy as np

def ggx_theta_energy_percent(roughness, energy=0.99, n_theta=200_000):
    """
    Compute the polar angle theta (radians) that contains `energy`
    fraction of GGX NDF energy over the hemisphere.

    Parameters
    ----------
    roughness : float
        Disney/UE roughness in [0,1]
    energy : float
        Target cumulative energy (e.g. 0.99)
    n_theta : int
        Number of samples for numerical integration

    Returns
    -------
    theta : float
        Polar angle in radians
    """
    # GGX alpha
    alpha = roughness ** 2

    # Theta grid [0, pi/2]
    theta = np.linspace(0.0, 0.5 * np.pi, n_theta)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # GGX NDF (Trowbridge–Reitz)
    D = (alpha ** 2) / (
        np.pi * (cos_t**2 * (alpha**2 - 1.0) + 1.0) ** 2
    )

    # Energy density over solid angle
    # dω = sinθ dθ dφ , φ integrated over [0, 2π]
    pdf = D * sin_t * 2.0 * np.pi

    # Cumulative energy
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]  # normalize to [0,1]

    # Find theta where cumulative energy reaches target
    idx = np.searchsorted(cdf, energy)
    return theta[min(idx, n_theta - 1)]


def generate_theta75_table():
    roughness_values = np.arange(0.01, 1.001, 0.01)
    theta99_deg = []
    tanh99 = []

    for r in roughness_values:
        theta = ggx_theta_energy_percent(r, energy=0.75)
        theta99_deg.append(np.degrees(theta))
        tanh99.append(np.sin(np.degrees(theta) / 180 * np.pi) / np.cos(np.degrees(theta) /  180 * np.pi))

    return roughness_values, np.array(theta99_deg),np.array(tanh99)

def generate_theta99_table():
    roughness_values = np.arange(0.01, 1.001, 0.01)
    theta99_deg = []
    tanh99 = []

    for r in roughness_values:
        theta = ggx_theta_energy_percent(r, energy=0.99)
        theta99_deg.append(np.degrees(theta))
        tanh99.append(np.sin(np.degrees(theta) / 180 * np.pi) / np.cos(np.degrees(theta) /  180 * np.pi))

    return roughness_values, np.array(theta99_deg),np.array(tanh99)
if __name__ == "__main__":
    roughness, theta99,tanh99 = generate_theta99_table()
    roughness, theta75,tanh75 = generate_theta75_table()

    print("roughness =", roughness.tolist())
    print("theta99_deg =", np.round(theta99, 3).tolist())
    print("tan99", np.round(tanh99, 3).tolist())
    # print("tan99", np.round(tanh99, 3).tolist())
    # print("tan99/75 ,"  , np.round(tanh99/ tanh75, 3).tolist())
    # print("tan99/75 ,"  , np.round(theta75, 3).tolist())
    # print("tan99/75 ,"  , np.round(tanh75, 3).tolist())