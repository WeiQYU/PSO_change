import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import cupy as cp
import emcee
import time
import argparse
from tqdm import tqdm

# Constants
G = 6.67430e-11  # Gravitational constant, m^3 kg^-1 s^-2
c = 2.998e8  # Speed of light, m/s
M_sun = 1.989e30  # Solar mass, kg
pc = 3.086e16  # Parsec in meters


def load_data(noise_path, data_path):
    """
    Load data files

    Parameters:
    noise_path: Path to noise data file (.mat)
    data_path: Path to signal data file (.mat)

    Returns:
    Tuple of (noise_data, signal_data)
    """
    print(f"Loading data from {noise_path} and {data_path}")
    noise_data = scio.loadmat(noise_path)
    signal_data = scio.loadmat(data_path)

    print("Data loading complete")
    return noise_data, signal_data


def crcbgenqcsig(dataX, r, m_c, tc, phi_c, A, delta_t):
    """
    Generate quadratic chirp signal

    Parameters:
    dataX: Time array
    r: Log10 of distance parameter
    m_c: Log10 of chirp mass parameter
    tc: Coalescence time
    phi_c: Coalescence phase
    A: Amplitude
    delta_t: Log10 of time duration

    Returns:
    Generated signal
    """
    # Convert to physical units
    r_phys = (10 ** r) * 1e6 * pc
    m_c_phys = (10 ** m_c) * M_sun
    delta_t_phys = (10 ** delta_t)

    # Calculate chirp mass parameter
    eta = (G * m_c_phys / c ** 3) ** (5 / 3)

    # Phase calculation
    phase = cp.zeros(len(dataX))
    idx = dataX <= tc

    # Calculate phase for pre-coalescence
    t_diff = tc - dataX[idx]
    phase[idx] = 2 * cp.pi * delta_t_phys * (t_diff - 0.5 * t_diff ** 2 / delta_t_phys)

    # Signal calculation
    sig = cp.zeros(len(dataX))
    sig[idx] = A * cp.cos(phi_c + phase[idx]) / (r_phys / pc)

    return sig


def generate_waveform(dataX, params):
    """
    Generate gravitational wave waveform

    Parameters:
    dataX: Time array
    params: Signal parameter dictionary, including r, m_c, tc, phi_c, A, delta_t

    Returns:
    Generated waveform
    """
    # Use waveform generation function
    waveform = crcbgenqcsig(
        dataX,
        params['r'],
        params['m_c'],
        params['tc'],
        params['phi_c'],
        params['A'],
        params['delta_t']
    )
    return waveform


def normsig4psd(sig, fs, psd, td_or_fd=1):
    """
    Normalize signal for PSD

    Parameters:
    sig: Signal
    fs: Sampling frequency
    psd: Power spectral density
    td_or_fd: Time domain (1) or frequency domain (2)

    Returns:
    Normalized signal
    """
    sigft = cp.fft.rfft(sig)
    df = fs / len(sig)
    sigft_norm = sigft / cp.sqrt(psd * df / 2)

    if td_or_fd == 1:
        # Time domain
        sig_norm = cp.fft.irfft(sigft_norm)
        return sig_norm, sigft_norm
    else:
        # Frequency domain
        return sigft_norm


def innerprodpsd(sig1, sig2, fs, psd):
    """
    Calculate inner product with PSD weighting

    Parameters:
    sig1, sig2: Signals
    fs: Sampling frequency
    psd: Power spectral density

    Returns:
    Inner product value
    """
    sig1ft = cp.fft.rfft(sig1)
    sig2ft = cp.fft.rfft(sig2)

    df = fs / len(sig1)
    innerp = 2 * cp.sum((sig1ft * cp.conj(sig2ft)) / (psd * df / 2)) * df

    return cp.real(innerp)


def setup_mcmc_model(data, time_array, sampFreq, psd, params_init=None):
    """
    Set up MCMC model

    Parameters:
    data: Observed data
    time_array: Time array
    sampFreq: Sampling frequency
    psd: Power spectral density
    params_init: Initial parameter estimates (optional)

    Returns:
    Configured MCMC model and prior
    """
    # Create variable parameter dictionary
    variable_params = [
        'r', 'm_c', 'tc', 'phi_c', 'A', 'delta_t'
    ]

    # Implement custom distribution class and prior distributions
    class CustomUniform:
        """Custom uniform distribution"""

        def __init__(self, name, min_bound, max_bound):
            self.name = name
            self.bounds = (min_bound, max_bound)

    # Set prior distribution ranges - use custom distribution class
    prior_dict = {
        'r': CustomUniform(name='r', min_bound=-2, max_bound=4),
        'm_c': CustomUniform(name='m_c', min_bound=0, max_bound=3),
        'tc': CustomUniform(name='tc', min_bound=0, max_bound=10),
        'phi_c': CustomUniform(name='phi_c', min_bound=0, max_bound=2 * np.pi),
        'A': CustomUniform(name='A', min_bound=0, max_bound=1),
        'delta_t': CustomUniform(name='delta_t', min_bound=0, max_bound=2),
    }

    # Create prior object - custom prior distribution dictionary
    class CustomPriorDict(dict):
        """Custom prior distribution dictionary class"""

        def __init__(self, prior_dict):
            super().__init__(prior_dict)

        def apply_boundary_conditions(self, **params):
            """Apply boundary conditions, ensure parameters are within range"""
            for param, value in params.items():
                if param in self:
                    if value < self[param].bounds[0]:
                        params[param] = self[param].bounds[0]
                    elif value > self[param].bounds[1]:
                        params[param] = self[param].bounds[1]
            return params

    prior = CustomPriorDict(prior_dict)

    # Create likelihood model
    class LensedGravitationalWaveModel:
        """Lensed gravitational wave signal model"""

        def __init__(self, variable_params, data, time_array, sampFreq, psd):
            self.variable_params = variable_params
            self.data = data
            self.time_array = time_array
            self.sampFreq = sampFreq
            self.psd = psd

        def _loglikelihood(self, **params):
            """
            Calculate log likelihood function
            """
            # Generate model waveform
            model_waveform = generate_waveform(self.time_array, params)

            # Calculate log likelihood
            model_waveform_norm, _ = normsig4psd(model_waveform, self.sampFreq, self.psd, 1)
            inner_prod = innerprodpsd(self.data, model_waveform_norm, self.sampFreq, self.psd)
            logL = inner_prod ** 2

            return float(cp.asnumpy(logL))

    # Create model instance
    model = LensedGravitationalWaveModel(
        variable_params=variable_params,
        data=cp.asarray(data),
        time_array=cp.asarray(time_array),
        sampFreq=sampFreq,
        psd=cp.asarray(psd)
    )

    return model, prior


def run_mcmc(model, prior, nwalkers=200, niterations=5000, params_init=None):
    """
    Run MCMC sampling

    Parameters:
    model: MCMC model
    prior: Prior distribution
    nwalkers: Number of walkers
    niterations: Number of iterations
    params_init: Initial parameter estimates (optional)

    Returns:
    MCMC sampling results
    """
    # Set initial points
    initial_points = {}

    if params_init is not None:
        # If there are initial results, use them as an initial point
        initial_point = {
            'r': params_init['r'],
            'm_c': params_init['m_c'],
            'tc': params_init['tc'],
            'phi_c': params_init['phi_c'],
            'A': params_init['A'],
            'delta_t': params_init['delta_t']
        }

        # Generate multiple walkers around the initial point
        for param in initial_point:
            param_range = prior[param].bounds[1] - prior[param].bounds[0]
            values = initial_point[param] + param_range * 0.01 * np.random.uniform(-1, 1, size=nwalkers)
            # Ensure values are within prior range
            values = np.clip(values, prior[param].bounds[0], prior[param].bounds[1])
            initial_points[param] = values
    else:
        # Randomly sample from prior distribution as initial points
        for param in model.variable_params:
            # Generate uniform random numbers
            min_val = prior[param].bounds[0]
            max_val = prior[param].bounds[1]
            initial_points[param] = np.random.uniform(min_val, max_val, size=nwalkers)

    # Define adapter class to directly use emcee library
    class EmceeAdapter:
        def __init__(self, model, nwalkers, niterations):
            self.model = model
            self.nwalkers = nwalkers
            self.niterations = niterations
            self.ndim = len(model.variable_params)
            self.variable_params = model.variable_params

            # Initialize emcee sampler
            self.sampler = emcee.EnsembleSampler(
                nwalkers,
                self.ndim,
                self._log_probability
            )

            # Store results
            self.all_samples = None
            self.all_lnprob = None

        def _log_probability(self, x):
            # Convert position vector to parameter dictionary
            params = {self.variable_params[i]: x[i] for i in range(len(x))}

            # Check if parameters are within prior range
            for param, value in params.items():
                if value < prior[param].bounds[0] or value > prior[param].bounds[1]:
                    return -np.inf

            # Calculate log likelihood
            try:
                return self.model._loglikelihood(**params)
            except Exception as e:
                print(f"Error calculating likelihood: {e}")
                return -np.inf

        def set_p0(self, initial_points):
            # Convert initial point format
            self.p0 = np.zeros((self.nwalkers, self.ndim))
            for i, param in enumerate(self.variable_params):
                self.p0[:, i] = initial_points[param]

        def run(self):
            # Run sampler
            print(f"Starting MCMC sampling with {self.nwalkers} walkers, {self.niterations} iterations...")

            # Save chains from all steps, not just the final positions
            self.all_samples = np.zeros((self.niterations, self.nwalkers, self.ndim))
            self.all_lnprob = np.zeros((self.niterations, self.nwalkers))

            pos = self.p0
            # Use tqdm to create a progress bar
            with tqdm(total=self.niterations, desc="MCMC Progress",
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
                for i in range(self.niterations):
                    # Perform one sampling step
                    pos, lnprob, _ = self.sampler.run_mcmc(pos, 1, progress=False)

                    # Store current state
                    self.all_samples[i] = pos
                    self.all_lnprob[i] = lnprob

                    # Update progress bar
                    pbar.update(1)

                    # Optional: add dynamic information to progress bar description
                    if (i + 1) % 50 == 0:
                        current_max_lnprob = np.max(lnprob)
                        pbar.set_description(f"MCMC Progress (max lnL: {current_max_lnprob:.2f})")

            print("MCMC sampling complete")
            return self.all_samples

        def get_chains(self):
            # Return sampling chains, format [ntemps, nwalkers, niterations, ndim]
            # Here we simulate a single temperature chain
            reshaped_samples = self.all_samples.transpose(1, 0, 2)  # [nwalkers, niterations, ndim]
            return np.array([reshaped_samples])  # [1, nwalkers, niterations, ndim]

        def get_posterior(self, flatten=False):
            # Return posterior probability
            if flatten:
                return self.all_lnprob.T.flatten()  # Transpose and flatten, consistent with chains format
            return self.all_lnprob.T

        def get_posterior_samples(self, param, flatten=True):
            # Get posterior samples for a parameter
            idx = self.variable_params.index(param)
            samples = self.all_samples[:, :, idx].T  # [nwalkers, niterations]
            if flatten:
                return samples.flatten()
            return samples

        @property
        def chains(self):
            # Compatible with PyCBC interface
            return self.get_chains()

    # Use emcee adapter
    emcee_sampler = EmceeAdapter(model, nwalkers, niterations)

    # Set initial points
    emcee_sampler.set_p0(initial_points)

    # Run MCMC sampling
    emcee_sampler.run()

    return emcee_sampler


def analyze_mcmc_results(sampler, burn_in=1000):
    """
    Analyze MCMC sampling results

    Parameters:
    sampler: MCMC sampler
    burn_in: Number of burn-in samples

    Returns:
    Analysis results
    """
    # Get sampling chains
    chains = sampler.get_chains()

    # If only a portion of data, adjust burn_in
    if burn_in >= chains.shape[2]:
        burn_in = int(chains.shape[2] * 0.5)  # Use half of the samples
        print(f"Warning: burn_in value exceeds the number of available samples, adjusted to {burn_in}")

    # Apply burn-in period
    chains = chains[:, :, burn_in:, :]

    # Get chains from the highest temperature (i.e., target distribution) and flatten to [n_samples, n_params]
    samples = chains[0].reshape(-1, chains.shape[-1])

    # Calculate parameter mean values
    params_mean = {param: np.mean(samples[:, i])
                   for i, param in enumerate(sampler.variable_params)}

    # Get the maximum likelihood point (MAP point)
    ln_post = sampler.get_posterior(flatten=True)
    if burn_in > 0:
        # Apply the same burn-in period
        n_per_walker = len(ln_post) // sampler.nwalkers
        ln_post = ln_post.reshape(sampler.nwalkers, n_per_walker)[:, burn_in:].flatten()

    map_idx = np.argmax(ln_post)

    # Get posterior samples for parameters, and apply the same burn-in period
    params_map = {}
    for param in sampler.variable_params:
        param_samples = sampler.get_posterior_samples(param, flatten=False)
        if burn_in > 0:
            param_samples = param_samples[:, burn_in:]
        params_map[param] = param_samples.flatten()[map_idx]

    # Calculate 95% confidence intervals for each parameter
    params_intervals = {}
    for i, param in enumerate(sampler.variable_params):
        param_samples = samples[:, i]
        lower, upper = np.percentile(param_samples, [2.5, 97.5])
        params_intervals[param] = (lower, upper)

    results = {
        'mean': params_mean,
        'map': params_map,
        'intervals': params_intervals,
        'samples': samples,
        'param_names': sampler.variable_params
    }

    return results


def generate_signals_from_chains(mcmc_results, data_x, n_samples=10):
    """
    Generate multiple signal waveforms from MCMC chains

    Parameters:
    mcmc_results: MCMC results
    data_x: Time array
    n_samples: Number of signals to generate

    Returns:
    List of signals
    """
    # Randomly select samples
    samples = mcmc_results['samples']
    param_names = mcmc_results['param_names']
    n_total = samples.shape[0]

    # Ensure n_samples does not exceed the number of available samples
    n_samples = min(n_samples, n_total)

    # Select samples at equal intervals
    indices = np.linspace(0, n_total - 1, n_samples, dtype=int)

    signals = []
    params_list = []

    for idx in indices:
        # Extract parameters
        params = {param: samples[idx, i] for i, param in enumerate(param_names)}

        # Generate signal
        signal = generate_waveform(cp.asarray(data_x), params)
        signals.append(cp.asnumpy(signal))
        params_list.append(params)

    return signals, params_list


def classify_signal(data, signal, noise, sampFreq, psd):
    """
    Classify signal type: pure noise, unlensed signal, or lensed signal

    Parameters:
    data: Original data
    signal: Estimated signal
    noise: Noise
    sampFreq: Sampling frequency
    psd: Power spectral density

    Returns:
    Classification result string
    """
    # Calculate SNR
    signal_cp = cp.asarray(signal)
    data_cp = cp.asarray(data)
    psd_cp = cp.asarray(psd)

    # Calculate match value
    match_value = innerprodpsd(signal_cp, data_cp, sampFreq, psd_cp) / \
                  cp.sqrt(innerprodpsd(signal_cp, signal_cp, sampFreq, psd_cp) * \
                          innerprodpsd(data_cp, data_cp, sampFreq, psd_cp))

    # Calculate signal-to-noise ratio
    snr = cp.sqrt(innerprodpsd(signal_cp, data_cp, sampFreq, psd_cp))

    # Calculate mismatch parameter
    epsilon = 1 - match_value

    # Calculate SNR threshold
    rho = snr ** (-2)

    # Classify signal based on conditions
    snr_value = float(cp.asnumpy(snr))
    epsilon_value = float(cp.asnumpy(epsilon))
    rho_value = float(cp.asnumpy(rho))

    if snr_value < 8:
        return "Pure Noise", snr_value, epsilon_value, rho_value
    else:
        if epsilon_value > rho_value:
            return "Lensed Signal", snr_value, epsilon_value, rho_value
        else:
            return "Unlensed Signal", snr_value, epsilon_value, rho_value


def plot_results(time, data, signals, params_list, noise, classification, mcmc_results, output_prefix="mcmc_results"):
    """
    Plot result images

    Parameters:
    time: Time array
    data: Original data
    signals: List of estimated signals
    params_list: Parameter list
    noise: Noise data
    classification: Classification result
    mcmc_results: MCMC results
    output_prefix: Output filename prefix
    """
    # 1. Plot time domain signal
    plt.figure(figsize=(12, 8))
    plt.plot(time, data, 'gray', alpha=0.5, label='Observed Data')

    # Plot MAP signal
    map_params = mcmc_results['map']
    map_signal = generate_waveform(cp.asarray(time), map_params)
    map_signal = cp.asnumpy(map_signal)
    plt.plot(time, map_signal, 'r', linewidth=2, label='Maximum A Posteriori (MAP) Signal')

    # Plot sampled signals
    for i, signal in enumerate(signals):
        if i == 0:
            plt.plot(time, signal, 'g', alpha=0.3, label='MCMC Sample Signals')
        else:
            plt.plot(time, signal, 'g', alpha=0.3)

    plt.xlabel('Time (s)')
    plt.ylabel('Strain')
    plt.title('Gravitational Wave Signal Waveform')
    plt.legend()
    plt.savefig(f'{output_prefix}_waveforms.png', dpi=200)
    plt.close()

    # 2. Plot parameter distributions
    samples = mcmc_results['samples']
    param_names = mcmc_results['param_names']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, param in enumerate(param_names):
        ax = axes[i]
        ax.hist(samples[:, i], bins=50, alpha=0.8)
        ax.axvline(mcmc_results['mean'][param], color='r', linestyle='-', label='Mean')
        ax.axvline(mcmc_results['map'][param], color='g', linestyle='--', label='MAP')
        ax.axvline(mcmc_results['intervals'][param][0], color='k', linestyle=':', label='95% CI')
        ax.axvline(mcmc_results['intervals'][param][1], color='k', linestyle=':')

        # Parameter post-processing, certain parameters need to be converted back to physical units
        if param == 'r' or param == 'm_c' or param == 'delta_t':
            # Convert to physical units
            param_value = 10 ** mcmc_results['map'][param]
            if param == 'r':
                param_unit = "Mpc"
            elif param == 'm_c':
                param_unit = "M_sun"
            else:
                param_unit = "s"
            ax.set_title(f'{param}: {param_value:.4f} {param_unit}')
        elif param == 'phi_c':
            ax.set_title(f'{param}: {mcmc_results["map"][param] / np.pi:.4f}π')
        else:
            ax.set_title(f'{param}: {mcmc_results["map"][param]:.4f}')

        if i == 0 or i == 3:  # Only add legend to the first and fourth subplots
            ax.legend()

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_parameter_distributions.png', dpi=200)
    plt.close()

    # 3. Plot comprehensive results
    class_result, snr, epsilon, rho = classification

    plt.figure(figsize=(15, 10))

    # Top left: Signal time domain plot
    plt.subplot(221)
    plt.plot(time, data, 'gray', alpha=0.5, label='Observed Data')
    plt.plot(time, map_signal, 'r', linewidth=2, label='MAP Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Strain')
    plt.title('Gravitational Wave Signal Waveform')
    plt.legend()

    # Bottom left: Residual plot
    plt.subplot(223)
    residual = data - map_signal
    plt.plot(time, residual, 'b', alpha=0.7, label='Residual')
    plt.plot(time, noise, 'g', alpha=0.3, label='Noise')
    plt.xlabel('Time (s)')
    plt.ylabel('Strain')
    plt.title('Residual vs. Noise Comparison')
    plt.legend()

    # Right side: Results summary
    plt.subplot(122)
    plt.axis('off')

    # Title
    plt.text(0.1, 0.95, "Gravitational Wave Signal Analysis Results", fontsize=16, fontweight='bold')

    # Classification result
    plt.text(0.1, 0.88, f"Signal Classification: {class_result}", fontsize=14,
             color='blue' if class_result == "Pure Noise" else 'red' if class_result == "Lensed Signal" else 'green',
             fontweight='bold')

    # Key metrics
    plt.text(0.1, 0.82, f"Signal-to-Noise Ratio (SNR): {snr:.4f}", fontsize=12)
    plt.text(0.1, 0.78, f"Mismatch Parameter (ε): {epsilon:.4f}", fontsize=12)
    plt.text(0.1, 0.74, f"Threshold Parameter (ρ): {rho:.4f}", fontsize=12)

    # Parameter estimation results
    plt.text(0.1, 0.66, "Parameter Estimation Results (MAP):", fontsize=14, fontweight='bold')

    param_y = 0.62
    for param in param_names:
        if param == 'r' or param == 'm_c' or param == 'delta_t':
            # Convert to physical units
            param_value = 10 ** mcmc_results['map'][param]
            if param == 'r':
                param_text = f"r: {param_value:.4f} Mpc"
            elif param == 'm_c':
                param_text = f"m_c: {param_value:.4f} M_sun"
            else:
                param_text = f"delta_t: {param_value:.4f} s"
        elif param == 'phi_c':
            param_text = f"phi_c: {mcmc_results['map'][param] / np.pi:.4f}π"
        else:
            param_text = f"{param}: {mcmc_results['map'][param]:.4f}"

        plt.text(0.1, param_y, param_text, fontsize=12)
        param_y -= 0.04

    # MCMC statistics
    plt.text(0.1, param_y - 0.04, "MCMC Statistics:", fontsize=14, fontweight='bold')
    plt.text(0.1, param_y - 0.08, f"Number of Samples: {len(samples)}", fontsize=12)

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_summary.png', dpi=200)
    plt.close()


def save_results_to_file(mcmc_results, classification, output_prefix="mcmc_results"):
    """
    Save analysis results to a text file

    Parameters:
    mcmc_results: MCMC analysis results
    classification: Signal classification results
    output_prefix: Output filename prefix
    """
    class_result, snr, epsilon, rho = classification

    with open(f"{output_prefix}_results.txt", "w") as f:
        f.write("============= MCMC Analysis Results =============\n\n")
        f.write(f"Signal Classification: {class_result}\n")
        f.write(f"Signal-to-Noise Ratio (SNR): {snr:.4f}\n")
        f.write(f"Mismatch Parameter (ε): {epsilon:.4f}\n")
        f.write(f"Threshold Parameter (ρ): {rho:.4f}\n\n")

        f.write("Parameter Estimation Results (MAP):\n")
        for param in mcmc_results['param_names']:
            if param == 'r' or param == 'm_c' or param == 'delta_t':
                # Convert to physical units
                param_value = 10 ** mcmc_results['map'][param]
                if param == 'r':
                    f.write(f"r: {param_value:.4f} Mpc\n")
                elif param == 'm_c':
                    f.write(f"m_c: {param_value:.4f} M_sun\n")
                else:
                    f.write(f"delta_t: {param_value:.4f} s\n")
            elif param == 'phi_c':
                f.write(f"phi_c: {mcmc_results['map'][param] / np.pi:.4f}π\n")
            else:
                f.write(f"{param}: {mcmc_results['map'][param]:.4f}\n")

        f.write("\n95% Confidence Intervals:\n")
        for param in mcmc_results['param_names']:
            lower, upper = mcmc_results['intervals'][param]
            f.write(f"{param}: [{lower:.4f}, {upper:.4f}]\n")

    print(f"Results saved to {output_prefix}_results.txt")


def main():
    """
    Main function
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MCMC analysis for gravitational wave signals')
    parser.add_argument('--noise', type=str, default='../generate_ligo/noise.mat',
                        help='Path to noise data file (.mat)')
    parser.add_argument('--data', type=str, default='../generate_ligo/data.mat',
                        help='Path to signal data file (.mat)')
    parser.add_argument('--walkers', type=int, default=100,
                        help='Number of MCMC walkers')
    parser.add_argument('--iterations', type=int, default=2000,
                        help='Number of MCMC iterations')
    parser.add_argument('--burn_in', type=int, default=500,
                        help='Number of burn-in samples')
    parser.add_argument('--output', type=str, default='mcmc_results',
                        help='Output filename prefix')
    parser.add_argument('--random_init', action='store_true',
                        help='Use random initialization instead of manual initial parameters')

    args = parser.parse_args()

    # Start timing
    start_time = time.time()

    # Load data
    noise_path = args.noise
    data_path = args.data

    noise_data, signal_data = load_data(noise_path, data_path)

    # Extract data
    noise = noise_data['noise'][0]
    data = signal_data['data'][0]
    psd = noise_data['psd'][0]
    Fs = float(signal_data['samples'][0][0])

    # Set time array
    dt = 1 / Fs
    t = np.arange(-90, 10, dt)

    # Set initial parameters
    if args.random_init:
        # Use random initialization
        init_params = None
        print("Using random initialization for MCMC walkers")
    else:
        # Set manual initial parameters
        init_params = {
            'r': 1.0,  # Log10 of distance in Mpc
            'm_c': 1.5,  # Log10 of chirp mass in solar masses
            'tc': 5.0,  # Coalescence time
            'phi_c': np.pi / 2,  # Coalescence phase
            'A': 0.5,  # Amplitude
            'delta_t': 1.0,  # Log10 of time duration
        }
        print("Using manual initialization for MCMC walkers:")
        for key, value in init_params.items():
            if key in ['r', 'm_c', 'delta_t']:
                print(f"{key}: {10 ** value:.4f}")
            elif key == 'phi_c':
                print(f"{key}: {value / np.pi:.4f}π")
            else:
                print(f"{key}: {value:.4f}")

    # Set up MCMC model
    print("Setting up MCMC model...")
    model, prior = setup_mcmc_model(data, t, Fs, psd, init_params)

    # Run MCMC sampling
    print("Starting MCMC sampling...")
    sampler = run_mcmc(model, prior, nwalkers=args.walkers, niterations=args.iterations, params_init=init_params)

    # Analyze MCMC results
    print("Analyzing MCMC results...")
    mcmc_results = analyze_mcmc_results(sampler, burn_in=args.burn_in)

    # Generate sample signals
    print("Generating sample signals...")
    sample_signals, sample_params = generate_signals_from_chains(mcmc_results, t, n_samples=10)

    # Generate signal using MAP parameters
    map_params = mcmc_results['map']
    map_signal = generate_waveform(cp.asarray(t), map_params)
    map_signal = cp.asnumpy(map_signal)

    # Classify signal
    print("Classifying signal...")
    classification = classify_signal(data, map_signal, noise, Fs, psd)

    # Plot results
    print("Plotting results...")
    plot_results(t, data, sample_signals, sample_params, noise, classification, mcmc_results, output_prefix=args.output)

    # Save results to text file
    save_results_to_file(mcmc_results, classification, output_prefix=args.output)

    # Calculate total runtime
    end_time = time.time()
    total_runtime = end_time - start_time
    hours, remainder = divmod(total_runtime, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Print results
    class_result, snr, epsilon, rho = classification
    print("\n============= Final Results =============")
    print(f"Signal Classification: {class_result}")
    print(f"Signal-to-Noise Ratio (SNR): {snr:.4f}")
    print(f"Mismatch Parameter (ε): {epsilon:.4f}")
    print(f"Threshold Parameter (ρ): {rho:.4f}")

    print("\nParameter Estimation Results (MAP):")
    for param in mcmc_results['param_names']:
        if param == 'r' or param == 'm_c' or param == 'delta_t':
            # Convert to physical units
            param_value = 10 ** mcmc_results['map'][param]
            if param == 'r':
                print(f"r: {param_value:.4f} Mpc")
            elif param == 'm_c':
                print(f"m_c: {param_value:.4f} M_sun")
            else:
                print(f"delta_t: {param_value:.4f} s")
        elif param == 'phi_c':
            print(f"phi_c: {mcmc_results['map'][param] / np.pi:.4f}π")
        else:
            print(f"{param}: {mcmc_results['map'][param]:.4f}")

    print("\n95% Confidence Intervals:")
    for param in mcmc_results['param_names']:
        lower, upper = mcmc_results['intervals'][param]
        print(f"{param}: [{lower:.4f}, {upper:.4f}]")

    # Print total runtime
    print(f"\nTotal Runtime: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")
    print(f"Results saved to {args.output}_*.png and {args.output}_results.txt")


if __name__ == "__main__":
    main()