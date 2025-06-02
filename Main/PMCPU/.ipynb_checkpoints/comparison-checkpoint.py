import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import time
import pandas as pd
import scipy.constants as const
from tqdm import tqdm
import seaborn as sns
import os
import warnings

# Import the PSO implementation
from PSO import (
    crcbqcpsopsd, crcbgenqcsig, normsig4psd, innerprodpsd,
    calculate_matched_filter_snr, pycbc_calculate_match
)

# Import Bilby for MCMC implementation
import bilby
from bilby.core.likelihood import Likelihood
from bilby.core.prior import Uniform, PriorDict
from bilby.gw.detector import InterferometerList
from bilby.gw.likelihood import GravitationalWaveTransient

from pycbc.types import FrequencySeries, TimeSeries
from pycbc.filter import match, matched_filter

# Constants
G = const.G  # Gravitational constant, m^3 kg^-1 s^-2
c = const.c  # Speed of light, m/s
M_sun = 1.989e30  # Solar mass, kg
pc = 3.086e16  # Parsec to meters

# Create results directory
results_dir = "cpu_results_lens_25w"
os.makedirs(results_dir, exist_ok=True)


def load_data():
    """Load gravitational wave data for analysis"""
    print("Loading data...")

    # Load noise data
    TrainingData = scio.loadmat('../generate_data/noise.mat')
    analysisData = scio.loadmat('../generate_data/data.mat')

    print("Data loaded successfully")

    # Convert data to NumPy arrays for CPU processing
    dataY = np.asarray(analysisData['data'][0])
    training_noise = np.asarray(TrainingData['noise'][0])
    dataY_only_signal = dataY - training_noise  # Extract signal part (for comparison)

    # Get basic parameters
    nSamples = dataY.size
    Fs = float(analysisData['samples'][0][0])
    dt = 1 / Fs
    t = np.arange(0, 8, dt)

    # Calculate PSD
    psdHigh = np.asarray(TrainingData['psd'][0])

    return {
        't': t,
        'dataY': dataY,
        'dataY_only_signal': dataY_only_signal,
        'noise': training_noise,
        'psdHigh': psdHigh,
        'sampFreq': Fs,
        'nSamples': nSamples
    }


def setup_parameters(data):
    """Set up parameter ranges for both PSO and MCMC"""

    # Define parameter ranges (min/max)
    param_ranges = {
        'rmin': np.array([-2, 0, 0.1, 0, 0, 0.1]),  # r, mc, tc, phi, A, delta_t
        'rmax': np.array([4, 2, 8.0, np.pi, 1.0, 4.0])
    }

    # Define actual parameters for validation (for synthetic data)
    actual_params = {
        'chirp_mass': 30.09,  # Solar masses
        'merger_time': 7.5,  # seconds
        'source_distance': 3100.0,  # Mpc
        'flux_ratio': 0.3333,  # A parameter
        'time_delay': 0.9854,  # seconds (delta_t)
        'phase': 0.25  # π fraction
    }

    # Set up PSO input parameters
    pso_params = {
        'dataX': data['t'],
        'dataY': data['dataY'],
        'dataY_only_signal': data['dataY_only_signal'],
        'sampFreq': data['sampFreq'],
        'psdHigh': data['psdHigh'],
        'rmin': param_ranges['rmin'],
        'rmax': param_ranges['rmax']
    }

    # Set up PSO configuration
    pso_config = {
        'popsize': 100,  # 80 particles
        'maxSteps': 2500,  # 2000 iterations
        'c1': 2.0,  # Individual learning factor
        'c2': 2.0,  # Social learning factor
        'w_start': 0.9,  # Initial inertia weight
        'w_end': 0.5,  # Final inertia weight
        'max_velocity': 0.4,
        'nbrhdSz': 6,
        'disable_early_stop': True
    }

    return param_ranges, pso_params, pso_config, actual_params


class GWLikelihood(Likelihood):
    """Fixed gravitational wave likelihood function for Bilby"""

    def __init__(self, data_dict):
        """Initialize the likelihood with data"""
        super().__init__(parameters={
            'r': None, 'm_c': None, 'tc': None,
            'phi_c': None, 'A': None, 'delta_t': None
        })
        self.data_dict = data_dict
        
        # Ensure data conversion to NumPy arrays
        self.dataX_np = np.asarray(data_dict['dataX']) if not isinstance(data_dict['dataX'], np.ndarray) else data_dict['dataX']
        self.dataY_only_signal_np = np.asarray(data_dict['dataY_only_signal']) if not isinstance(
            data_dict['dataY_only_signal'], np.ndarray) else data_dict['dataY_only_signal']
        self.psd_np = np.asarray(data_dict['psdHigh']) if not isinstance(data_dict['psdHigh'], np.ndarray) else data_dict['psdHigh']
        
        # Add debug counter
        self.debug_count = 0
        print(f"GWLikelihood initialized with:")
        print(f"  Data shape: {self.dataY_only_signal_np.shape}")
        print(f"  PSD shape: {self.psd_np.shape}")
        print(f"  Sampling frequency: {self.data_dict['sampFreq']}")

    def log_likelihood(self):
        """Fixed log-likelihood function based on reference code approach"""
        try:
            self.debug_count += 1
            
            # Get parameters
            params = np.array([
                self.parameters['r'],
                self.parameters['m_c'],
                self.parameters['tc'],
                self.parameters['phi_c'],
                self.parameters['A'],
                self.parameters['delta_t']
            ])

            # Check if parameters are in [0,1] range
            if not np.all((params >= 0) & (params <= 1)):
                return -np.inf

            # Map from [0,1] to original parameter range
            unscaled_params = np.zeros(6)
            for i in range(6):
                unscaled_params[i] = params[i] * (self.data_dict['rmax'][i] - self.data_dict['rmin'][i]) + \
                                     self.data_dict['rmin'][i]

            r, m_c, tc, phi_c, A, delta_t = unscaled_params

            # Add parameter reasonableness checks
            if r < -2 or r > 4:  # log10(distance)
                return -np.inf
            if m_c < 0 or m_c > 2:  # log10(chirp_mass)
                return -np.inf
            if tc < 0.1 or tc > 8.0:  # merger time
                return -np.inf

            # Determine lensing usage based on A parameter - consistent with PSO
            use_lensing = A >= 0.01

            # Generate signal - using same method as PSO
            signal = crcbgenqcsig(
                self.dataX_np, r, m_c, tc, phi_c, A, delta_t,
                use_lensing=use_lensing
            )

            # Check if signal generation was successful
            if signal is None or np.isnan(signal).any() or np.all(signal == 0):
                if self.debug_count <= 5:  # Only output debug info for first few calls
                    print(f"Debug {self.debug_count}: Signal generation failed")
                return -np.inf

            # Normalize signal - using same method as PSO
            try:
                signal_normalized, normFac = normsig4psd(signal, self.data_dict['sampFreq'], self.psd_np, 1)
                
                if normFac == 0 or np.isnan(normFac) or np.all(signal_normalized == 0):
                    if self.debug_count <= 5:
                        print(f"Debug {self.debug_count}: Normalization failed, normFac={normFac}")
                    return -np.inf
                    
            except Exception as e:
                if self.debug_count <= 5:
                    print(f"Debug {self.debug_count}: Normalization exception: {e}")
                return -np.inf

            # Optimize amplitude - using same method as PSO
            try:
                estAmp = innerprodpsd(
                    self.dataY_only_signal_np, signal_normalized,
                    self.data_dict['sampFreq'], self.psd_np
                )

                # Check for valid amplitude
                if estAmp is None or np.isnan(estAmp) or abs(estAmp) < 1e-15:
                    if self.debug_count <= 5:
                        print(f"Debug {self.debug_count}: Amplitude estimation failed, estAmp={estAmp}")
                    return -np.inf

                signal_final = estAmp * signal_normalized
                
            except Exception as e:
                if self.debug_count <= 5:
                    print(f"Debug {self.debug_count}: Amplitude optimization exception: {e}")
                return -np.inf

            # Calculate log-likelihood using reference code method - using PyCBC match function
            try:
                # Using PyCBC's match function - reference code approach
                delta_t = 1.0 / self.data_dict['sampFreq']
                ts_signal = TimeSeries(signal_final, delta_t=delta_t)
                ts_data = TimeSeries(self.dataY_only_signal_np, delta_t=delta_t)

                delta_f = 1.0 / (len(signal_final) * delta_t)
                psd_series = FrequencySeries(self.psd_np, delta_f=delta_f)

                # Catch potential numerical issues
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    match_value, _ = match(ts_signal, ts_data, psd=psd_series, low_frequency_cutoff=10.0)

                    # Ensure match_value is valid and in range [0,1]
                    if match_value is None or np.isnan(match_value) or match_value <= 0 or match_value > 1:
                        if self.debug_count <= 5:
                            print(f"Debug {self.debug_count}: Invalid match value: {match_value}")
                        return -np.inf

                    # Convert match to log-likelihood (avoiding numerical issues)
                    # Using more robust log-likelihood calculation
                    log_likelihood_value = 0.5 * len(self.dataY_only_signal_np) * np.log(max(match_value, 1e-10))

                    # Catch extreme values
                    if np.isnan(log_likelihood_value) or np.isinf(log_likelihood_value):
                        if self.debug_count <= 5:
                            print(f"Debug {self.debug_count}: Invalid log_likelihood: {log_likelihood_value}")
                        return -np.inf

                    # Output progress info every 1000 calls
                    if self.debug_count % 1000 == 0:
                        print(f"MCMC Progress: {self.debug_count} calls, match={match_value:.6f}, log_L={log_likelihood_value:.2f}")

                    return log_likelihood_value
                    
            except Exception as e:
                if self.debug_count <= 5:
                    print(f"Debug {self.debug_count}: Match calculation error: {str(e)}")
                return -np.inf
                
        except Exception as e:
            if self.debug_count <= 5:
                print(f"Debug {self.debug_count}: General likelihood error: {str(e)}")
            return -np.inf


def run_bilby_mcmc(data_dict, param_ranges, n_live_points=500, n_iter=500):
    """Fixed Bilby MCMC execution function"""
    print("Starting Fixed Bilby MCMC analysis...")

    # Create dictionary for MCMC with numpy arrays - ensure correct data types
    mcmc_data = {
        'dataX': np.asarray(data_dict['t']) if not isinstance(data_dict['t'], np.ndarray) else data_dict['t'],
        'dataY': np.asarray(data_dict['dataY']) if not isinstance(data_dict['dataY'], np.ndarray) else data_dict['dataY'],
        'dataY_only_signal': np.asarray(data_dict['dataY_only_signal']) if not isinstance(data_dict['dataY_only_signal'],
                                                                                      np.ndarray) else data_dict['dataY_only_signal'],
        'sampFreq': data_dict['sampFreq'],
        'psdHigh': np.asarray(data_dict['psdHigh']) if not isinstance(data_dict['psdHigh'], np.ndarray) else data_dict['psdHigh'],
        'rmin': np.asarray(param_ranges['rmin']) if not isinstance(param_ranges['rmin'], np.ndarray) else param_ranges['rmin'],
        'rmax': np.asarray(param_ranges['rmax']) if not isinstance(param_ranges['rmax'], np.ndarray) else param_ranges['rmax']
    }

    # Start timing
    start_time = time.time()

    # Create likelihood
    likelihood = GWLikelihood(mcmc_data)

    # Create priors (uniform in [0,1] for all parameters)
    priors = PriorDict()
    priors['r'] = Uniform(minimum=0, maximum=1, name='r')
    priors['m_c'] = Uniform(minimum=0, maximum=1, name='m_c')
    priors['tc'] = Uniform(minimum=0, maximum=1, name='tc')
    priors['phi_c'] = Uniform(minimum=0, maximum=1, name='phi_c')
    priors['A'] = Uniform(minimum=0, maximum=1, name='A')
    priors['delta_t'] = Uniform(minimum=0, maximum=1, name='delta_t')

    # Store iteration history for MCMC
    mcmc_iteration_history = []

    try:
        # Run sampler using reference code settings
        result = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sampler='dynesty',
            nlive=n_live_points,  # Set number of live points
            walks=25,  # Number of walks for each live point
            verbose=True,
            maxiter=n_iter,  # Maximum iterations
            outdir=results_dir,
            label='gw_analysis_fixed',
            resume=False,
            check_point_plot=False,
            bound='multi',  # More robust boundary scheme
            sample='rwalk',  # Random walk sampling
            check_point=True,  # Enable checkpointing
            check_point_delta_t=600,  # Save every 10 minutes
            plot=True,
        )

        # Extract iteration history from Bilby result
        if hasattr(result, 'sampler') and hasattr(result.sampler, 'results'):
            # Get log evidence evolution over iterations
            sampler_results = result.sampler.results
            if hasattr(sampler_results, 'logz'):
                mcmc_iteration_history = sampler_results.logz
            elif len(result.posterior) > 0:
                # Use log likelihood values as proxy for performance
                mcmc_iteration_history = result.posterior['log_likelihood'].values
        
    except Exception as e:
        print(f"Error in Bilby MCMC: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return minimal result structure to prevent further errors
        end_time = time.time()
        return {
            'duration': end_time - start_time,
            'best_params': {
                'r': 0, 'm_c': 0, 'tc': 0, 'phi_c': 0, 'A': 0, 'delta_t': 0
            },
            'best_signal': np.zeros_like(data_dict['dataY_only_signal']),
            'snr': 0,
            'is_lensed': False,
            'classification': "error",
            'match': 0.0,
            'param_ranges': mcmc_data,
            'iteration_history': []
        }

    end_time = time.time()
    mcmc_duration = end_time - start_time
    print(f"Fixed Bilby MCMC completed in {mcmc_duration:.2f} seconds")

    try:
        # Get best fit parameters (maximum likelihood sample)
        if len(result.posterior) > 0:
            best_idx = np.argmax(result.posterior['log_likelihood'].values)
            best_params_bilby = result.posterior.iloc[best_idx]

            # Unscale parameters to original range
            best_params = np.zeros(6)
            param_names = ['r', 'm_c', 'tc', 'phi_c', 'A', 'delta_t']
            for i, param in enumerate(param_names):
                best_params[i] = best_params_bilby[param] * (mcmc_data['rmax'][i] - mcmc_data['rmin'][i]) + \
                                 mcmc_data['rmin'][i]

            r, m_c, tc, phi_c, A, delta_t = best_params

            print(f"MCMC Best Parameters:")
            print(f"  r (log10 distance): {r:.4f}")
            print(f"  m_c (log10 chirp mass): {m_c:.4f}")
            print(f"  tc (merger time): {tc:.4f}")
            print(f"  phi_c (phase): {phi_c:.4f}")
            print(f"  A (flux ratio): {A:.4f}")
            print(f"  delta_t (time delay): {delta_t:.4f}")

            # Generate best signal using SAME method as PSO
            use_lensing = A >= 0.01
            print(f"MCMC using lensing: {use_lensing}")
            
            best_signal = crcbgenqcsig(
                mcmc_data['dataX'], r, m_c, tc, phi_c, A, delta_t,
                use_lensing=use_lensing
            )
            
            # Apply SAME normalization as PSO
            best_signal, normFac = normsig4psd(best_signal, mcmc_data['sampFreq'], mcmc_data['psdHigh'], 1)
            print(f"MCMC normalization factor: {normFac}")

            # Apply SAME amplitude optimization as PSO
            estAmp = innerprodpsd(
                mcmc_data['dataY_only_signal'], best_signal,
                mcmc_data['sampFreq'], mcmc_data['psdHigh']
            )
            best_signal = estAmp * best_signal
            print(f"MCMC estimated amplitude: {estAmp}")
            print(f"MCMC signal max amplitude: {np.max(np.abs(best_signal))}")

            # Calculate SNR using SAME method as PSO
            snr = calculate_matched_filter_snr(
                best_signal, mcmc_data['dataY_only_signal'],
                mcmc_data['psdHigh'], mcmc_data['sampFreq']
            )

            # Determine classification using SAME logic as PSO
            is_lensed = A >= 0.01
            if snr < 8:
                classification = "noise"
            elif is_lensed:
                classification = "lens_signal"
            else:
                classification = "signal"

            # Calculate match
            match_value = pycbc_calculate_match(
                best_signal, mcmc_data['dataY_only_signal'],
                mcmc_data['sampFreq'], mcmc_data['psdHigh']
            )

            print(f"MCMC Results Summary:")
            print(f"  SNR: {snr:.2f}")
            print(f"  Match: {match_value:.4f}")
            print(f"  Classification: {classification}")
            print(f"  Is Lensed: {is_lensed}")

            # Convert posterior samples to array
            samples = result.posterior[param_names].values
        else:
            print("Warning: No posterior samples found. Using default values.")
            best_params = np.zeros(6)
            r, m_c, tc, phi_c, A, delta_t = best_params
            best_signal = np.zeros_like(mcmc_data['dataY_only_signal'])
            snr = 0
            is_lensed = False
            classification = "error"
            match_value = 0.0
            samples = np.zeros((1, 6))

        # Prepare results
        mcmc_results = {
            'duration': mcmc_duration,
            'best_params': {
                'r': r,
                'm_c': m_c,
                'tc': tc,
                'phi_c': phi_c,
                'A': A,
                'delta_t': delta_t
            },
            'best_signal': best_signal if isinstance(best_signal, np.ndarray) else best_signal,
            'snr': float(snr) if isinstance(snr, (np.ndarray, float)) else snr,
            'is_lensed': is_lensed,
            'classification': classification,
            'match': match_value,  # Use match instead of mismatch
            'samples': samples if 'samples' in locals() else np.zeros((1, 6)),
            'log_probs': result.posterior['log_likelihood'].values if len(result.posterior) > 0 else np.array(
                [-np.inf]),
            'bilby_result': result,  # Store the original Bilby result for additional analysis
            'param_ranges': {
                'rmin': mcmc_data['rmin'],
                'rmax': mcmc_data['rmax']
            },
            'iteration_history': mcmc_iteration_history
        }

        return mcmc_results

    except Exception as e:
        print(f"Error in processing Bilby results: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'duration': mcmc_duration,
            'best_params': {
                'r': 0, 'm_c': 0, 'tc': 0, 'phi_c': 0, 'A': 0, 'delta_t': 0
            },
            'best_signal': np.zeros_like(mcmc_data['dataY_only_signal']),
            'snr': 0,
            'is_lensed': False,
            'classification': "error",
            'match': 0.0,
            'param_ranges': {
                'rmin': mcmc_data['rmin'],
                'rmax': mcmc_data['rmax']
            },
            'iteration_history': []
        }


def run_pso(data_dict, pso_params, pso_config, actual_params, n_runs=1):
    """Run PSO analysis and return results"""
    print("Starting PSO analysis...")

    # Time PSO execution
    start_time = time.time()

    try:
        # Run PSO with multiple starts
        outResults, outStruct = crcbqcpsopsd(
            pso_params, pso_config, n_runs,
            use_two_step=True,
            actual_params=actual_params
        )

        end_time = time.time()
        pso_duration = end_time - start_time
        print(f"PSO completed in {pso_duration:.2f} seconds")

        # Extract best run
        best_run_idx = outResults['bestRun']

        # Get best signal
        best_signal = outResults['bestSig']

        # Calculate match explicitly (instead of mismatch)
        dataY_only_signal = np.asarray(data_dict['dataY_only_signal'])
        match_value = pycbc_calculate_match(
            best_signal,
            dataY_only_signal,
            data_dict['sampFreq'],
            data_dict['psdHigh']
        )

        # Extract iteration history from PSO runs
        pso_iteration_history = []
        if 'fitnessHistory' in outStruct[best_run_idx]:
            # Convert fitness to match values (inverse relationship)
            fitness_history = outStruct[best_run_idx]['fitnessHistory']
            # Convert negative fitness to positive match-like values
            pso_iteration_history = [-f for f in fitness_history]

        # Prepare results dictionary
        pso_results = {
            'duration': pso_duration,
            'best_params': {
                'r': outResults['r'],
                'm_c': outResults['m_c'],
                'tc': outResults['tc'],
                'phi_c': outResults['phi_c'],
                'A': outResults['A'],
                'delta_t': outResults['delta_t']
            },
            'best_signal': best_signal if isinstance(best_signal, np.ndarray) else best_signal,
            'snr': outResults['allRunsOutput'][best_run_idx]['SNR_pycbc'],
            'is_lensed': outResults['is_lensed'],
            'classification': outResults['classification'],
            'match': match_value,  # Use match instead of mismatch
            'all_runs': outResults['allRunsOutput'],
            'structures': outStruct,
            'iteration_history': pso_iteration_history
        }

        return pso_results

    except Exception as e:
        print(f"Error in PSO: {str(e)}")
        # Return minimal results
        return {
            'duration': time.time() - start_time,
            'best_params': {
                'r': 0, 'm_c': 0, 'tc': 0, 'phi_c': 0, 'A': 0, 'delta_t': 0
            },
            'best_signal': np.zeros_like(data_dict['dataY_only_signal']),
            'snr': 0,
            'is_lensed': False,
            'classification': "error",
            'match': 0.0,
            'iteration_history': []
        }


def evaluate_results(pso_results, mcmc_results, actual_params, data_dict):
    """Compare PSO and MCMC results and generate metrics"""

    print("\n============= Performance Comparison =============")

    try:
        # Convert actual parameters to log scale where needed
        actual_r_log = np.log10(actual_params['source_distance'])
        actual_m_c_log = np.log10(actual_params['chirp_mass'])

        # Calculate parameter errors for PSO
        pso_errors = {
            'r': abs((10 ** float(pso_results['best_params']['r']) - actual_params['source_distance']) / actual_params[
                'source_distance']),
            'm_c': abs((10 ** float(pso_results['best_params']['m_c']) - actual_params['chirp_mass']) / actual_params[
                'chirp_mass']),
            'tc': abs((float(pso_results['best_params']['tc']) - actual_params['merger_time']) / actual_params[
                'merger_time']),
            'phi_c': abs(abs(float(pso_results['best_params']['phi_c']) - actual_params['phase'] * np.pi)) / (
                        2 * np.pi),
            'A': abs(
                (float(pso_results['best_params']['A']) - actual_params['flux_ratio']) / actual_params['flux_ratio']),
            'delta_t': abs((float(pso_results['best_params']['delta_t']) - actual_params['time_delay']) / actual_params[
                'time_delay'])
        }

        # Calculate parameter errors for MCMC
        mcmc_errors = {
            'r': abs((10 ** float(mcmc_results['best_params']['r']) - actual_params['source_distance']) / actual_params[
                'source_distance']),
            'm_c': abs((10 ** float(mcmc_results['best_params']['m_c']) - actual_params['chirp_mass']) / actual_params[
                'chirp_mass']),
            'tc': abs((float(mcmc_results['best_params']['tc']) - actual_params['merger_time']) / actual_params[
                'merger_time']),
            'phi_c': abs(abs(float(mcmc_results['best_params']['phi_c']) - actual_params['phase'] * np.pi)) / (
                        2 * np.pi),
            'A': abs(
                (float(mcmc_results['best_params']['A']) - actual_params['flux_ratio']) / actual_params['flux_ratio']),
            'delta_t': abs(
                (float(mcmc_results['best_params']['delta_t']) - actual_params['time_delay']) / actual_params[
                    'time_delay'])
        }

        # Calculate match with actual signal (using match instead of correlation)
        dataY_only_signal_np = np.asarray(data_dict['dataY_only_signal']) if not isinstance(data_dict['dataY_only_signal'],
                                                                                        np.ndarray) else data_dict[
            'dataY_only_signal']

        pso_match = pycbc_calculate_match(pso_results['best_signal'], dataY_only_signal_np, 
                                         data_dict['sampFreq'], data_dict['psdHigh'])
        mcmc_match = pycbc_calculate_match(mcmc_results['best_signal'], dataY_only_signal_np,
                                          data_dict['sampFreq'], data_dict['psdHigh'])

        # Prepare comparison metrics
        comparison = {
            'execution_time': {
                'PSO': pso_results['duration'],
                'MCMC': mcmc_results['duration'],
                'ratio': mcmc_results['duration'] / pso_results['duration'] if pso_results['duration'] > 0 else float('inf')
            },
            'parameter_errors': {
                'PSO': pso_errors,
                'MCMC': mcmc_errors
            },
            'signal_quality': {
                'PSO': {
                    'SNR': float(pso_results['snr']),
                    'match': pso_match,  # Use match instead of correlation
                    'match_final': float(pso_results['match'])  # Use match instead of mismatch
                },
                'MCMC': {
                    'SNR': float(mcmc_results['snr']),
                    'match': mcmc_match,  # Use match instead of correlation
                    'match_final': float(mcmc_results['match'])  # Use match instead of mismatch
                }
            },
            'classification': {
                'PSO': {
                    'is_lensed': pso_results['is_lensed'],
                                        'classification': pso_results['classification'],
                },
                'MCMC': {
                    'is_lensed': mcmc_results['is_lensed'],
                    'classification': mcmc_results['classification'],
                }
            }
        }

        # Print comparison results
        print(
            f"Execution Time: PSO: {comparison['execution_time']['PSO']:.2f}s, Bilby MCMC: {comparison['execution_time']['MCMC']:.2f}s")
        print(f"Speed Improvement: Bilby MCMC is {comparison['execution_time']['ratio']:.2f}x slower than PSO")

        # Print parameter errors
        print("\nParameter Estimation Errors (relative):")
        for param in pso_errors.keys():
            print(f"  {param}: PSO: {pso_errors[param] * 100:.2f}%, Bilby MCMC: {mcmc_errors[param] * 100:.2f}%")

        # Print signal quality metrics
        print("\nSignal Quality Metrics:")
        print(
            f"  SNR: PSO: {comparison['signal_quality']['PSO']['SNR']:.2f}, Bilby MCMC: {comparison['signal_quality']['MCMC']['SNR']:.2f}")
        print(
            f"  Match: PSO: {comparison['signal_quality']['PSO']['match']:.4f}, Bilby MCMC: {comparison['signal_quality']['MCMC']['match']:.4f}")
        print(
            f"  Final Match: PSO: {comparison['signal_quality']['PSO']['match_final']:.4f}, Bilby MCMC: {comparison['signal_quality']['MCMC']['match_final']:.4f}")

        return comparison

    except Exception as e:
        print(f"Error in evaluate_results: {str(e)}")
        # Return minimal comparison structure
        return {
            'execution_time': {
                'PSO': pso_results.get('duration', 0),
                'MCMC': mcmc_results.get('duration', 0),
                'ratio': 1.0
            },
            'parameter_errors': {
                'PSO': {'r': 0, 'm_c': 0, 'tc': 0, 'phi_c': 0, 'A': 0, 'delta_t': 0},
                'MCMC': {'r': 0, 'm_c': 0, 'tc': 0, 'phi_c': 0, 'A': 0, 'delta_t': 0}
            },
            'signal_quality': {
                'PSO': {'SNR': 0, 'match': 0, 'match_final': 0.0},
                'MCMC': {'SNR': 0, 'match': 0, 'match_final': 0.0}
            },
            'classification': {
                'PSO': {'is_lensed': False, 'classification': 'error'},
                'MCMC': {'is_lensed': False, 'classification': 'error'}
            }
        }


def create_parameter_comparison_table(pso_results, mcmc_results, actual_params, param_ranges):
    """Create a comprehensive parameter comparison table"""
    print("\nCreating parameter comparison table...")
    
    try:
        # Parameter names and their descriptions
        param_info = {
            'r': {'name': 'Distance', 'unit': 'Mpc', 'transform': lambda x: 10**x},
            'm_c': {'name': 'Chirp Mass', 'unit': 'M☉', 'transform': lambda x: 10**x},
            'tc': {'name': 'Merger Time', 'unit': 's', 'transform': lambda x: x},
            'phi_c': {'name': 'Phase', 'unit': 'rad', 'transform': lambda x: x/np.pi},
            'A': {'name': 'Flux Ratio', 'unit': '', 'transform': lambda x: x},
            'delta_t': {'name': 'Time Delay', 'unit': 's', 'transform': lambda x: x}
        }
        
        # Actual parameter values
        actual_values = {
            'r': actual_params['source_distance'],
            'm_c': actual_params['chirp_mass'],
            'tc': actual_params['merger_time'],
            'phi_c': actual_params['phase'] * np.pi,
            'A': actual_params['flux_ratio'],
            'delta_t': actual_params['time_delay']
        }
        
        # Create comparison data
        table_data = []
        
        for param in param_info.keys():
            # Get estimated values
            pso_raw = float(pso_results['best_params'][param])
            mcmc_raw = float(mcmc_results['best_params'][param])
            
            # Transform to physical units
            pso_value = param_info[param]['transform'](pso_raw)
            mcmc_value = param_info[param]['transform'](mcmc_raw)
            actual_value = actual_values[param]
            
            # Calculate errors
            pso_error = abs((pso_value - actual_value) / actual_value) * 100
            mcmc_error = abs((mcmc_value - actual_value) / actual_value) * 100
            
            # Add to table
            table_data.append({
                'Parameter': param_info[param]['name'],
                'Unit': param_info[param]['unit'],
                'True Value': f"{actual_value:.4f}",
                'PSO Estimate': f"{pso_value:.4f}",
                'PSO Error (%)': f"{pso_error:.2f}",
                'MCMC Estimate': f"{mcmc_value:.4f}",
                'MCMC Error (%)': f"{mcmc_error:.2f}"
            })
        
        # Create DataFrame
        param_df = pd.DataFrame(table_data)
        
        # Save to CSV
        param_df.to_csv(f"{results_dir}/parameter_comparison_table.csv", index=False)
        
        # Create formatted table plot
        fig, ax = plt.subplots(figsize=(14, 8), dpi=200)
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=param_df.values,
                        colLabels=param_df.columns,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Color header
        for i in range(len(param_df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color cells based on error values
        for i in range(1, len(param_df) + 1):
            # PSO error column
            pso_error_val = float(param_df.iloc[i-1]['PSO Error (%)'])
            if pso_error_val < 5:
                table[(i, 4)].set_facecolor('#E8F5E8')
            elif pso_error_val < 10:
                table[(i, 4)].set_facecolor('#FFF3CD')
            else:
                table[(i, 4)].set_facecolor('#F8D7DA')
            
            # MCMC error column
            mcmc_error_val = float(param_df.iloc[i-1]['MCMC Error (%)'])
            if mcmc_error_val < 5:
                table[(i, 6)].set_facecolor('#E8F5E8')
            elif mcmc_error_val < 10:
                table[(i, 6)].set_facecolor('#FFF3CD')
            else:
                table[(i, 6)].set_facecolor('#F8D7DA')
        
        plt.title('Parameter Estimation Comparison: PSO vs MCMC', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.savefig(f"{results_dir}/parameter_comparison_table.png", 
                   bbox_inches='tight', dpi=200)
        plt.close()
        
        print(f"Parameter comparison table saved to {results_dir}/parameter_comparison_table.csv")
        print(f"Parameter comparison table plot saved to {results_dir}/parameter_comparison_table.png")
        
        return param_df
        
    except Exception as e:
        print(f"Error creating parameter comparison table: {str(e)}")
        return pd.DataFrame()


def create_signal_ratio_plot(pso_results, mcmc_results, data_dict):
    """Create signal ratio plot comparing PSO and MCMC reconstructed signals to injected signal"""
    print("\nCreating signal ratio plot...")
    
    try:
        # Extract data
        t_np = np.array(data_dict['t'])
        injected = np.array(data_dict['dataY_only_signal'])
        pso_signal = np.array(pso_results['best_signal'])
        mcmc_signal = np.array(mcmc_results['best_signal'])
        
        # Handle zero division for 10^-21 scale signals
        epsilon = 1e-25
        
        # Calculate ratios, handling zeros
        pso_ratio = np.divide(pso_signal, injected, 
                             out=np.ones_like(pso_signal), 
                             where=np.abs(injected) > epsilon)
        
        mcmc_ratio = np.divide(mcmc_signal, injected, 
                              out=np.ones_like(mcmc_signal), 
                              where=np.abs(injected) > epsilon)
        
        # Clip extreme values for better visualization
        pso_ratio = np.clip(pso_ratio, -5, 5)
        mcmc_ratio = np.clip(mcmc_ratio, -5, 5)
        
        # Create plot
        plt.figure(figsize=(12, 6), dpi=150)
        
        plt.plot(t_np, pso_ratio, 'r-', linewidth=1.2, alpha=0.8, label='PSO/Injected')
        plt.plot(t_np, mcmc_ratio, 'b-', linewidth=1.2, alpha=0.8, label='MCMC/Injected')
        plt.axhline(y=1, color='black', linestyle='--', alpha=0.7, linewidth=1.5, 
                   label='Perfect Reconstruction')
        
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Ratio', fontsize=12)
        plt.title('Signal Reconstruction Ratio Comparison', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-5.5, 5.5)
        
        plt.tight_layout()
        plt.savefig(f"{results_dir}/signal_ratio_comparison.png", 
                   bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"Signal ratio plot saved to {results_dir}/signal_ratio_comparison.png")
        
        return {'status': 'success'}
        
    except Exception as e:
        print(f"Error creating signal ratio plot: {e}")
        return {'status': 'error'}


def generate_comparison_plots(pso_results, mcmc_results, data_dict, comparison, actual_params, param_ranges=None):
    """Generate plots comparing PSO and MCMC results"""
    print("\nGenerating comparison plots...")

    try:
        # Get data as numpy arrays for plotting
        t_np = np.asarray(data_dict['t']) if not isinstance(data_dict['t'], np.ndarray) else data_dict['t']
        dataY_np = np.asarray(data_dict['dataY']) if not isinstance(data_dict['dataY'], np.ndarray) else data_dict['dataY']
        dataY_only_signal_np = np.asarray(data_dict['dataY_only_signal']) if not isinstance(data_dict['dataY_only_signal'],
                                                                                        np.ndarray) else data_dict[
            'dataY_only_signal']

        # 1. Signal Comparison Plot
        plt.figure(figsize=(12, 8), dpi=200)
        plt.plot(t_np, dataY_np, 'gray', alpha=0.3, label='Observed Data (Signal + Noise)')
        plt.plot(t_np, dataY_only_signal_np, 'k', lw=1.5, label='Actual Signal (No Noise)')
        plt.plot(t_np, pso_results['best_signal'], 'r', lw=1.5,
                 label=f'PSO Estimate (SNR: {comparison["signal_quality"]["PSO"]["SNR"]:.2f})')
        plt.plot(t_np, mcmc_results['best_signal'], 'b', lw=1.5,
                 label=f'Bilby MCMC Estimate (SNR: {comparison["signal_quality"]["MCMC"]["SNR"]:.2f})')
        plt.title('Signal Reconstruction Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Strain', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{results_dir}/signal_comparison.png", bbox_inches='tight')
        plt.close()

        # 2. Residual Plot
        plt.figure(figsize=(12, 6), dpi=200)
        pso_residual = pso_results['best_signal'] - dataY_only_signal_np
        mcmc_residual = mcmc_results['best_signal'] - dataY_only_signal_np
        plt.plot(t_np, pso_residual, 'r', alpha=0.7,
                 label=f'PSO Residual (Match: {comparison["signal_quality"]["PSO"]["match_final"]:.4f})')
        plt.plot(t_np, mcmc_residual, 'b', alpha=0.7,
                 label=f'Bilby MCMC Residual (Match: {comparison["signal_quality"]["MCMC"]["match_final"]:.4f})')
        plt.title('Residual Comparison (Estimate - Actual Signal)', fontsize=14, fontweight='bold')
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Strain Difference', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{results_dir}/residual_comparison.png", bbox_inches='tight')
        plt.close()

        # 3. Parameter Errors Bar Plot
        param_names = ['r', 'm_c', 'tc', 'phi_c', 'A', 'delta_t']
        param_labels = ['Distance', 'Chirp Mass', 'Merger Time', 'Phase', 'Flux Ratio (A)', 'Time Delay']

        pso_error_vals = [comparison['parameter_errors']['PSO'][p] * 100 for p in param_names]
        mcmc_error_vals = [comparison['parameter_errors']['MCMC'][p] * 100 for p in param_names]

        plt.figure(figsize=(12, 8), dpi=200)
        x = np.arange(len(param_names))
        width = 0.35

        plt.bar(x - width / 2, pso_error_vals, width, label='PSO', color='red', alpha=0.7)
        plt.bar(x + width / 2, mcmc_error_vals, width, label='Bilby MCMC', color='blue', alpha=0.7)

        plt.xlabel('Parameters', fontsize=12)
        plt.ylabel('Relative Error (%)', fontsize=12)
        plt.title('Parameter Estimation Error Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x, param_labels, rotation=45)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{results_dir}/parameter_errors.png", bbox_inches='tight')
        plt.close()

        # 4. Execution Time Comparison
        plt.figure(figsize=(8, 6), dpi=200)
        plt.bar(['PSO', 'Bilby MCMC'],
                [comparison['execution_time']['PSO'], comparison['execution_time']['MCMC']],
                color=['red', 'blue'], alpha=0.7)
        plt.ylabel('Execution Time (seconds)', fontsize=12)
        plt.title(f'Execution Time Comparison\nBilby MCMC is {comparison["execution_time"]["ratio"]:.2f}x slower', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        for i, v in enumerate([comparison['execution_time']['PSO'], comparison['execution_time']['MCMC']]):
            plt.text(i, v + 1, f"{v:.2f}s", ha='center', fontsize=10)
        plt.savefig(f"{results_dir}/execution_time.png", bbox_inches='tight')
        plt.close()

        print(f"All comparison plots saved to {results_dir} directory")

    except Exception as e:
        print(f"Error in generate_comparison_plots: {str(e)}")


def main():
    """Main function to run the PSO vs Bilby MCMC comparison"""

    print("=" * 50)
    print("PSO vs Bilby MCMC Comparison for Gravitational Wave Analysis")
    print("=" * 50)

    try:
        # Load data
        data_dict = load_data()

        # Set up parameters
        param_ranges, pso_params, pso_config, actual_params = setup_parameters(data_dict)

        # Run PSO
        pso_results = run_pso(data_dict, pso_params, pso_config, actual_params)

        # Run Bilby MCMC with parameters to match PSO computational effort
        # PSO: 80 particles × 2000 iterations = 160,000 evaluations
        # MCMC: 400 live points × 400 iterations = 160,000 evaluations
        mcmc_results = run_bilby_mcmc(data_dict, param_ranges, n_live_points=400, n_iter=400)

        # Evaluate and compare results
        comparison = evaluate_results(pso_results, mcmc_results, actual_params, data_dict)

        # Create parameter comparison table
        param_table = create_parameter_comparison_table(pso_results, mcmc_results, actual_params, param_ranges)

        # Create signal ratio plot
        ratio_stats = create_signal_ratio_plot(pso_results, mcmc_results, data_dict)

        # Generate comparison plots
        generate_comparison_plots(pso_results, mcmc_results, data_dict, comparison, actual_params, param_ranges)

        # Save comparison summary
        summary_df = pd.DataFrame({
            'Metric': [
                'Execution Time (s)',
                'Speed Improvement',
                'SNR',
                'Match (Signal Quality)',
                'Final Match'
            ],
            'PSO': [
                f"{comparison['execution_time']['PSO']:.2f}",
                f"{comparison['execution_time']['ratio']:.2f}x faster",
                f"{comparison['signal_quality']['PSO']['SNR']:.2f}",
                f"{comparison['signal_quality']['PSO']['match']:.4f}",
                f"{comparison['signal_quality']['PSO']['match_final']:.4f}",
            ],
            'Bilby MCMC': [
                f"{comparison['execution_time']['MCMC']:.2f}",
                "baseline",
                f"{comparison['signal_quality']['MCMC']['SNR']:.2f}",
                f"{comparison['signal_quality']['MCMC']['match']:.4f}",
                f"{comparison['signal_quality']['MCMC']['match_final']:.4f}",
            ]
        })

        summary_df.to_csv(f"{results_dir}/summary.csv", index=False)

        print("\nComparison complete! Results saved in the cpu_results_fixed directory.")
        print(f"PSO is {comparison['execution_time']['ratio']:.2f}x faster than Bilby MCMC")

        # Generate final message based on overall comparison
        pso_wins = 0
        mcmc_wins = 0

        # Compare execution time
        if comparison['execution_time']['PSO'] < comparison['execution_time']['MCMC']:
            pso_wins += 1
        else:
            mcmc_wins += 1

        # Compare SNR
        if comparison['signal_quality']['PSO']['SNR'] > comparison['signal_quality']['MCMC']['SNR']:
            pso_wins += 1
        else:
            mcmc_wins += 1

        # Compare match (correlation)
        if comparison['signal_quality']['PSO']['match'] > comparison['signal_quality']['MCMC']['match']:
            pso_wins += 1
        else:
            mcmc_wins += 1

        # Compare final match (higher is better)
        if comparison['signal_quality']['PSO']['match_final'] > comparison['signal_quality']['MCMC']['match_final']:
            pso_wins += 1
        else:
            mcmc_wins += 1

        # Compare parameter errors
        pso_avg_error = np.mean(
            [comparison['parameter_errors']['PSO'][p] for p in comparison['parameter_errors']['PSO']])
        mcmc_avg_error = np.mean(
            [comparison['parameter_errors']['MCMC'][p] for p in comparison['parameter_errors']['MCMC']])

        if pso_avg_error < mcmc_avg_error:
            pso_wins += 1
        else:
            mcmc_wins += 1

        if pso_wins > mcmc_wins:
            print(f"\nConclusion: PSO outperforms Bilby MCMC with {pso_wins} wins vs {mcmc_wins} wins for Bilby MCMC")
            print(
                "PSO demonstrates superior performance in gravitational wave parameter estimation, combining faster execution with equal or better accuracy.")
        elif mcmc_wins > pso_wins:
            print(f"\nConclusion: Bilby MCMC outperforms PSO with {mcmc_wins} wins vs {pso_wins} wins for PSO")
            print("Bilby MCMC demonstrates better overall accuracy despite slower execution time.")
        else:
            print(f"\nConclusion: PSO and Bilby MCMC are tied with {pso_wins} wins each")
            print("PSO offers significant speed advantages with comparable accuracy to Bilby MCMC.")

    except Exception as e:
        print(f"Error in main function: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()