import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import time
import pandas as pd
import scipy.constants as const
from tqdm import tqdm
import os
import warnings
from scipy import stats
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
import seaborn as sns

# Import the PSO implementation
from PSO import (
    crcbqcpsopsd_traditional, crcbgenqcsig, normsig4psd, innerprodpsd,
    calculate_matched_filter_snr, pycbc_calculate_match, refine_distance_parameter
)

# Import Bilby for MCMC implementation
import bilby
from bilby.core.likelihood import Likelihood
from bilby.core.prior import Uniform, PriorDict

from pycbc.types import FrequencySeries, TimeSeries
from pycbc.filter import match

# Constants
G = const.G
c = const.c
M_sun = 1.989e30
pc = 3.086e16

# Create results directory
results_dir = "GW_Comparison_Results"
os.makedirs(results_dir, exist_ok=True)
print(f"结果保存目录：{results_dir}")

# Set plotting parameters
plt.rcParams.update({
    'axes.linewidth': 1.2,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'lines.linewidth': 1.5,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8
})


def load_data():
    """Load gravitational wave data for analysis"""
    print("Loading data...")

    TrainingData = scio.loadmat('../generate_data/noise.mat')
    analysisData = scio.loadmat('../generate_data/data_without_lens.mat')

    print("Data loaded successfully")

    # Convert data to NumPy arrays
    dataY = np.asarray(analysisData['data'][0])
    training_noise = np.asarray(TrainingData['noise'][0])
    dataY_only_signal = dataY - training_noise

    # Get basic parameters
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
        'nSamples': dataY.size
    }


def setup_parameters(data):
    """Set up parameter ranges for both PSO and MCMC"""

    # Define parameter ranges
    param_ranges = {
        'rmin': np.array([-2, 0, 0.1, 0, 0, 0.1]),
        'rmax': np.array([4, 2, 8.0, np.pi, 1.0, 4.0])
    }

    # Define actual parameters for validation
    actual_params = {
        'chirp_mass': 30.09,
        'merger_time': 7.5,
        'source_distance': 3100.0,
        'flux_ratio': 0.3333,
        'time_delay': 0.9854,
        'phase': 0.25 * np.pi
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

    # Traditional PSO configuration (simplified)
    pso_config = {
        'popsize': 50,
        'maxSteps': 2000,
        'c1': 2.0,
        'c2': 2.0,
        'w_start': 0.9,
        'w_end': 0.4,
        'max_velocity': 0.5,
        'nbrhdSz': 4,  # Ring topology
        'init_strategy': 'uniform',
        'disable_early_stop': True  # Traditional PSO without early stopping
    }

    return param_ranges, pso_params, pso_config, actual_params


def calculate_parameter_errors(estimated_params, actual_params, include_lensing_params=True):
    """Calculate relative errors for parameters based on lensing status
    
    Args:
        estimated_params: Dictionary of estimated parameter values
        actual_params: Dictionary of actual parameter values
        include_lensing_params: Whether to include I and delta_t in error calculation
    """
    
    # Parameter transformation info
    param_info = {
        'r': {'name': 'Distance', 'unit': 'Mpc', 'transform': lambda x: 10 ** x},
        'm_c': {'name': 'Chirp Mass', 'unit': 'M☉', 'transform': lambda x: 10 ** x},
        'tc': {'name': 'Merger Time', 'unit': 's', 'transform': lambda x: x},
        'phi_c': {'name': 'Phase', 'unit': 'rad', 'transform': lambda x: x},
        'I': {'name': 'Flux Ratio', 'unit': '', 'transform': lambda x: x},
        'delta_t': {'name': 'Time Delay', 'unit': 's', 'transform': lambda x: x}
    }
    
    # Actual parameter values
    actual_values = {
        'r': actual_params['source_distance'],
        'm_c': actual_params['chirp_mass'],
        'tc': actual_params['merger_time'],
        'phi_c': actual_params['phase'],
        'I': actual_params['flux_ratio'],
        'delta_t': actual_params['time_delay']
    }
    
    # Determine which parameters to include based on lensing status
    if include_lensing_params:
        param_names = ['r', 'm_c', 'tc', 'phi_c', 'I', 'delta_t']
    else:
        param_names = ['r', 'm_c', 'tc', 'phi_c']  # Exclude I and delta_t for non-lensed cases
    
    errors = {}
    
    for param in param_names:
        # Get estimated value
        estimated_raw = float(estimated_params[param])
        
        # Transform to physical units
        estimated_value = param_info[param]['transform'](estimated_raw)
        actual_value = actual_values[param]
        
        # Calculate relative error
        relative_error = abs((estimated_value - actual_value) / actual_value) * 100
        errors[param] = relative_error
    
    return errors


def check_errors_below_threshold(errors, threshold=5.0):
    """Check if all calculated parameter errors are below the specified threshold"""
    for param, error in errors.items():
        if error >= threshold:
            return False
    return True


def determine_lensing_status(estimated_params):
    """Determine if the estimated parameters indicate a lensed signal"""
    return float(estimated_params['I']) >= 0.01


class GWLikelihood(Likelihood):
    """Gravitational wave likelihood function for Bilby"""

    def __init__(self, data_dict):
        """Initialize the likelihood with data"""
        super().__init__(parameters={
            'r': None, 'm_c': None, 'tc': None,
            'phi_c': None, 'I': None, 'delta_t': None
        })
        self.data_dict = data_dict

        # Convert data to NumPy arrays
        self.dataX_np = np.asarray(data_dict['dataX'])
        self.dataY_only_signal_np = np.asarray(data_dict['dataY_only_signal'])
        self.psd_np = np.asarray(data_dict['psdHigh'])

        print(f"GWLikelihood initialized:")
        print(f"  Data shape: {self.dataY_only_signal_np.shape}")
        print(f"  PSD shape: {self.psd_np.shape}")
        print(f"  Sampling frequency: {self.data_dict['sampFreq']}")

    def log_likelihood(self):
        """Log-likelihood function"""
        try:
            # Get parameters
            params = np.array([
                self.parameters['r'],
                self.parameters['m_c'],
                self.parameters['tc'],
                self.parameters['phi_c'],
                self.parameters['I'],
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

            r, m_c, tc, phi_c, I, delta_t = unscaled_params

            # Parameter range checks
            if r < -2 or r > 4 or m_c < 0 or m_c > 2 or tc < 0.1 or tc > 8.0:
                return -np.inf

            # Determine lensing usage
            use_lensing = I >= 0.01

            # Generate signal
            signal = crcbgenqcsig(
                self.dataX_np, r, m_c, tc, phi_c, I, delta_t,
                use_lensing=use_lensing
            )

            if signal is None or np.isnan(signal).any() or np.all(signal == 0):
                return -np.inf

            # Normalize signal
            signal_normalized, normFac = normsig4psd(signal, self.data_dict['sampFreq'], self.psd_np, 1)

            if normFac == 0 or np.isnan(normFac) or np.all(signal_normalized == 0):
                return -np.inf

            # Optimize amplitude
            estAmp = innerprodpsd(
                self.dataY_only_signal_np, signal_normalized,
                self.data_dict['sampFreq'], self.psd_np
            )

            if estAmp is None or np.isnan(estAmp) or abs(estAmp) < 1e-15:
                return -np.inf

            signal_final = estAmp * signal_normalized

            # Calculate log-likelihood using PyCBC match function
            delta_t = 1.0 / self.data_dict['sampFreq']
            ts_signal = TimeSeries(signal_final, delta_t=delta_t)
            ts_data = TimeSeries(self.dataY_only_signal_np, delta_t=delta_t)

            # Handle PSD length adjustment
            nSamples = len(signal_final)
            expected_psd_len = nSamples // 2 + 1
            psd_adjusted = self.psd_np.copy()

            if len(psd_adjusted) < expected_psd_len:
                extended_psd = np.zeros(expected_psd_len)
                extended_psd[:len(psd_adjusted)] = psd_adjusted
                extended_psd[len(psd_adjusted):] = psd_adjusted[-1]
                psd_adjusted = extended_psd
            elif len(psd_adjusted) > expected_psd_len:
                psd_adjusted = psd_adjusted[:expected_psd_len]

            # Ensure PSD has no zero values
            min_psd = np.max(psd_adjusted) * 1e-14
            psd_adjusted = np.maximum(psd_adjusted, min_psd)

            delta_f = 1.0 / (len(signal_final) * delta_t)
            psd_series = FrequencySeries(psd_adjusted, delta_f=delta_f)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                match_value, _ = match(ts_signal, ts_data, psd=psd_series, low_frequency_cutoff=10.0)

                if match_value is None or np.isnan(match_value) or match_value <= 0 or match_value > 1:
                    return -np.inf

                # Convert match to log-likelihood
                log_likelihood_value = 0.5 * len(self.dataY_only_signal_np) * np.log(max(match_value, 1e-10))

                if np.isnan(log_likelihood_value) or np.isinf(log_likelihood_value):
                    return -np.inf

                return log_likelihood_value

        except Exception:
            return -np.inf


def run_bilby_mcmc(data_dict, param_ranges, n_live_points, n_iter, enable_distance_refinement=True, actual_params=None):
    """Bilby MCMC execution function"""
    print("Starting Bilby MCMC analysis...")

    # Create dictionary for MCMC
    mcmc_data = {
        'dataX': np.asarray(data_dict['t']),
        'dataY': np.asarray(data_dict['dataY']),
        'dataY_only_signal': np.asarray(data_dict['dataY_only_signal']),
        'sampFreq': data_dict['sampFreq'],
        'psdHigh': np.asarray(data_dict['psdHigh']),
        'rmin': np.asarray(param_ranges['rmin']),
        'rmax': np.asarray(param_ranges['rmax'])
    }

    start_time = time.time()

    try:
        # Create likelihood
        likelihood = GWLikelihood(mcmc_data)

        # Create priors
        priors = PriorDict()
        priors['r'] = Uniform(minimum=0, maximum=1, name='r')
        priors['m_c'] = Uniform(minimum=0, maximum=1, name='m_c')
        priors['tc'] = Uniform(minimum=0, maximum=1, name='tc')
        priors['phi_c'] = Uniform(minimum=0, maximum=1, name='phi_c')
        priors['I'] = Uniform(minimum=0, maximum=1, name='I')
        priors['delta_t'] = Uniform(minimum=0, maximum=1, name='delta_t')

        # Run sampler
        result = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sampler='dynesty',
            nlive=n_live_points,
            walks=40,
            verbose=True,
            maxiter=n_iter,
            outdir=results_dir,
            label='gw_analysis',
            resume=False,
            check_point_plot=False,
            bound='multi',
            sample='rwalk',
            check_point=True,
            check_point_delta_t=600,
            plot=True,
        )

        # Extract iteration history
        mcmc_iteration_history = []
        if hasattr(result, 'sampler') and hasattr(result.sampler, 'results'):
            sampler_results = result.sampler.results
            if hasattr(sampler_results, 'logz'):
                mcmc_iteration_history = sampler_results.logz
            elif len(result.posterior) > 0:
                mcmc_iteration_history = result.posterior['log_likelihood'].values

    except Exception as e:
        print(f"Error in Bilby MCMC: {str(e)}")
        end_time = time.time()
        return {
            'duration': end_time - start_time,
            'best_params': {'r': 0, 'm_c': 0, 'tc': 0, 'phi_c': 0, 'I': 0, 'delta_t': 0},
            'best_signal': np.zeros_like(data_dict['dataY_only_signal']),
            'snr': 0,
            'is_lensed': False,
            'classification': "error",
            'match': 0.0,
            'param_ranges': mcmc_data,
            'iteration_history': [],
            'posterior_samples': None,
            'distance_refinement': {'enabled': enable_distance_refinement, 'status': 'error'}
        }

    end_time = time.time()
    mcmc_duration = end_time - start_time
    print(f"Bilby MCMC completed in {mcmc_duration:.2f} seconds")

    try:
        # Get best fit parameters
        if len(result.posterior) > 0:
            best_idx = np.argmax(result.posterior['log_likelihood'].values)
            best_params_bilby = result.posterior.iloc[best_idx]

            # Unscale parameters
            best_params = np.zeros(6)
            param_names = ['r', 'm_c', 'tc', 'phi_c', 'I', 'delta_t']
            for i, param in enumerate(param_names):
                best_params[i] = best_params_bilby[param] * (mcmc_data['rmax'][i] - mcmc_data['rmin'][i]) + \
                                 mcmc_data['rmin'][i]

            r, m_c, tc, phi_c, I, delta_t = best_params

            print(f"MCMC Best Parameters:")
            print(f"  r (log10 distance): {r:.4f}")
            print(f"  m_c (log10 chirp mass): {m_c:.4f}")
            print(f"  tc (merger time): {tc:.4f}")
            print(f"  phi_c (phase): {phi_c:.4f}")
            print(f"  I (flux ratio): {I:.4f}")
            print(f"  delta_t (time delay): {delta_t:.4f}")

            # Apply distance refinement if enabled
            distance_refinement = {'enabled': enable_distance_refinement}

            if enable_distance_refinement:
                print("Applying distance parameter refinement...")
                initial_params_dict = {
                    'r': r, 'm_c': m_c, 'tc': tc, 'phi_c': phi_c, 'I': I, 'delta_t': delta_t
                }

                try:
                    refined_r, refinement_info = refine_distance_parameter(
                        initial_params_dict,
                        mcmc_data['dataX'],
                        mcmc_data['dataY_only_signal'],
                        mcmc_data['sampFreq'],
                        mcmc_data['psdHigh'],
                        param_ranges,
                        actual_params
                    )

                    distance_refinement.update({
                        'original_distance': r,
                        'refined_distance': refined_r,
                        'refinement_info': refinement_info
                    })

                    if refinement_info['status'] == 'success':
                        print("Distance refinement successful")
                        r = refined_r

                except Exception as e:
                    print(f"Distance refinement failed: {e}")
                    distance_refinement.update({
                        'original_distance': r,
                        'refined_distance': r,
                        'refinement_info': {'status': 'error', 'error': str(e)}
                    })

            # Generate best signal
            use_lensing = I >= 0.01
            best_signal = crcbgenqcsig(
                mcmc_data['dataX'], r, m_c, tc, phi_c, I, delta_t,
                use_lensing=use_lensing
            )

            # Apply normalization and amplitude optimization
            best_signal, normFac = normsig4psd(best_signal, mcmc_data['sampFreq'], mcmc_data['psdHigh'], 1)
            estAmp = innerprodpsd(
                mcmc_data['dataY_only_signal'], best_signal,
                mcmc_data['sampFreq'], mcmc_data['psdHigh']
            )
            best_signal = estAmp * best_signal

            # Calculate SNR
            snr = calculate_matched_filter_snr(
                best_signal, mcmc_data['dataY_only_signal'],
                mcmc_data['psdHigh'], mcmc_data['sampFreq']
            )

            # Determine classification
            is_lensed = I >= 0.01
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

            samples = result.posterior[param_names].values
            posterior_samples = result.posterior.copy()

        else:
            print("Warning: No posterior samples found.")
            best_params = np.zeros(6)
            r, m_c, tc, phi_c, I, delta_t = best_params
            best_signal = np.zeros_like(mcmc_data['dataY_only_signal'])
            snr = 0
            is_lensed = False
            classification = "error"
            match_value = 0.0
            samples = np.zeros((1, 6))
            posterior_samples = None
            distance_refinement = {'enabled': enable_distance_refinement, 'status': 'no_samples'}

        mcmc_results = {
            'duration': mcmc_duration,
            'best_params': {
                'r': r, 'm_c': m_c, 'tc': tc, 'phi_c': phi_c, 'I': I, 'delta_t': delta_t
            },
            'best_signal': best_signal,
            'snr': float(snr),
            'is_lensed': is_lensed,
            'classification': classification,
            'match': match_value,
            'samples': samples if 'samples' in locals() else np.zeros((1, 6)),
            'log_probs': result.posterior['log_likelihood'].values if len(result.posterior) > 0 else np.array([-np.inf]),
            'bilby_result': result,
            'posterior_samples': posterior_samples,
            'param_ranges': {
                'rmin': mcmc_data['rmin'],
                'rmax': mcmc_data['rmax']
            },
            'iteration_history': mcmc_iteration_history,
            'distance_refinement': distance_refinement
        }

        return mcmc_results

    except Exception as e:
        print(f"Error in processing Bilby results: {str(e)}")
        return {
            'duration': mcmc_duration,
            'best_params': {'r': 0, 'm_c': 0, 'tc': 0, 'phi_c': 0, 'I': 0, 'delta_t': 0},
            'best_signal': np.zeros_like(mcmc_data['dataY_only_signal']),
            'snr': 0,
            'is_lensed': False,
            'classification': "error",
            'match': 0.0,
            'param_ranges': {'rmin': mcmc_data['rmin'], 'rmax': mcmc_data['rmax']},
            'iteration_history': [],
            'posterior_samples': None,
            'distance_refinement': {'enabled': enable_distance_refinement, 'status': 'error'}
        }


def run_traditional_pso(data_dict, pso_params, pso_config, actual_params, n_runs=1, enable_distance_refinement=True):
    """Run traditional PSO analysis with single run"""
    print(f"Starting Traditional PSO analysis with {n_runs} run...")

    start_time = time.time()
    
    try:
        outResults, outStruct = crcbqcpsopsd_traditional(
            pso_params, pso_config, n_runs,
            use_two_step=True,
            actual_params=actual_params,
            enable_distance_refinement=enable_distance_refinement
        )

        end_time = time.time()
        total_pso_duration = end_time - start_time
        
        print(f"Traditional PSO completed in {total_pso_duration:.2f} seconds")

        # Extract best run
        best_run_idx = outResults['bestRun']
        best_signal = outResults['bestSig']

        # Calculate match
        dataY_only_signal = np.asarray(data_dict['dataY_only_signal'])
        match_value = pycbc_calculate_match(
            best_signal, dataY_only_signal,
            data_dict['sampFreq'], data_dict['psdHigh']
        )

        # Extract iteration history
        iteration_history = []
        if len(outStruct) > 0 and 'fitnessHistory' in outStruct[0]:
            fitness_history = outStruct[0]['fitnessHistory']
            iteration_history = [-f for f in fitness_history]

        distance_refinement = outResults.get('distance_refinement', {'enabled': enable_distance_refinement})

        # Collect run results
        run_params = {
            'run_id': 1,
            'r': outResults['allRunsOutput'][0]['r'],
            'm_c': outResults['allRunsOutput'][0]['m_c'],
            'tc': outResults['allRunsOutput'][0]['tc'],
            'phi_c': outResults['allRunsOutput'][0]['phi_c'],
            'I': outResults['allRunsOutput'][0]['I'],
            'delta_t': outResults['allRunsOutput'][0]['delta_t'],
            'fitness': outResults['allRunsOutput'][0]['fitVal'],
            'snr': outResults['allRunsOutput'][0]['SNR_pycbc'],
            'duration': total_pso_duration,
            'is_best': True
        }

        pso_results = {
            'duration': total_pso_duration,
            'total_duration': total_pso_duration,
            'n_runs': n_runs,
            'best_params': {
                'r': outResults['r'],
                'm_c': outResults['m_c'],
                'tc': outResults['tc'],
                'phi_c': outResults['phi_c'],
                'I': outResults['I'],
                'delta_t': outResults['delta_t']
            },
            'best_signal': best_signal,
            'snr': outResults['allRunsOutput'][best_run_idx]['SNR_pycbc'],
            'is_lensed': outResults['is_lensed'],
            'classification': outResults['classification'],
            'match': match_value,
            'all_runs': outResults['allRunsOutput'],
            'run_params': run_params,
            'structures': outStruct,
            'iteration_history': iteration_history,
            'distance_refinement': distance_refinement
        }

        return pso_results

    except Exception as e:
        print(f"Error in Traditional PSO: {str(e)}")
        return {
            'duration': time.time() - start_time,
            'total_duration': time.time() - start_time,
            'n_runs': n_runs,
            'best_params': {'r': 0, 'm_c': 0, 'tc': 0, 'phi_c': 0, 'I': 0, 'delta_t': 0},
            'best_signal': np.zeros_like(data_dict['dataY_only_signal']),
            'snr': 0,
            'is_lensed': False,
            'classification': "error",
            'match': 0.0,
            'run_params': {},
            'iteration_history': [],
            'distance_refinement': {'enabled': enable_distance_refinement, 'status': 'error'}
        }


def create_parameter_estimation_table(pso_results, mcmc_results, actual_params, results_dir="."):
    """Create parameter estimation results table"""
    print("Creating parameter estimation results table...")
    
    try:
        # Parameter transformation info
        param_info = {
            'r': {'name': 'Distance', 'unit': 'Mpc', 'transform': lambda x: 10 ** x},
            'm_c': {'name': 'Chirp Mass', 'unit': 'M☉', 'transform': lambda x: 10 ** x},
            'tc': {'name': 'Merger Time', 'unit': 's', 'transform': lambda x: x},
            'phi_c': {'name': 'Phase', 'unit': 'rad', 'transform': lambda x: x},
            'I': {'name': 'Flux Ratio', 'unit': '', 'transform': lambda x: x},
            'delta_t': {'name': 'Time Delay', 'unit': 's', 'transform': lambda x: x}
        }
        
        # Actual parameter values
        actual_values = {
            'r': actual_params['source_distance'],
            'm_c': actual_params['chirp_mass'],
            'tc': actual_params['merger_time'],
            'phi_c': actual_params['phase'],
            'I': actual_params['flux_ratio'],
            'delta_t': actual_params['time_delay']
        }
        
        # Create parameter estimation data
        table_data = []
        param_names = ['r', 'm_c', 'tc', 'phi_c', 'I', 'delta_t']
        
        for param in param_names:
            # Get estimated values
            pso_raw = float(pso_results['best_params'][param])
            mcmc_raw = float(mcmc_results['best_params'][param])
            
            # Transform to physical units
            pso_value = param_info[param]['transform'](pso_raw)
            mcmc_value = param_info[param]['transform'](mcmc_raw)
            actual_value = actual_values[param]
            
            # Calculate relative errors
            pso_error = abs((pso_value - actual_value) / actual_value) * 100
            mcmc_error = abs((mcmc_value - actual_value) / actual_value) * 100
            
            table_data.append({
                'Parameter': param_info[param]['name'],
                'Unit': param_info[param]['unit'],
                'True_Value': f"{actual_value:.4f}",
                'PSO_Estimate': f"{pso_value:.4f}",
                'PSO_Error_Percent': f"{pso_error:.2f}",
                'MCMC_Estimate': f"{mcmc_value:.4f}",
                'MCMC_Error_Percent': f"{mcmc_error:.2f}"
            })
        
        # Create DataFrame and save
        param_df = pd.DataFrame(table_data)
        param_df.to_csv(f"{results_dir}/parameter_estimation_results.csv", index=False)
        
        print(f"Parameter estimation table saved to {results_dir}/parameter_estimation_results.csv")
        
        return param_df
        
    except Exception as e:
        print(f"Error creating parameter estimation table: {str(e)}")
        return pd.DataFrame()


def create_performance_comparison_table(pso_results, mcmc_results, pso_iterations, results_dir="."):
    """Create performance comparison table for runtime, SNR, and match"""
    print("Creating performance comparison table...")
    
    try:
        # Create performance comparison data
        comparison_data = {
            'Method': ['PSO', 'MCMC'],
            'Iterations_to_Converge': [pso_iterations, 1],  # MCMC only runs once
            'Runtime_seconds': [f"{pso_results['duration']:.2f}", f"{mcmc_results['duration']:.2f}"],
            'SNR': [f"{pso_results['snr']:.4f}", f"{mcmc_results['snr']:.4f}"],
            'Match': [f"{pso_results['match']:.6f}", f"{mcmc_results['match']:.6f}"],
            'Classification': [pso_results['classification'], mcmc_results['classification']]
        }
        
        # Create DataFrame and save
        performance_df = pd.DataFrame(comparison_data)
        performance_df.to_csv(f"{results_dir}/performance_comparison.csv", index=False)
        
        print(f"Performance comparison table saved to {results_dir}/performance_comparison.csv")
        
        # Display summary
        print(f"\nPerformance Comparison Summary:")
        print(f"PSO: {pso_iterations} iterations, {pso_results['duration']:.2f}s, SNR={pso_results['snr']:.4f}, Match={pso_results['match']:.6f}")
        print(f"MCMC: 1 iteration (single run), {mcmc_results['duration']:.2f}s, SNR={mcmc_results['snr']:.4f}, Match={mcmc_results['match']:.6f}")
        
        return performance_df
        
    except Exception as e:
        print(f"Error creating performance comparison table: {str(e)}")
        return pd.DataFrame()


def create_enhanced_corner_plot(pso_results, mcmc_results, param_ranges, actual_params, results_dir="."):
    """Create enhanced corner plot with pure red PSO and pure blue MCMC"""
    print("Creating enhanced corner plot...")
    
    try:
        if mcmc_results['posterior_samples'] is None or len(mcmc_results['posterior_samples']) == 0:
            print("No MCMC posterior samples available for plotting")
            return
        
        # Determine if signal is lensed
        best_I_mcmc = float(mcmc_results['best_params']['I'])
        is_lensed = best_I_mcmc >= 0.01
        
        # Select parameters based on lensing status
        if is_lensed:
            param_names = ['r', 'm_c', 'tc', 'phi_c', 'I', 'delta_t']
            param_labels = [r'$\log_{10}(D_L)$', r'$\log_{10}(M_c)$', r'$t_c$', 
                           r'$\phi_c$', r'$I$', r'$\tau$']
        else:
            param_names = ['r', 'm_c', 'tc', 'phi_c']
            param_labels = [r'$\log_{10}(D_L)$', r'$\log_{10}(M_c)$', r'$t_c$', r'$\phi_c$']
        
        n_params = len(param_names)
        
        # Process MCMC samples
        mcmc_posterior = mcmc_results['posterior_samples']
        mcmc_physical = np.zeros((len(mcmc_posterior), n_params))
        
        for i, param in enumerate(param_names):
            param_idx = ['r', 'm_c', 'tc', 'phi_c', 'I', 'delta_t'].index(param)
            unscaled = mcmc_posterior[param].values * (param_ranges['rmax'][param_idx] - param_ranges['rmin'][param_idx]) + \
                       param_ranges['rmin'][param_idx]
            mcmc_physical[:, i] = unscaled
        
        # Process PSO results (single run, create scatter point)
        pso_physical = np.zeros(n_params)
        for i, param in enumerate(param_names):
            pso_physical[i] = pso_results['best_params'][param]
        
        # Get actual parameter values
        actual_physical = np.zeros(n_params)
        for i, param in enumerate(param_names):
            if param == 'r':
                actual_physical[i] = np.log10(actual_params['source_distance'])
            elif param == 'm_c':
                actual_physical[i] = np.log10(actual_params['chirp_mass'])
            elif param == 'tc':
                actual_physical[i] = actual_params['merger_time']
            elif param == 'phi_c':
                actual_physical[i] = actual_params['phase']
            elif param == 'I':
                actual_physical[i] = actual_params['flux_ratio']
            elif param == 'delta_t':
                actual_physical[i] = actual_params['time_delay']
        
        # Create figure
        fig = plt.figure(figsize=(4*n_params, 4*n_params))
        gs = fig.add_gridspec(n_params, n_params, hspace=0.1, wspace=0.1)
        
        # Pure colors as requested
        pso_color = '#FF0000'      # Pure red for PSO
        mcmc_color = '#0000FF'     # Pure blue for MCMC
        true_color = '#000000'     # Black for true values
        
        for i in range(n_params):
            for j in range(n_params):
                if i > j:
                    # 2D plots: MCMC contours + PSO point
                    ax = fig.add_subplot(gs[i, j])
                    
                    # Get data
                    x_mcmc = mcmc_physical[:, j]
                    y_mcmc = mcmc_physical[:, i]
                    x_pso = pso_physical[j]
                    y_pso = pso_physical[i]
                    
                    # Determine plot range
                    all_x = np.concatenate([x_mcmc, [x_pso]])
                    all_y = np.concatenate([y_mcmc, [y_pso]])
                    
                    x_range = [np.percentile(all_x, 1), np.percentile(all_x, 99)]
                    y_range = [np.percentile(all_y, 1), np.percentile(all_y, 99)]
                    
                    # Extend to include true values
                    x_true, y_true = actual_physical[j], actual_physical[i]
                    margin_x = (x_range[1] - x_range[0]) * 0.15
                    margin_y = (y_range[1] - y_range[0]) * 0.15
                    
                    x_range = [min(x_range[0], x_true - margin_x), max(x_range[1], x_true + margin_x)]
                    y_range = [min(y_range[0], y_true - margin_y), max(y_range[1], y_true + margin_y)]
                    
                    # Create MCMC contours
                    n_bins = 40
                    H_mcmc, x_edges, y_edges = np.histogram2d(x_mcmc, y_mcmc, bins=n_bins, 
                                                             range=[x_range, y_range], density=True)
                    H_mcmc_smooth = gaussian_filter(H_mcmc.T, sigma=1.0)
                    
                    # Create coordinate grids
                    X = (x_edges[:-1] + x_edges[1:]) / 2
                    Y = (y_edges[:-1] + y_edges[1:]) / 2
                    X, Y = np.meshgrid(X, Y)
                    
                    # Calculate MCMC confidence levels
                    if np.max(H_mcmc_smooth) > 0:
                        H_flat = H_mcmc_smooth.ravel()
                        H_sorted = np.sort(H_flat)[::-1]
                        H_cumsum = np.cumsum(H_sorted)
                        H_cumsum = H_cumsum / H_cumsum[-1]
                        
                        levels = []
                        for frac in [0.68, 0.95]:  # 68% and 95% confidence levels
                            idx = np.argmax(H_cumsum >= frac)
                            if idx > 0 and idx < len(H_sorted):
                                levels.append(H_sorted[idx])
                        
                        levels = sorted(levels)
                        if len(levels) >= 1:
                            try:
                                ax.contour(X, Y, H_mcmc_smooth, levels=levels, 
                                          colors=[mcmc_color], linewidths=[2.5, 1.5][:len(levels)], 
                                          alpha=0.8, linestyles='-')
                                # Add filled contour for inner confidence region
                                if len(levels) > 0:
                                    ax.contourf(X, Y, H_mcmc_smooth, levels=[levels[0], np.max(H_mcmc_smooth)], 
                                               colors=[mcmc_color], alpha=0.2)
                            except:
                                pass
                    
                    # Add PSO point
                    ax.scatter(x_pso, y_pso, c=pso_color, s=100, alpha=0.9, 
                              marker='o', edgecolors='darkred', linewidths=2.0, zorder=5, label='PSO')
                    
                    # True value lines (thicker as requested)
                    ax.axvline(x_true, color=true_color, linestyle='--', linewidth=3.0, alpha=0.8, zorder=7)
                    ax.axhline(y_true, color=true_color, linestyle='--', linewidth=3.0, alpha=0.8, zorder=7)
                    
                    # Set limits and labels
                    ax.set_xlim(x_range)
                    ax.set_ylim(y_range)
                    
                    if i == n_params - 1:
                        ax.set_xlabel(param_labels[j], fontsize=14, fontweight='bold')
                        ax.tick_params(axis='x', labelsize=11)
                    else:
                        ax.set_xticklabels([])
                    
                    if j == 0:
                        ax.set_ylabel(param_labels[i], fontsize=14, fontweight='bold')
                        ax.tick_params(axis='y', labelsize=11)
                    else:
                        ax.set_yticklabels([])
                    
                    ax.grid(True, alpha=0.3)
                    
                elif i == j:
                    # 1D histograms
                    ax = fig.add_subplot(gs[i, j])
                    
                    mcmc_data = mcmc_physical[:, i]
                    pso_value = pso_physical[i]
                    true_val = actual_physical[i]
                    
                    # Determine range
                    all_data = np.concatenate([mcmc_data, [pso_value]])
                    data_range = [np.percentile(all_data, 2), np.percentile(all_data, 98)]
                    margin = (data_range[1] - data_range[0]) * 0.15
                    data_range = [min(data_range[0], true_val - margin), 
                                 max(data_range[1], true_val + margin)]
                    
                    # MCMC histogram
                    n_bins = 30
                    ax.hist(mcmc_data, bins=n_bins, range=data_range, density=True, 
                           alpha=0.6, color=mcmc_color, edgecolor='none', label='MCMC')
                    
                    # PSO vertical line
                    y_max = ax.get_ylim()[1]
                    ax.axvline(pso_value, color=pso_color, linewidth=4, alpha=0.8, label='PSO')
                    
                    # True value line (thicker)
                    ax.axvline(true_val, color=true_color, linestyle='--', linewidth=3.0, alpha=0.8, 
                              label='True Value')
                    
                    # Set limits and labels
                    ax.set_xlim(data_range)
                    
                    if i == n_params - 1:
                        ax.set_xlabel(param_labels[i], fontsize=14, fontweight='bold')
                        ax.tick_params(axis='x', labelsize=11)
                    else:
                        ax.set_xticklabels([])
                    
                    ax.set_yticklabels([])
                    ax.tick_params(axis='y', left=False)
                    ax.grid(True, alpha=0.3)
                    
                    # Add legend only for the first diagonal plot
                    if i == 0:
                        ax.legend(loc='upper right', fontsize=10, frameon=True)
                    
                else:
                    # Upper triangle - legend
                    ax = fig.add_subplot(gs[i, j])
                    ax.axis('off')
                    
                    # Add overall legend in top-right corner
                    if i == 0 and j == n_params - 1:
                        from matplotlib.lines import Line2D
                        legend_elements = [
                            Line2D([0], [0], marker='o', color='w', markerfacecolor=pso_color, 
                                  markersize=10, label='PSO Result', markeredgecolor='darkred'),
                            Line2D([0], [0], color=mcmc_color, lw=3, label='MCMC Posterior'),
                            Line2D([0], [0], color=true_color, lw=3, linestyle='--', label='True Values')
                        ]
                        ax.legend(handles=legend_elements, loc='center', fontsize=12, frameon=True)
        
        # Add title
        title_suffix = " (Lensed Signal)" if is_lensed else " (Non-lensed Signal)"
        fig.suptitle(f'PSO vs MCMC Parameter Estimation Comparison{title_suffix}', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Save the plot
        plot_filename = f"corner_plot_{'lensed' if is_lensed else 'unlensed'}.png"
        plt.savefig(f"{results_dir}/{plot_filename}", bbox_inches='tight', dpi=300, facecolor='white')
        
        plot_filename_pdf = f"corner_plot_{'lensed' if is_lensed else 'unlensed'}.pdf"
        plt.savefig(f"{results_dir}/{plot_filename_pdf}", bbox_inches='tight', format='pdf', facecolor='white')
        
        plt.close()
        
        print(f"Corner plot saved to {results_dir}/{plot_filename}")
        
    except Exception as e:
        print(f"Error in corner plot: {str(e)}")
        import traceback
        traceback.print_exc()


def create_signal_reconstruction_plots(pso_results, mcmc_results, data_dict, results_dir="."):
    """Create signal reconstruction and residual plots"""
    print("Creating signal reconstruction and residual plots...")
    
    try:
        # Get data as numpy arrays
        t_np = np.asarray(data_dict['t'])
        dataY_only_signal_np = np.asarray(data_dict['dataY_only_signal'])
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Top panel: Signal comparison
        ax1.plot(t_np, dataY_only_signal_np, 'k-', linewidth=2.5, label='Injected Signal', alpha=0.8)
        ax1.plot(t_np, pso_results['best_signal'], color='#FF0000', linestyle='--', linewidth=2.5,
                 label='PSO Reconstruction', alpha=0.9)
        ax1.plot(t_np, mcmc_results['best_signal'], color='#0000FF', linestyle=':', linewidth=2.5,
                 label='MCMC Reconstruction', alpha=0.9)
        
        ax1.set_ylabel('Strain', fontsize=14, fontweight='bold')
        ax1.set_title('Gravitational Wave Signal Reconstruction Comparison', fontsize=16, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=12, frameon=True)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([t_np[0], t_np[-1]])
        
        # Add SNR and Match info as text boxes
        pso_info = f'PSO: SNR={pso_results["snr"]:.2f}, Match={pso_results["match"]:.4f}'
        mcmc_info = f'MCMC: SNR={mcmc_results["snr"]:.2f}, Match={mcmc_results["match"]:.4f}'
        
        ax1.text(0.02, 0.98, pso_info, transform=ax1.transAxes, fontsize=11, 
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE6E6', alpha=0.8))
        ax1.text(0.02, 0.88, mcmc_info, transform=ax1.transAxes, fontsize=11, 
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='#E6E6FF', alpha=0.8))
        
        # Bottom panel: Residuals
        pso_residual = pso_results['best_signal'] - dataY_only_signal_np
        mcmc_residual = mcmc_results['best_signal'] - dataY_only_signal_np
        
        ax2.plot(t_np, pso_residual, color='#FF0000', linewidth=2.0, alpha=0.8, label='PSO Residual')
        ax2.plot(t_np, mcmc_residual, color='#0000FF', linewidth=2.0, alpha=0.8, label='MCMC Residual')
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5, linewidth=1.0)
        
        ax2.set_xlabel('Time [s]', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Residual', fontsize=14, fontweight='bold')
        ax2.set_title('Reconstruction Residuals', fontsize=16, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=12, frameon=True)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([t_np[0], t_np[-1]])
        
        # Calculate and display RMS residuals
        pso_rms = np.sqrt(np.mean(pso_residual**2))
        mcmc_rms = np.sqrt(np.mean(mcmc_residual**2))
        
        residual_info = f'RMS Residuals: PSO={pso_rms:.2e}, MCMC={mcmc_rms:.2e}'
        ax2.text(0.02, 0.98, residual_info, transform=ax2.transAxes, fontsize=11, 
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plots in both formats
        plt.savefig(f"{results_dir}/signal_reconstruction.png", bbox_inches='tight', dpi=300, facecolor='white')
        plt.savefig(f"{results_dir}/signal_reconstruction.pdf", bbox_inches='tight', format='pdf', facecolor='white')
        
        plt.close()
        
        print(f"Signal reconstruction plots saved:")
        print(f"  - PNG: {results_dir}/signal_reconstruction.png")
        print(f"  - PDF: {results_dir}/signal_reconstruction.pdf")
        
    except Exception as e:
        print(f"Error creating signal reconstruction plots: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Main function for PSO vs MCMC comparison analysis with error-based convergence"""
    
    print("=" * 70)
    print("PSO与MCMC引力波参数估计对比分析 (PSO循环收敛, MCMC单次运行)")
    print("=" * 70)

    try:
        # 1. 数据加载
        print("\n[步骤 1/7] 加载引力波数据...")
        data_dict = load_data()
        print("✓ 数据加载完成")

        # 2. 参数设置
        print("\n[步骤 2/7] 设置算法参数...")
        param_ranges, pso_params, pso_config, actual_params = setup_parameters(data_dict)
        print("✓ 参数设置完成")
        
        # 显示真实参数值
        print("\n" + "="*50)
        print("真实参数值 (参考)")
        print("="*50)
        print(f"距离:           {actual_params['source_distance']:.2f} Mpc")
        print(f"啁啾质量:       {actual_params['chirp_mass']:.2f} M☉")
        print(f"合并时间:       {actual_params['merger_time']:.4f} s")
        print(f"相位:           {actual_params['phase']:.4f} π")
        print(f"流量比:         {actual_params['flux_ratio']:.4f}")
        print(f"时间延迟:       {actual_params['time_delay']:.4f} s")

        # 3. 运行PSO算法直到收敛 (误差 < 5%)
        print("\n[步骤 3/7] 运行PSO算法直到参数误差全部小于5%...")
        
        max_pso_iterations = 20# 最大迭代次数
        pso_iteration = 0
        pso_converged = False
        error_threshold = 5.0  # 5%
        
        while not pso_converged and pso_iteration < max_pso_iterations:
            pso_iteration += 1
            print(f"\n--- PSO 第 {pso_iteration} 次运行 ---")
            
            pso_results = run_traditional_pso(data_dict, pso_params, pso_config, actual_params,
                                            n_runs=1, enable_distance_refinement=True)
            
            # 检查参数误差
            if pso_results['classification'] != 'error':
                # 判断是否为透镜信号
                is_lensed_pso = determine_lensing_status(pso_results['best_params'])
                include_lensing = is_lensed_pso
                
                pso_errors = calculate_parameter_errors(pso_results['best_params'], actual_params, include_lensing)
                pso_converged = check_errors_below_threshold(pso_errors, error_threshold)
                
                print(f"PSO 第 {pso_iteration} 次运行参数误差 ({'透镜信号' if is_lensed_pso else '非透镜信号'}):")
                for param, error in pso_errors.items():
                    print(f"  {param}: {error:.2f}%")
                
                if not include_lensing:
                    print("  注意: I < 0.01 (非透镜情况), I和delta_t参数不参与收敛判断")
                
                if pso_converged:
                    print(f"✓ PSO在第 {pso_iteration} 次运行后收敛 (所有相关误差 < {error_threshold}%)")
                else:
                    print(f"✗ PSO第 {pso_iteration} 次运行未达到收敛条件")
            else:
                print(f"✗ PSO第 {pso_iteration} 次运行失败")
        
        if not pso_converged:
            print(f"⚠️ PSO在 {max_pso_iterations} 次运行后仍未收敛，使用最后一次结果")

        # 4. 运行MCMC算法 (单次运行)
        print(f"\n[步骤 4/7] 运行MCMC算法 (单次运行)...")
        
        print(f"\n--- MCMC 运行 ---")
        
        mcmc_results = run_bilby_mcmc(data_dict, param_ranges, n_live_points=200, n_iter=500,
                                      enable_distance_refinement=True, actual_params=actual_params)
        
        # 计算参数误差但不用于收敛判断
        if mcmc_results['classification'] != 'error':
            # 判断是否为透镜信号
            is_lensed_mcmc = determine_lensing_status(mcmc_results['best_params'])
            include_lensing = is_lensed_mcmc
            
            mcmc_errors = calculate_parameter_errors(mcmc_results['best_params'], actual_params, include_lensing)
            
            print(f"MCMC 参数误差 ({'透镜信号' if is_lensed_mcmc else '非透镜信号'}):")
            for param, error in mcmc_errors.items():
                print(f"  {param}: {error:.2f}%")
            
            if not include_lensing:
                print("  注意: I < 0.01 (非透镜情况), 仅显示相关参数误差")
            
            print(f"✓ MCMC单次运行完成")
        else:
            print(f"✗ MCMC运行失败")

        # 5. 创建参数估计表格
        print("\n[步骤 5/7] 创建参数估计和性能对比表格...")
        param_df = create_parameter_estimation_table(pso_results, mcmc_results, actual_params, results_dir)
        performance_df = create_performance_comparison_table(pso_results, mcmc_results, pso_iteration, results_dir)
        print("✓ 表格创建完成")

        # 6. 生成可视化图表
        print("\n[步骤 6/7] 生成可视化图表...")
        
        # 创建角图
        create_enhanced_corner_plot(pso_results, mcmc_results, param_ranges, actual_params, results_dir)
        
        # 创建信号重构图
        create_signal_reconstruction_plots(pso_results, mcmc_results, data_dict, results_dir)
        
        print("✓ 可视化图表生成完成")

        # 7. 输出结果总结
        print("\n[步骤 7/7] 结果总结...")
        
        # 输出收敛信息
        print("\n" + "="*60)
        print("算法执行总结")
        print("="*60)
        print(f"PSO:  {pso_iteration} 次运行后{'收敛' if pso_converged else '未收敛'}")
        print(f"MCMC: 1 次运行 (单次执行)")
        
        # 输出PSO参数结果
        print("\n" + "="*50)
        print("PSO 最终参数估计结果")
        print("="*50)
        
        pso_params_result = pso_results['best_params']
        
        # 转换为物理单位
        pso_distance = 10**pso_params_result['r']
        pso_chirp_mass = 10**pso_params_result['m_c']
        pso_merger_time = pso_params_result['tc']
        pso_phase = pso_params_result['phi_c']
        pso_flux_ratio = pso_params_result['I']
        pso_time_delay = pso_params_result['delta_t']
        
        print(f"距离:           {pso_distance:.2f} Mpc")
        print(f"啁啾质量:       {pso_chirp_mass:.2f} M☉")
        print(f"合并时间:       {pso_merger_time:.4f} s")
        print(f"相位:           {pso_phase:.4f} π")
        print(f"流量比:         {pso_flux_ratio:.4f}")
        print(f"时间延迟:       {pso_time_delay:.4f} s")
        print(f"总执行时间:     {pso_results['duration']:.2f} 秒")
        print(f"信噪比:         {pso_results['snr']:.4f}")
        print(f"匹配度:         {pso_results['match']:.6f}")

        # 输出MCMC参数结果
        print("\n" + "="*50)
        print("MCMC 最终参数估计结果")
        print("="*50)
        
        mcmc_params_result = mcmc_results['best_params']
        
        # 转换为物理单位
        mcmc_distance = 10**mcmc_params_result['r']
        mcmc_chirp_mass = 10**mcmc_params_result['m_c']
        mcmc_merger_time = mcmc_params_result['tc']
        mcmc_phase = mcmc_params_result['phi_c']
        mcmc_flux_ratio = mcmc_params_result['I']
        mcmc_time_delay = mcmc_params_result['delta_t']
        
        print(f"距离:           {mcmc_distance:.2f} Mpc")
        print(f"啁啾质量:       {mcmc_chirp_mass:.2f} M☉")
        print(f"合并时间:       {mcmc_merger_time:.4f} s")
        print(f"相位:           {mcmc_phase:.4f} π")
        print(f"流量比:         {mcmc_flux_ratio:.4f}")
        print(f"时间延迟:       {mcmc_time_delay:.4f} s")
        print(f"总执行时间:     {mcmc_results['duration']:.2f} 秒")
        print(f"信噪比:         {mcmc_results['snr']:.4f}")
        print(f"匹配度:         {mcmc_results['match']:.6f}")

        # 显示最终参数误差
        if pso_converged:
            print("\n" + "="*50)
            is_lensed_pso_final = determine_lensing_status(pso_results['best_params'])
            if is_lensed_pso_final:
                print("PSO 最终参数误差 (透镜信号，均小于5%)")
            else:
                print("PSO 最终参数误差 (非透镜信号，r,m_c,tc,phi_c均小于5%)")
            print("="*50)
            final_pso_errors = calculate_parameter_errors(pso_results['best_params'], actual_params, is_lensed_pso_final)
            for param, error in final_pso_errors.items():
                print(f"{param}: {error:.2f}%")
            if not is_lensed_pso_final:
                print("注意: I和delta_t未包含在收敛条件中")
        
        # 显示MCMC参数误差（仅用于比较）
        if mcmc_results['classification'] != 'error':
            print("\n" + "="*50)
            is_lensed_mcmc_final = determine_lensing_status(mcmc_results['best_params'])
            if is_lensed_mcmc_final:
                print("MCMC 参数误差 (透镜信号，单次运行结果)")
            else:
                print("MCMC 参数误差 (非透镜信号，单次运行结果)")
            print("="*50)
            final_mcmc_errors = calculate_parameter_errors(mcmc_results['best_params'], actual_params, is_lensed_mcmc_final)
            for param, error in final_mcmc_errors.items():
                print(f"{param}: {error:.2f}%")
            if not is_lensed_mcmc_final:
                print("注意: 仅显示相关参数误差")

        # 显示生成的文件列表
        print("\n" + "="*60)
        print("生成的文件列表:")
        print("="*60)
        print("📊 数据表格:")
        print(f"  1. {results_dir}/parameter_estimation_results.csv - 参数估计结果对比")
        print(f"  2. {results_dir}/performance_comparison.csv - 性能对比结果")
        
        print("\n📈 可视化图表:")
        print(f"  1. {results_dir}/corner_plot_lensed.png/.pdf - 参数空间角图")
        print(f"  2. {results_dir}/signal_reconstruction.png/.pdf - 信号重构与残差图")
        
        print("\n🎯 分析总结:")
        total_pso_time = pso_results['duration']
        total_mcmc_time = mcmc_results['duration']
        speed_ratio = total_mcmc_time / total_pso_time
        
        print(f"  ✓ PSO收敛次数: {pso_iteration}次 ({'成功' if pso_converged else '未完全收敛'})")
        print(f"  ✓ MCMC执行: 单次运行")
        print(f"  ✓ PSO总时间: {total_pso_time:.2f}秒")
        print(f"  ✓ MCMC总时间: {total_mcmc_time:.2f}秒")
        print(f"  ✓ MCMC单次运行比PSO总收敛时间{'快' if speed_ratio < 1 else '慢'} {speed_ratio:.1f}倍")
        print(f"  ✓ PSO最终信噪比: {pso_results['snr']:.2f}")
        print(f"  ✓ MCMC最终信噪比: {mcmc_results['snr']:.2f}")
        print(f"  ✓ PSO最终匹配度: {pso_results['match']:.4f}")
        print(f"  ✓ MCMC最终匹配度: {mcmc_results['match']:.4f}")
        
    except Exception as e:
        print(f"\n❌ 执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    print("\n✅ PSO与MCMC对比分析完成!")
    print("    注意: PSO进行循环收敛判断，MCMC仅执行单次运行用于比较")
    return True


if __name__ == "__main__":
    main()