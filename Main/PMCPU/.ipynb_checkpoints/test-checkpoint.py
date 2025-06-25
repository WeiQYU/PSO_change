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

# Import the PSO implementation
from testPSO import (
    crcbqcpsopsd, crcbgenqcsig, normsig4psd, innerprodpsd,
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
results_dir = "L_Result_circulate"
os.makedirs(results_dir, exist_ok=True)

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
    analysisData = scio.loadmat('../generate_data/data.mat')

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
        'flux_ratio': 0.1429,
        'time_delay': 1.4781,
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

    # Set up PSO configuration
    pso_config = {
        'popsize': 50, 
        'maxSteps': 2000,  
        'c1': 2.0,
        'c2': 2.0,
        'w_start': 0.9,
        'w_end': 0.5,
        'max_velocity': 0.4,
        'nbrhdSz': 6,
        'disable_early_stop': False,
        'track_particles': True  # Enable A parameter tracking
    }

    return param_ranges, pso_params, pso_config, actual_params


class GWLikelihood(Likelihood):
    """Gravitational wave likelihood function for Bilby"""

    def __init__(self, data_dict):
        """Initialize the likelihood with data"""
        super().__init__(parameters={
            'r': None, 'm_c': None, 'tc': None,
            'phi_c': None, 'A': None, 'delta_t': None
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

            # Parameter range checks
            if r < -2 or r > 4 or m_c < 0 or m_c > 2 or tc < 0.1 or tc > 8.0:
                return -np.inf

            # Determine lensing usage - MCMC threshold
            use_lensing = A >= 0.01

            # Generate signal
            signal = crcbgenqcsig(
                self.dataX_np, r, m_c, tc, phi_c, A, delta_t,
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
            delta_t_calc = 1.0 / self.data_dict['sampFreq']
            ts_signal = TimeSeries(signal_final, delta_t=delta_t_calc)
            ts_data = TimeSeries(self.dataY_only_signal_np, delta_t=delta_t_calc)

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

            delta_f = 1.0 / (len(signal_final) * delta_t_calc)
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
        priors['A'] = Uniform(minimum=0, maximum=1, name='A')
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

        # Extract iteration history for A parameter tracking
        mcmc_iteration_history = []
        mcmc_a_samples = []
        if hasattr(result, 'sampler') and hasattr(result.sampler, 'results'):
            sampler_results = result.sampler.results
            if hasattr(sampler_results, 'logz'):
                mcmc_iteration_history = sampler_results.logz
            elif len(result.posterior) > 0:
                mcmc_iteration_history = result.posterior['log_likelihood'].values
                # Extract A parameter samples for tracking
                mcmc_a_samples = result.posterior['A'].values

    except Exception as e:
        print(f"Error in Bilby MCMC: {str(e)}")
        end_time = time.time()
        return {
            'duration': end_time - start_time,
            'best_params': {'r': 0, 'm_c': 0, 'tc': 0, 'phi_c': 0, 'A': 0, 'delta_t': 0},
            'best_signal': np.zeros_like(data_dict['dataY_only_signal']),
            'snr': 0,
            'is_lensed': False,
            'classification': "error",
            'match': 0.0,
            'param_ranges': mcmc_data,
            'iteration_history': [],
            'posterior_samples': None,
            'a_parameter_samples': [],
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

            # Apply distance refinement if enabled
            distance_refinement = {'enabled': enable_distance_refinement}

            if enable_distance_refinement:
                print("Applying distance parameter refinement...")
                initial_params_dict = {
                    'r': r, 'm_c': m_c, 'tc': tc, 'phi_c': phi_c, 'A': A, 'delta_t': delta_t
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
            use_lensing = A >= 0.01  # MCMC threshold
            best_signal = crcbgenqcsig(
                mcmc_data['dataX'], r, m_c, tc, phi_c, A, delta_t,
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
            is_lensed = A >= 0.01  # MCMC threshold
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
            r, m_c, tc, phi_c, A, delta_t = best_params
            best_signal = np.zeros_like(mcmc_data['dataY_only_signal'])
            snr = 0
            is_lensed = False
            classification = "error"
            match_value = 0.0
            samples = np.zeros((1, 6))
            posterior_samples = None
            mcmc_a_samples = []
            distance_refinement = {'enabled': enable_distance_refinement, 'status': 'no_samples'}

        mcmc_results = {
            'duration': mcmc_duration,
            'best_params': {
                'r': r, 'm_c': m_c, 'tc': tc, 'phi_c': phi_c, 'A': A, 'delta_t': delta_t
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
            'a_parameter_samples': mcmc_a_samples,
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
            'best_params': {'r': 0, 'm_c': 0, 'tc': 0, 'phi_c': 0, 'A': 0, 'delta_t': 0},
            'best_signal': np.zeros_like(mcmc_data['dataY_only_signal']),
            'snr': 0,
            'is_lensed': False,
            'classification': "error",
            'match': 0.0,
            'param_ranges': {'rmin': mcmc_data['rmin'], 'rmax': mcmc_data['rmax']},
            'iteration_history': [],
            'posterior_samples': None,
            'a_parameter_samples': [],
            'distance_refinement': {'enabled': enable_distance_refinement, 'status': 'error'}
        }


def run_pso(data_dict, pso_params, pso_config, actual_params, n_runs=1, enable_distance_refinement=True):
    """Run PSO analysis with A parameter tracking"""
    print("Starting PSO analysis...")

    start_time = time.time()

    try:
        outResults, outStruct = crcbqcpsopsd(
            pso_params, pso_config, n_runs,
            use_two_step=True,
            actual_params=actual_params,
            enable_distance_refinement=enable_distance_refinement
        )

        end_time = time.time()
        pso_duration = end_time - start_time
        print(f"PSO completed in {pso_duration:.2f} seconds")

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
        pso_iteration_history = []
        pso_a_evolution = []
        if 'fitnessHistory' in outStruct[best_run_idx]:
            fitness_history = outStruct[best_run_idx]['fitnessHistory']
            pso_iteration_history = [-f for f in fitness_history]
        
        # Extract A parameter evolution
        if 'aParameterEvolution' in outStruct[best_run_idx]:
            pso_a_evolution = outStruct[best_run_idx]['aParameterEvolution']

        distance_refinement = outResults.get('distance_refinement', {'enabled': enable_distance_refinement})

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
            'best_signal': best_signal,
            'snr': outResults['allRunsOutput'][best_run_idx]['SNR_pycbc'],
            'is_lensed': outResults['is_lensed'],
            'classification': outResults['classification'],
            'match': match_value,
            'all_runs': outResults['allRunsOutput'],
            'structures': outStruct,
            'iteration_history': pso_iteration_history,
            'a_parameter_evolution': pso_a_evolution,
            'distance_refinement': distance_refinement
        }

        return pso_results

    except Exception as e:
        print(f"Error in PSO: {str(e)}")
        return {
            'duration': time.time() - start_time,
            'best_params': {'r': 0, 'm_c': 0, 'tc': 0, 'phi_c': 0, 'A': 0, 'delta_t': 0},
            'best_signal': np.zeros_like(data_dict['dataY_only_signal']),
            'snr': 0,
            'is_lensed': False,
            'classification': "error",
            'match': 0.0,
            'iteration_history': [],
            'a_parameter_evolution': [],
            'distance_refinement': {'enabled': enable_distance_refinement, 'status': 'error'}
        }


def plot_a_parameter_analysis(pso_results, mcmc_results, actual_params, param_ranges):
    """Create comprehensive visualization of A parameter search and distribution"""
    print("\nGenerating A parameter analysis plots...")
    
    try:
        # Get true A value
        true_A = actual_params['flux_ratio']
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
        
        # Colors and styles
        pso_color = '#d62728'
        mcmc_color = '#1f77b4'
        true_color = '#2ca02c'
        threshold_color_pso = '#ff7f0e'
        threshold_color_mcmc = '#9467bd'
        
        # ======================
        # Top Left: PSO A Parameter Evolution
        # ======================
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Extract A parameter evolution from PSO
        pso_a_evolution = pso_results.get('a_parameter_evolution', [])
        
        if pso_a_evolution and len(pso_a_evolution) > 0:
            # Plot particle swarm evolution
            for i, a_values in enumerate(pso_a_evolution[::5]):  # Plot every 5th iteration
                if len(a_values) > 0:
                    # Convert from [0,1] to actual range
                    a_actual = a_values * (param_ranges['rmax'][4] - param_ranges['rmin'][4]) + param_ranges['rmin'][4]
                    ax1.scatter([i*5] * len(a_actual), a_actual, 
                               c=pso_color, alpha=0.3, s=8, rasterized=True)
            
            # Plot best particle evolution
            best_evolution = []
            for a_values in pso_a_evolution:
                if len(a_values) > 0:
                    a_actual = a_values * (param_ranges['rmax'][4] - param_ranges['rmin'][4]) + param_ranges['rmin'][4]
                    best_evolution.append(np.min(a_actual))
                    
            if best_evolution:
                iterations = range(len(best_evolution))
                ax1.plot(iterations, best_evolution, color=pso_color, linewidth=3, 
                        label=f'PSO Best A = {pso_results["best_params"]["A"]:.4f}', alpha=0.9)
        else:
            # Fallback: show final result only
            final_a = pso_results['best_params']['A']
            ax1.axhline(y=final_a, color=pso_color, linewidth=3, 
                       label=f'PSO Final A = {final_a:.4f}')
        
        # Add threshold lines
        ax1.axhline(y=0.01, color=threshold_color_pso, linestyle='--', linewidth=2, 
                   label='PSO Lensing Threshold (A = 0.01)', alpha=0.8)
        ax1.axhline(y=true_A, color=true_color, linestyle='-', linewidth=2.5, 
                   label=f'True A = {true_A:.4f}', alpha=0.9)
        
        ax1.set_xlabel('PSO Iteration', fontsize=12, fontweight='bold')
        ax1.set_ylabel('A Parameter Value', fontsize=12, fontweight='bold')
        ax1.set_title('PSO: A Parameter Evolution During Optimization', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10, loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # ======================
        # Top Right: MCMC A Parameter Samples
        # ======================
        ax2 = fig.add_subplot(gs[0, 1])
        
        if mcmc_results['posterior_samples'] is not None and len(mcmc_results['posterior_samples']) > 0:
            # Extract A parameter samples
            mcmc_a_samples = mcmc_results['posterior_samples']['A'].values
            
            # Convert from normalized [0,1] to actual range
            mcmc_a_actual = mcmc_a_samples * (param_ranges['rmax'][4] - param_ranges['rmin'][4]) + param_ranges['rmin'][4]
            
            # Plot sampling trace
            sample_indices = range(len(mcmc_a_actual))
            ax2.plot(sample_indices, mcmc_a_actual, color=mcmc_color, alpha=0.7, linewidth=1, 
                    label=f'MCMC Samples (mean = {np.mean(mcmc_a_actual):.4f})')
            
            # Add running mean
            window_size = max(len(mcmc_a_actual) // 50, 10)
            running_mean = pd.Series(mcmc_a_actual).rolling(window=window_size, center=True).mean()
            ax2.plot(sample_indices, running_mean, color='darkred', linewidth=2, 
                    label='Running Mean', alpha=0.8)
        else:
            # Fallback: show final result only
            final_a = mcmc_results['best_params']['A']
            ax2.axhline(y=final_a, color=mcmc_color, linewidth=3, 
                       label=f'MCMC Best A = {final_a:.4f}')
        
        # Add threshold lines
        ax2.axhline(y=0.01, color=threshold_color_mcmc, linestyle='--', linewidth=2, 
                   label='MCMC Lensing Threshold (A = 0.01)', alpha=0.8)
        ax2.axhline(y=true_A, color=true_color, linestyle='-', linewidth=2.5, 
                   label=f'True A = {true_A:.4f}', alpha=0.9)
        
        ax2.set_xlabel('MCMC Sample Number', fontsize=12, fontweight='bold')
        ax2.set_ylabel('A Parameter Value', fontsize=12, fontweight='bold')
        ax2.set_title('MCMC: A Parameter Posterior Sampling', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10, loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # ======================
        # Bottom Left: A Parameter Distribution Comparison
        # ======================
        ax3 = fig.add_subplot(gs[1, :])
        
        # PSO distribution (if we have evolution data)
        if pso_a_evolution and len(pso_a_evolution) > 0:
            # Get final iterations
            final_iterations = pso_a_evolution[-min(50, len(pso_a_evolution)):]
            all_pso_a = []
            for a_values in final_iterations:
                if len(a_values) > 0:
                    a_actual = a_values * (param_ranges['rmax'][4] - param_ranges['rmin'][4]) + param_ranges['rmin'][4]
                    all_pso_a.extend(a_actual)
            
            if all_pso_a:
                ax3.hist(all_pso_a, bins=40, alpha=0.6, color=pso_color, 
                        label=f'PSO Final Iterations (n={len(all_pso_a)})', density=True, 
                        edgecolor='black', linewidth=0.5)
        
        # MCMC distribution
        if mcmc_results['posterior_samples'] is not None:
            mcmc_a_samples = mcmc_results['posterior_samples']['A'].values
            mcmc_a_actual = mcmc_a_samples * (param_ranges['rmax'][4] - param_ranges['rmin'][4]) + param_ranges['rmin'][4]
            
            ax3.hist(mcmc_a_actual, bins=40, alpha=0.6, color=mcmc_color, 
                    label=f'MCMC Posterior (n={len(mcmc_a_actual)})', density=True, 
                    edgecolor='black', linewidth=0.5)
            
            # Add KDE curve for MCMC
            try:
                kde = stats.gaussian_kde(mcmc_a_actual)
                x_kde = np.linspace(0, 1, 200)
                y_kde = kde(x_kde)
                ax3.plot(x_kde, y_kde, color='darkblue', linewidth=2.5, 
                        label='MCMC KDE', alpha=0.8)
            except:
                pass
        
        # Add vertical lines for important values
        ax3.axvline(x=true_A, color=true_color, linestyle='-', linewidth=2.5, 
                   label=f'True A = {true_A:.4f}', alpha=0.9)
        ax3.axvline(x=0.01, color=threshold_color_pso, linestyle='--', linewidth=2, 
                   label='PSO Threshold (0.01)', alpha=0.8)
        ax3.axvline(x=0.01, color=threshold_color_mcmc, linestyle='--', linewidth=2, 
                   label='MCMC Threshold (0.01)', alpha=0.8)
        
        # Add best estimates
        pso_best = pso_results['best_params']['A']
        mcmc_best = mcmc_results['best_params']['A']
        ax3.axvline(x=pso_best, color=pso_color, linestyle=':', linewidth=2, 
                   label=f'PSO Best = {pso_best:.4f}', alpha=0.8)
        ax3.axvline(x=mcmc_best, color=mcmc_color, linestyle=':', linewidth=2, 
                   label=f'MCMC Best = {mcmc_best:.4f}', alpha=0.8)
        
        ax3.set_xlabel('A Parameter Value', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
        ax3.set_title('A Parameter Distribution Comparison', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=9, loc='upper right', ncol=2)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 1)
        
        # ======================
        # Bottom: Algorithm Comparison Statistics
        # ======================
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        # Create comparison table
        comparison_data = []
        
        # PSO statistics
        pso_final_a = pso_results['best_params']['A']
        pso_error = abs(pso_final_a - true_A) / true_A * 100
        pso_threshold = 0.01
        pso_classification = "Lensed" if pso_final_a >= pso_threshold else "Unlensed"
        
        comparison_data.append([
            'PSO', f'{pso_final_a:.4f}', f'{pso_error:.2f}%', 
            f'{pso_threshold:.3f}', pso_classification,
            f'{pso_results["duration"]:.1f}s'
        ])
        
        # MCMC statistics
        mcmc_final_a = mcmc_results['best_params']['A']
        mcmc_error = abs(mcmc_final_a - true_A) / true_A * 100
        mcmc_threshold = 0.01
        mcmc_classification = "Lensed" if mcmc_final_a >= mcmc_threshold else "Unlensed"
        
        mcmc_std = 0.0
        if mcmc_results['posterior_samples'] is not None:
            mcmc_a_samples = mcmc_results['posterior_samples']['A'].values
            mcmc_a_actual = mcmc_a_samples * (param_ranges['rmax'][4] - param_ranges['rmin'][4]) + param_ranges['rmin'][4]
            mcmc_std = np.std(mcmc_a_actual)
        
        comparison_data.append([
            'MCMC', f'{mcmc_final_a:.4f} ± {mcmc_std:.4f}', f'{mcmc_error:.2f}%', 
            f'{mcmc_threshold:.3f}', mcmc_classification,
            f'{mcmc_results["duration"]:.1f}s'
        ])
        
        # True value
        true_classification = "Lensed" if true_A >= 0.01 else "Unlensed"
        comparison_data.append([
            'True', f'{true_A:.4f}', '0.00%', 
            'N/A', true_classification, 'N/A'
        ])
        
        # Create table
        headers = ['Method', 'A Value', 'Error', 'Threshold', 'Classification', 'Time']
        table = ax4.table(cellText=comparison_data, colLabels=headers, 
                         cellLoc='center', loc='center', 
                         bbox=[0.1, 0.3, 0.8, 0.6])
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(1, len(comparison_data) + 1):
            for j in range(len(headers)):
                if i == 1:  # PSO row
                    table[(i, j)].set_facecolor('#FFE6E6')
                elif i == 2:  # MCMC row
                    table[(i, j)].set_facecolor('#E6F0FF')
                else:  # True row
                    table[(i, j)].set_facecolor('#E6FFE6')
        
        ax4.set_title('A Parameter Search Comparison Summary', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Add overall title
        fig.suptitle('Gravitational Wave Lensing Parameter (A) Analysis\nPSO vs MCMC Algorithm Comparison', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save the plot
        plt.savefig(f"{results_dir}/a_parameter_analysis.png", 
                   bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        
        print(f"A parameter analysis plot saved to {results_dir}/a_parameter_analysis.png")
        
        # Create additional detailed statistics
        create_a_parameter_statistics_report(pso_results, mcmc_results, actual_params, param_ranges)
        
    except Exception as e:
        print(f"Error in A parameter analysis: {str(e)}")
        import traceback
        traceback.print_exc()


def create_a_parameter_statistics_report(pso_results, mcmc_results, actual_params, param_ranges):
    """Create detailed statistical report for A parameter analysis"""
    print("Creating detailed A parameter statistics report...")
    
    try:
        true_A = actual_params['flux_ratio']
        
        # Prepare report data
        report_data = {
            'True_A_Value': true_A,
            'PSO_Analysis': {
                'Best_A': pso_results['best_params']['A'],
                'Error_Percent': abs(pso_results['best_params']['A'] - true_A) / true_A * 100,
                'Threshold_Used': 0.01,
                'Classification': pso_results['classification'],
                'Is_Lensed': pso_results['is_lensed'],
                'Duration_seconds': pso_results['duration']
            },
            'MCMC_Analysis': {
                'Best_A': mcmc_results['best_params']['A'],
                'Error_Percent': abs(mcmc_results['best_params']['A'] - true_A) / true_A * 100,
                'Threshold_Used': 0.01,
                'Classification': mcmc_results['classification'],
                'Is_Lensed': mcmc_results['is_lensed'],
                'Duration_seconds': mcmc_results['duration']
            }
        }
        
        # Add MCMC posterior statistics if available
        if mcmc_results['posterior_samples'] is not None:
            mcmc_a_samples = mcmc_results['posterior_samples']['A'].values
            mcmc_a_actual = mcmc_a_samples * (param_ranges['rmax'][4] - param_ranges['rmin'][4]) + param_ranges['rmin'][4]
            
            report_data['MCMC_Analysis'].update({
                'Posterior_Mean': np.mean(mcmc_a_actual),
                'Posterior_Std': np.std(mcmc_a_actual),
                'Posterior_Median': np.median(mcmc_a_actual),
                'Confidence_Interval_68': [np.percentile(mcmc_a_actual, 16), np.percentile(mcmc_a_actual, 84)],
                'Confidence_Interval_95': [np.percentile(mcmc_a_actual, 2.5), np.percentile(mcmc_a_actual, 97.5)],
                'Number_of_Samples': len(mcmc_a_actual)
            })
        
        # Save as JSON
        import json
        with open(f"{results_dir}/a_parameter_statistics.json", 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"A parameter statistics saved to {results_dir}/a_parameter_statistics.json")
        
    except Exception as e:
        print(f"Error creating A parameter statistics report: {str(e)}")


def main():
    """Main execution function"""
    
    print("=" * 60)
    print("引力波参数估计与A参数搜索分析")
    print("=" * 60)

    try:
        # 1. Load data
        print("\n[步骤 1/5] 加载引力波数据...")
        data_dict = load_data()
        print("✓ 数据加载完成")

        # 2. Setup parameters
        print("\n[步骤 2/5] 设置算法参数...")
        param_ranges, pso_params, pso_config, actual_params = setup_parameters(data_dict)
        print("✓ 参数设置完成")
        
        # Display true parameters
        print("\n" + "="*40)
        print("真实参数值 (参考)")
        print("="*40)
        print(f"距离:           {actual_params['source_distance']:.2f} Mpc")
        print(f"啁啾质量:       {actual_params['chirp_mass']:.2f} M☉")
        print(f"合并时间:       {actual_params['merger_time']:.4f} s")
        print(f"相位:           {actual_params['phase']:.4f} π")
        print(f"流量比 (A):     {actual_params['flux_ratio']:.4f}")
        print(f"时间延迟:       {actual_params['time_delay']:.4f} s")

        # 3. Run PSO algorithm
        print("\n[步骤 3/5] 运行PSO算法...")
        pso_results = run_pso(data_dict, pso_params, pso_config, actual_params,
                              n_runs=1, enable_distance_refinement=True)
        print("✓ PSO算法完成")

        # 4. Run MCMC algorithm
        print("\n[步骤 4/5] 运行MCMC算法...")
        mcmc_results = run_bilby_mcmc(data_dict, param_ranges, n_live_points=200, n_iter=500,
                                      enable_distance_refinement=True, actual_params=actual_params)
        print("✓ MCMC算法完成")

        # 5. Generate A parameter analysis
        print("\n[步骤 5/5] 生成A参数专门分析...")
        plot_a_parameter_analysis(pso_results, mcmc_results, actual_params, param_ranges)
        print("✓ A参数分析完成")

        # Output results
        print("\n" + "="*50)
        print("PSO 参数估计结果")
        print("="*50)
        
        pso_params_result = pso_results['best_params']
        pso_distance = 10**pso_params_result['r']
        pso_chirp_mass = 10**pso_params_result['m_c']
        pso_merger_time = pso_params_result['tc']
        pso_phase = pso_params_result['phi_c']
        pso_flux_ratio = pso_params_result['A']
        pso_time_delay = pso_params_result['delta_t']
        
        print(f"距离:           {pso_distance:.2f} Mpc")
        print(f"啁啾质量:       {pso_chirp_mass:.2f} M☉")
        print(f"合并时间:       {pso_merger_time:.4f} s")
        print(f"相位:           {pso_phase:.4f} π")
        print(f"流量比 (A):     {pso_flux_ratio:.4f}")
        print(f"时间延迟:       {pso_time_delay:.4f} s")
        print(f"执行时间:       {pso_results['duration']:.2f} 秒")
        print(f"信噪比:         {pso_results['snr']:.4f}")
        print(f"透镜分类:       {pso_results['classification']}")

        print("\n" + "="*50)
        print("MCMC 参数估计结果")
        print("="*50)
        
        mcmc_params_result = mcmc_results['best_params']
        mcmc_distance = 10**mcmc_params_result['r']
        mcmc_chirp_mass = 10**mcmc_params_result['m_c']
        mcmc_merger_time = mcmc_params_result['tc']
        mcmc_phase = mcmc_params_result['phi_c']
        mcmc_flux_ratio = mcmc_params_result['A']
        mcmc_time_delay = mcmc_params_result['delta_t']
        
        print(f"距离:           {mcmc_distance:.2f} Mpc")
        print(f"啁啾质量:       {mcmc_chirp_mass:.2f} M☉")
        print(f"合并时间:       {mcmc_merger_time:.4f} s")
        print(f"相位:           {mcmc_phase:.4f} π")
        print(f"流量比 (A):     {mcmc_flux_ratio:.4f}")
        print(f"时间延迟:       {mcmc_time_delay:.4f} s")
        print(f"执行时间:       {mcmc_results['duration']:.2f} 秒")
        print(f"信噪比:         {mcmc_results['snr']:.4f}")
        print(f"透镜分类:       {mcmc_results['classification']}")

        # A parameter search analysis summary
        print("\n" + "="*60)
        print("A参数搜索机制分析总结")
        print("="*60)
        
        print("\nPSO算法对A参数的搜索特点：")
        print(f"  - 搜索范围: [0, 1.0]")
        print(f"  - 透镜阈值: A ≥ 0.01")
        print(f"  - 最终估计: {pso_flux_ratio:.6f}")
        print(f"  - 真实值误差: {abs(pso_flux_ratio - actual_params['flux_ratio']) / actual_params['flux_ratio'] * 100:.2f}%")
        print(f"  - 搜索方式: 粒子群体协同搜索")
        
        print("\nMCMC算法对A参数的搜索特点：")
        print(f"  - 搜索范围: [0, 1.0]")
        print(f"  - 透镜阈值: A ≥ 0.01")
        print(f"  - 最终估计: {mcmc_flux_ratio:.6f}")
        print(f"  - 真实值误差: {abs(mcmc_flux_ratio - actual_params['flux_ratio']) / actual_params['flux_ratio'] * 100:.2f}%")
        print(f"  - 搜索方式: 贝叶斯采样")
        
        if mcmc_results['posterior_samples'] is not None:
            mcmc_a_samples = mcmc_results['posterior_samples']['A'].values
            mcmc_a_actual = mcmc_a_samples * (param_ranges['rmax'][4] - param_ranges['rmin'][4]) + param_ranges['rmin'][4]
            print(f"  - 后验统计: 均值={np.mean(mcmc_a_actual):.4f}, 标准差={np.std(mcmc_a_actual):.4f}")
        
        print(f"\n关键差异：")
        print(f"  - PSO收敛速度更快，提供点估计")
        print(f"  - MCMC提供完整的不确定性量化")
        print(f"  - 两种算法使用不同的透镜判断阈值")
        
        print("\n✓ A参数专门分析已完成")
        print("✓ 结果已保存到 L_Result_circulate/ 目录")
        
    except Exception as e:
        print(f"\n❌ 执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    print("\n✅ 参数估计与A参数分析完成!")
    return True


if __name__ == "__main__":
    main()