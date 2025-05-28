import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import time
import pandas as pd
import scipy.constants as const
from tqdm import tqdm
import os
import warnings

# Import the PSO implementation
from PSO_main_demo import (
    crcbqcpsopsd, crcbgenqcsig, normsig4psd, innerprodpsd,
    calculate_matched_filter_snr, pycbc_calculate_match
)

# Import Bilby for MCMC implementation
import bilby
from bilby.core.likelihood import Likelihood
from bilby.core.prior import Uniform, PriorDict

from pycbc.types import FrequencySeries, TimeSeries
from pycbc.filter import match, matched_filter

# Constants
G = const.G  # Gravitational constant, m^3 kg^-1 s^-2
c = const.c  # Speed of light, m/s
M_sun = 1.989e30  # Solar mass, kg
pc = 3.086e16  # Parsec to meters

# Create results directory
results_dir = "PSO_MCMC_unlens_signal_result"
os.makedirs(results_dir, exist_ok=True)


def load_data():
    """Load gravitational wave data for analysis"""
    print("Loading data...")

    # Load noise data
    TrainingData = scio.loadmat('../generate_data/noise.mat')
    analysisData = scio.loadmat('../generate_data/data_without_lens.mat')

    print("Data loaded successfully")

    # Convert data to numpy arrays for CPU processing
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

    # Define actual parameters for validation (ONLY for final performance comparison)
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

    # Set up PSO configuration - Modified to 100 particles and 3000 iterations
    pso_config = {
        'popsize': 100, 
        'maxSteps': 3000,  
        'c1': 2.0,
        'c2': 2.0,
        'w_start': 0.9,
        'w_end': 0.5,
        'max_velocity': 0.4,
        'nbrhdSz': 6,
        'disable_early_stop': True
    }

    return param_ranges, pso_params, pso_config, actual_params


class GWLikelihood(Likelihood):
    """CPU-based Gravitational wave likelihood for Bilby"""

    def __init__(self, data_dict):
        """Initialize the likelihood with data - CPU version"""
        super().__init__(parameters={
            'r': None, 'm_c': None, 'tc': None,
            'phi_c': None, 'A': None, 'delta_t': None
        })
        self.data_dict = data_dict
        
        # Keep data as numpy arrays for CPU computation
        self.dataX_np = np.asarray(data_dict['dataX'])
        self.dataY_only_signal_np = np.asarray(data_dict['dataY_only_signal'])
        self.psd_np = np.asarray(data_dict['psdHigh'])
        
        # Pre-compute some constants
        self.sampFreq = data_dict['sampFreq']
        self.rmin = np.asarray(data_dict['rmin'])
        self.rmax = np.asarray(data_dict['rmax'])
        
        print("CPU-based likelihood initialized")

    def log_likelihood(self):
        """CPU-based log-likelihood function for Bilby"""
        try:
            # Get parameters as numpy arrays
            params = np.array([
                self.parameters['r'],
                self.parameters['m_c'],
                self.parameters['tc'],
                self.parameters['phi_c'],
                self.parameters['A'],
                self.parameters['delta_t']
            ])
            
            # Map from [0,1] to original parameter range
            unscaled_params = params * (self.rmax - self.rmin) + self.rmin
            
            # Convert to float values for signal generation
            r, m_c, tc, phi_c, A, delta_t = unscaled_params
            
            # Determine if we should use lensing based on A parameter
            use_lensing = A >= 0.01
            
            # Generate the signal on CPU
            signal = crcbgenqcsig(
                self.dataX_np, r, m_c, tc, phi_c, A, delta_t,
                use_lensing=use_lensing
            )
            
            # Check if signal generation was successful
            if signal is None or np.isnan(signal).any():
                return -np.inf
            
            # Normalize signal
            signal, _ = normsig4psd(signal, self.sampFreq, self.psd_np, 1)
            
            # Calculate match using PyCBC
            match_value = pycbc_calculate_match(
                signal, self.dataY_only_signal_np, self.sampFreq, self.psd_np
            )
            
            if match_value is None or np.isnan(match_value) or match_value <= 0 or match_value > 1:
                return -np.inf
            
            # Convert match to log-likelihood
            n_samples = len(self.dataY_only_signal_np)
            log_likelihood = 0.5 * n_samples * np.log(max(match_value, 1e-10))
            
            if np.isnan(log_likelihood) or np.isinf(log_likelihood):
                return -np.inf
            
            return log_likelihood
            
        except Exception as e:
            print(f"Likelihood calculation failed: {e}")
            return -np.inf


def run_bilby_mcmc(data_dict, param_ranges, n_live_points=500, n_iter=600):
    """Run CPU-based MCMC analysis using Bilby with parameters matching PSO for fair comparison"""
    print("Starting CPU-based Bilby MCMC analysis...")
    print(f"MCMC Parameters: nlive={n_live_points}, maxiter={n_iter}")
    
    # Create dictionary for MCMC
    mcmc_data = {
        'dataX': data_dict['t'],
        'dataY': data_dict['dataY'],
        'dataY_only_signal': data_dict['dataY_only_signal'],
        'sampFreq': data_dict['sampFreq'],
        'psdHigh': data_dict['psdHigh'],
        'rmin': param_ranges['rmin'],
        'rmax': param_ranges['rmax']
    }
    
    start_time = time.time()
    
    # Create CPU-based likelihood
    likelihood = GWLikelihood(mcmc_data)
    
    # Create priors (uniform in [0,1] for all parameters)
    priors = PriorDict()
    priors['r'] = Uniform(minimum=0, maximum=1, name='r')
    priors['m_c'] = Uniform(minimum=0, maximum=1, name='m_c')
    priors['tc'] = Uniform(minimum=0, maximum=1, name='tc')
    priors['phi_c'] = Uniform(minimum=0, maximum=1, name='phi_c')
    priors['A'] = Uniform(minimum=0, maximum=1, name='A')
    priors['delta_t'] = Uniform(minimum=0, maximum=1, name='delta_t')
    
    try:
        # Run the bilby sampler with parameters matching PSO for fair comparison
        # Generate unique label to avoid checkpoint reuse
        import datetime
        unique_label = f'cpu_gw_analysis_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
        result = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sampler='dynesty',
            nlive=n_live_points,  # Match PSO particle count
            walks=25,
            verbose=True,
            maxiter=n_iter,  # Match PSO iteration count
            outdir=results_dir,
            label=unique_label,  # Use unique label to prevent checkpoint reuse
            resume=False,
            check_point_plot=False,
            bound='multi',
            sample='rwalk',
            check_point=False,  # Disable checkpoints for clean timing
            plot=False,  # Disable default plots
        )
        
    except Exception as e:
        print(f"Error in Bilby MCMC: {str(e)}")
        end_time = time.time()
        return {
            'duration': end_time - start_time,
            'best_params': {'r': 0, 'm_c': 0, 'tc': 0, 'phi_c': 0, 'A': 0, 'delta_t': 0},
            'best_signal': np.zeros_like(data_dict['dataY_only_signal']),
            'snr': 0,
            'is_lensed': False,
            'match': 0.0,
            'param_ranges': mcmc_data
        }
    
    end_time = time.time()
    mcmc_duration = end_time - start_time
    print(f"CPU-based Bilby MCMC completed in {mcmc_duration:.2f} seconds")
    
    try:
        # Get best fit parameters
        if len(result.posterior) > 0:
            best_idx = np.argmax(result.posterior['log_likelihood'].values)
            best_params_bilby = result.posterior.iloc[best_idx]
            
            # Unscale parameters to original range
            best_params = np.zeros(6)
            param_names = ['r', 'm_c', 'tc', 'phi_c', 'A', 'delta_t']
            for i, param in enumerate(param_names):
                best_params[i] = best_params_bilby[param] * (mcmc_data['rmax'][i] - mcmc_data['rmin'][i]) + mcmc_data['rmin'][i]
            
            r, m_c, tc, phi_c, A, delta_t = best_params
            
            # Generate best signal
            use_lensing = A >= 0.01
            best_signal = crcbgenqcsig(
                mcmc_data['dataX'], r, m_c, tc, phi_c, A, delta_t,
                use_lensing=use_lensing
            )
            best_signal, _ = normsig4psd(best_signal, mcmc_data['sampFreq'], mcmc_data['psdHigh'], 1)
            
            # Calculate SNR
            snr = calculate_matched_filter_snr(
                best_signal, mcmc_data['dataY_only_signal'], mcmc_data['psdHigh'], mcmc_data['sampFreq']
            )
            
            # Calculate match
            match_value = pycbc_calculate_match(
                best_signal, mcmc_data['dataY_only_signal'], mcmc_data['sampFreq'], mcmc_data['psdHigh']
            )
            
        else:
            print("Warning: No posterior samples found. Using default values.")
            best_params = np.zeros(6)
            r, m_c, tc, phi_c, A, delta_t = best_params
            best_signal = np.zeros_like(mcmc_data['dataY_only_signal'])
            snr = 0
            match_value = 0.0
        
        # Prepare results
        mcmc_results = {
            'duration': mcmc_duration,
            'best_params': {
                'r': r, 'm_c': m_c, 'tc': tc, 'phi_c': phi_c, 'A': A, 'delta_t': delta_t
            },
            'best_signal': best_signal,
            'snr': float(snr),
            'is_lensed': A >= 0.01,
            'match': match_value,
            'param_ranges': {'rmin': mcmc_data['rmin'], 'rmax': mcmc_data['rmax']}
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
            'match': 0.0,
            'param_ranges': {'rmin': mcmc_data['rmin'], 'rmax': mcmc_data['rmax']}
        }


def run_pso(data_dict, pso_params, pso_config, actual_params, n_runs=8):
    """Run PSO analysis"""
    print("Starting PSO analysis...")
    print(f"PSO Parameters: popsize={pso_config['popsize']}, maxSteps={pso_config['maxSteps']}")
    
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
        best_signal = outResults['bestSig']
        
        # Calculate match
        match_value = pycbc_calculate_match(
            best_signal, data_dict['dataY_only_signal'], data_dict['sampFreq'], data_dict['psdHigh']
        )
        
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
            'best_signal': best_signal,
            'snr': outResults['allRunsOutput'][best_run_idx]['SNR_pycbc'],
            'is_lensed': outResults['is_lensed'],
            'match': match_value,
            'all_runs': outResults['allRunsOutput'],
            'structures': outStruct
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
            'match': 0.0
        }


def generate_pso_all_runs_csv(pso_results, data_dict, actual_params, param_ranges):
    """Generate CSV with detailed results from all 8 PSO runs"""
    print("Generating CSV for all PSO runs...")
    
    try:
        # Prepare data for all PSO runs
        csv_data = []
        
        for run_idx, run_result in enumerate(pso_results['all_runs']):
            # Get parameters for this run
            r_val = run_result.get('r', 0)
            m_c_val = run_result.get('m_c', 0)
            tc_val = run_result.get('tc', 0)
            phi_c_val = run_result.get('phi_c', 0)
            A_val = run_result.get('A', 0)
            delta_t_val = run_result.get('delta_t', 0)
            
            # Convert to physical units
            distance_mpc = 10**float(r_val)
            chirp_mass_msun = 10**float(m_c_val)
            merger_time_s = float(tc_val)
            phase_pi = float(phi_c_val) / np.pi
            flux_ratio = float(A_val)
            time_delay_s = float(delta_t_val)
            
            # Calculate errors
            distance_error = abs((distance_mpc - actual_params['source_distance']) / actual_params['source_distance']) * 100
            chirp_mass_error = abs((chirp_mass_msun - actual_params['chirp_mass']) / actual_params['chirp_mass']) * 100
            merger_time_error = abs((merger_time_s - actual_params['merger_time']) / actual_params['merger_time']) * 100
            phase_error = abs(float(phi_c_val) - actual_params['phase'] * np.pi) / (2 * np.pi) * 100
            flux_ratio_error = abs((flux_ratio - actual_params['flux_ratio']) / actual_params['flux_ratio']) * 100
            time_delay_error = abs((time_delay_s - actual_params['time_delay']) / actual_params['time_delay']) * 100
            
            # Get SNR and match for this run
            snr_val = run_result.get('SNR_pycbc', 0)
            
            # Calculate match for this run
            if 'bestSig' in run_result and run_result['bestSig'] is not None:
                match_val = pycbc_calculate_match(
                    run_result['bestSig'], data_dict['dataY_only_signal'], 
                    data_dict['sampFreq'], data_dict['psdHigh']
                )
            else:
                match_val = 0.0
            
            # Determine if lensed
            is_lensed = flux_ratio >= 0.01
            
            # Add row to CSV data
            csv_data.append({
                'Run_Number': run_idx + 1,
                'Distance_Log10': f"{r_val:.6f}",
                'Distance_Mpc': f"{distance_mpc:.2f}",
                'Distance_Error_Percent': f"{distance_error:.4f}",
                'ChirpMass_Log10': f"{m_c_val:.6f}",
                'ChirpMass_Msun': f"{chirp_mass_msun:.4f}",
                'ChirpMass_Error_Percent': f"{chirp_mass_error:.4f}",
                'MergerTime_s': f"{merger_time_s:.6f}",
                'MergerTime_Error_Percent': f"{merger_time_error:.4f}",
                'Phase_rad': f"{phi_c_val:.6f}",
                'Phase_pi': f"{phase_pi:.6f}",
                'Phase_Error_Percent': f"{phase_error:.4f}",
                'FluxRatio_A': f"{flux_ratio:.6f}",
                'FluxRatio_Error_Percent': f"{flux_ratio_error:.4f}",
                'TimeDelay_s': f"{time_delay_s:.6f}",
                'TimeDelay_Error_Percent': f"{time_delay_error:.4f}",
                'SNR': f"{snr_val:.4f}",
                'Match': f"{match_val:.6f}",
                'Is_Lensed': is_lensed,
                'Avg_Error_Percent': f"{np.mean([distance_error, chirp_mass_error, merger_time_error, phase_error, flux_ratio_error, time_delay_error]):.4f}"
            })
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(csv_data)
        csv_filename = f"{results_dir}/pso_all_runs_detailed_results.csv"
        df.to_csv(csv_filename, index=False)
        
        print(f"✅ PSO all runs CSV saved to: {csv_filename}")
        return csv_filename
        
    except Exception as e:
        print(f"Error generating PSO all runs CSV: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def generate_comparison_csv(pso_results, mcmc_results, actual_params):
    """Generate CSV comparing best PSO results with MCMC results"""
    print("Generating comparison CSV...")
    
    try:
        # Calculate physical parameter values and errors
        
        # PSO best results (physical units)
        pso_distance = 10**float(pso_results['best_params']['r'])
        pso_chirp_mass = 10**float(pso_results['best_params']['m_c'])
        pso_merger_time = float(pso_results['best_params']['tc'])
        pso_phase = float(pso_results['best_params']['phi_c']) / np.pi
        pso_flux_ratio = float(pso_results['best_params']['A'])
        pso_time_delay = float(pso_results['best_params']['delta_t'])
        
        # MCMC best results (physical units)
        mcmc_distance = 10**float(mcmc_results['best_params']['r'])
        mcmc_chirp_mass = 10**float(mcmc_results['best_params']['m_c'])
        mcmc_merger_time = float(mcmc_results['best_params']['tc'])
        mcmc_phase = float(mcmc_results['best_params']['phi_c']) / np.pi
        mcmc_flux_ratio = float(mcmc_results['best_params']['A'])
        mcmc_time_delay = float(mcmc_results['best_params']['delta_t'])
        
        # Calculate errors
        pso_errors = {
            'distance': abs((pso_distance - actual_params['source_distance']) / actual_params['source_distance']) * 100,
            'chirp_mass': abs((pso_chirp_mass - actual_params['chirp_mass']) / actual_params['chirp_mass']) * 100,
            'merger_time': abs((pso_merger_time - actual_params['merger_time']) / actual_params['merger_time']) * 100,
            'phase': abs(float(pso_results['best_params']['phi_c']) - actual_params['phase'] * np.pi) / (2 * np.pi) * 100,
            'flux_ratio': abs((pso_flux_ratio - actual_params['flux_ratio']) / actual_params['flux_ratio']) * 100,
            'time_delay': abs((pso_time_delay - actual_params['time_delay']) / actual_params['time_delay']) * 100
        }
        
        mcmc_errors = {
            'distance': abs((mcmc_distance - actual_params['source_distance']) / actual_params['source_distance']) * 100,
            'chirp_mass': abs((mcmc_chirp_mass - actual_params['chirp_mass']) / actual_params['chirp_mass']) * 100,
            'merger_time': abs((mcmc_merger_time - actual_params['merger_time']) / actual_params['merger_time']) * 100,
            'phase': abs(float(mcmc_results['best_params']['phi_c']) - actual_params['phase'] * np.pi) / (2 * np.pi) * 100,
            'flux_ratio': abs((mcmc_flux_ratio - actual_params['flux_ratio']) / actual_params['flux_ratio']) * 100,
            'time_delay': abs((mcmc_time_delay - actual_params['time_delay']) / actual_params['time_delay']) * 100
        }
        
        # Prepare comparison data
        comparison_data = [
            {
                'Parameter': 'Distance (Mpc)',
                'True_Value': f"{actual_params['source_distance']:.2f}",
                'PSO_Estimate': f"{pso_distance:.2f}",
                'PSO_Error_Percent': f"{pso_errors['distance']:.4f}",
                'MCMC_Estimate': f"{mcmc_distance:.2f}",
                'MCMC_Error_Percent': f"{mcmc_errors['distance']:.4f}",
                'Best_Method': 'PSO' if pso_errors['distance'] < mcmc_errors['distance'] else 'MCMC'
            },
            {
                'Parameter': 'Chirp Mass (M☉)',
                'True_Value': f"{actual_params['chirp_mass']:.4f}",
                'PSO_Estimate': f"{pso_chirp_mass:.4f}",
                'PSO_Error_Percent': f"{pso_errors['chirp_mass']:.4f}",
                'MCMC_Estimate': f"{mcmc_chirp_mass:.4f}",
                'MCMC_Error_Percent': f"{mcmc_errors['chirp_mass']:.4f}",
                'Best_Method': 'PSO' if pso_errors['chirp_mass'] < mcmc_errors['chirp_mass'] else 'MCMC'
            },
            {
                'Parameter': 'Merger Time (s)',
                'True_Value': f"{actual_params['merger_time']:.4f}",
                'PSO_Estimate': f"{pso_merger_time:.4f}",
                'PSO_Error_Percent': f"{pso_errors['merger_time']:.4f}",
                'MCMC_Estimate': f"{mcmc_merger_time:.4f}",
                'MCMC_Error_Percent': f"{mcmc_errors['merger_time']:.4f}",
                'Best_Method': 'PSO' if pso_errors['merger_time'] < mcmc_errors['merger_time'] else 'MCMC'
            },
            {
                'Parameter': 'Phase (π)',
                'True_Value': f"{actual_params['phase']:.6f}",
                'PSO_Estimate': f"{pso_phase:.6f}",
                'PSO_Error_Percent': f"{pso_errors['phase']:.4f}",
                'MCMC_Estimate': f"{mcmc_phase:.6f}",
                'MCMC_Error_Percent': f"{mcmc_errors['phase']:.4f}",
                'Best_Method': 'PSO' if pso_errors['phase'] < mcmc_errors['phase'] else 'MCMC'
            },
            {
                'Parameter': 'Flux Ratio',
                'True_Value': f"{actual_params['flux_ratio']:.6f}",
                'PSO_Estimate': f"{pso_flux_ratio:.6f}",
                'PSO_Error_Percent': f"{pso_errors['flux_ratio']:.4f}",
                'MCMC_Estimate': f"{mcmc_flux_ratio:.6f}",
                'MCMC_Error_Percent': f"{mcmc_errors['flux_ratio']:.4f}",
                'Best_Method': 'PSO' if pso_errors['flux_ratio'] < mcmc_errors['flux_ratio'] else 'MCMC'
            },
            {
                'Parameter': 'Time Delay (s)',
                'True_Value': f"{actual_params['time_delay']:.6f}",
                'PSO_Estimate': f"{pso_time_delay:.6f}",
                'PSO_Error_Percent': f"{pso_errors['time_delay']:.4f}",
                'MCMC_Estimate': f"{mcmc_time_delay:.6f}",
                'MCMC_Error_Percent': f"{mcmc_errors['time_delay']:.4f}",
                'Best_Method': 'PSO' if pso_errors['time_delay'] < mcmc_errors['time_delay'] else 'MCMC'
            }
        ]
        
        # Add performance metrics
        avg_pso_error = np.mean(list(pso_errors.values()))
        avg_mcmc_error = np.mean(list(mcmc_errors.values()))
        
        performance_data = [
            {
                'Parameter': 'Execution Time (s)',
                'True_Value': 'N/A',
                'PSO_Estimate': f"{pso_results['duration']/8:.2f}",
                'PSO_Error_Percent': 'N/A',
                'MCMC_Estimate': f"{mcmc_results['duration']:.2f}",
                'MCMC_Error_Percent': 'N/A',
                'Best_Method': 'PSO'
            },
            {
                'Parameter': 'SNR',
                'True_Value': 'N/A',
                'PSO_Estimate': f"{pso_results['snr']:.4f}",
                'PSO_Error_Percent': 'N/A',
                'MCMC_Estimate': f"{mcmc_results['snr']:.4f}",
                'MCMC_Error_Percent': 'N/A',
                'Best_Method': 'PSO' if pso_results['snr'] > mcmc_results['snr'] else 'MCMC'
            },
            {
                'Parameter': 'Match Score',
                'True_Value': 'N/A',
                'PSO_Estimate': f"{pso_results['match']:.6f}",
                'PSO_Error_Percent': 'N/A',
                'MCMC_Estimate': f"{mcmc_results['match']:.6f}",
                'MCMC_Error_Percent': 'N/A',
                'Best_Method': 'PSO' if pso_results['match'] > mcmc_results['match'] else 'MCMC'
            },
            {
                'Parameter': 'Average Parameter Error (%)',
                'True_Value': 'N/A',
                'PSO_Estimate': f"{avg_pso_error:.4f}",
                'PSO_Error_Percent': 'N/A',
                'MCMC_Estimate': f"{avg_mcmc_error:.4f}",
                'MCMC_Error_Percent': 'N/A',
                'Best_Method': 'PSO' if avg_pso_error < avg_mcmc_error else 'MCMC'
            }
        ]
        
        # Combine all data
        all_data = comparison_data + performance_data
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(all_data)
        csv_filename = f"{results_dir}/pso_vs_mcmc_comparison.csv"
        df.to_csv(csv_filename, index=False)
        
        print(f"✅ Comparison CSV saved to: {csv_filename}")
        return csv_filename
        
    except Exception as e:
        print(f"Error generating comparison CSV: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def generate_required_plots(pso_results, mcmc_results, data_dict, actual_params):
    """Generate the three required plots according to user specifications"""
    print("\n生成图表...")
    
    try:
        # Get data as numpy arrays
        t = data_dict['t']
        dataY_only_signal = data_dict['dataY_only_signal']
        
        # Calculate parameter errors for the tables
        param_keys = ['r', 'm_c', 'tc', 'phi_c', 'A', 'delta_t']
        
        # Convert actual parameters to log scale where needed
        actual_r_log = np.log10(actual_params['source_distance'])
        actual_m_c_log = np.log10(actual_params['chirp_mass'])
        
        # Calculate errors
        pso_errors = {
            'r': abs((10**float(pso_results['best_params']['r']) - actual_params['source_distance']) / actual_params['source_distance']) * 100,
            'm_c': abs((10**float(pso_results['best_params']['m_c']) - actual_params['chirp_mass']) / actual_params['chirp_mass']) * 100,
            'tc': abs((float(pso_results['best_params']['tc']) - actual_params['merger_time']) / actual_params['merger_time']) * 100,
            'phi_c': abs(float(pso_results['best_params']['phi_c']) - actual_params['phase'] * np.pi) / (2 * np.pi) * 100,
            'A': abs((float(pso_results['best_params']['A']) - actual_params['flux_ratio']) / actual_params['flux_ratio']) * 100,
            'delta_t': abs((float(pso_results['best_params']['delta_t']) - actual_params['time_delay']) / actual_params['time_delay']) * 100
        }
        
        mcmc_errors = {
            'r': abs((10**float(mcmc_results['best_params']['r']) - actual_params['source_distance']) / actual_params['source_distance']) * 100,
            'm_c': abs((10**float(mcmc_results['best_params']['m_c']) - actual_params['chirp_mass']) / actual_params['chirp_mass']) * 100,
            'tc': abs((float(mcmc_results['best_params']['tc']) - actual_params['merger_time']) / actual_params['merger_time']) * 100,
            'phi_c': abs(float(mcmc_results['best_params']['phi_c']) - actual_params['phase'] * np.pi) / (2 * np.pi) * 100,
            'A': abs((float(mcmc_results['best_params']['A']) - actual_params['flux_ratio']) / actual_params['flux_ratio']) * 100,
            'delta_t': abs((float(mcmc_results['best_params']['delta_t']) - actual_params['time_delay']) / actual_params['time_delay']) * 100
        }
        
        # 1. Comparison Summary Table (without classification)
        print("1. 生成对比总结表格 (comparison_summary.png)...")
        fig1 = plt.figure(figsize=(10, 6), dpi=150)
        ax1 = fig1.add_subplot(111)
        ax1.axis('tight')
        ax1.axis('off')
        
        # Summary table data (without classification as requested)
        summary_data = [
            ['Metric', 'PSO', 'Bilby MCMC'],
            ['Execution Time (s)', f"{pso_results['duration']/8:.2f}", f"{mcmc_results['duration']:.2f}"],
            ['Speed Ratio', '%.2fx faster' % (mcmc_results['duration']/(pso_results['duration']/8)), 'baseline'],
            ['SNR', f"{pso_results['snr']:.2f}", f"{mcmc_results['snr']:.2f}"],
            ['Match Score', f"{pso_results['match']:.4f}", f"{mcmc_results['match']:.4f}"],
            ['Distance Error (%)', f"{pso_errors['r']:.2f}", f"{mcmc_errors['r']:.2f}"],
            ['Chirp Mass Error (%)', f"{pso_errors['m_c']:.2f}", f"{mcmc_errors['m_c']:.2f}"],
            ['Merger Time Error (%)', f"{pso_errors['tc']:.2f}", f"{mcmc_errors['tc']:.2f}"],
            ['Phase Error (%)', f"{pso_errors['phi_c']:.2f}", f"{mcmc_errors['phi_c']:.2f}"],
            ['Flux Ratio Error (%)', f"{pso_errors['A']:.2f}", f"{mcmc_errors['A']:.2f}"],
            ['Time Delay Error (%)', f"{pso_errors['delta_t']:.2f}", f"{mcmc_errors['delta_t']:.2f}"]
        ]
        
        table1 = ax1.table(cellText=summary_data[1:], colLabels=summary_data[0],
                          cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table1.auto_set_font_size(False)
        table1.set_fontsize(10)
        table1.scale(1, 2)
        
        # Color coding for performance
        for i in range(1, len(summary_data)):
            if i in [3, 4]:  # SNR and Match - higher is better
                if float(summary_data[i][1].split()[0]) > float(summary_data[i][2].split()[0]):
                    table1[(i, 1)].set_facecolor('#d4f7d4')
                else:
                    table1[(i, 2)].set_facecolor('#d4f7d4')
            elif i >= 5:  # Errors - lower is better
                if float(summary_data[i][1]) < float(summary_data[i][2]):
                    table1[(i, 1)].set_facecolor('#d4f7d4')
                else:
                    table1[(i, 2)].set_facecolor('#d4f7d4')
            elif i == 1:  # Execution time - lower is better
                table1[(i, 1)].set_facecolor('#d4f7d4')
        
        plt.title('Algorithm Performance Summary(PSO vs Bilby MCMC)', fontsize=14, pad=20)
        plt.savefig(f"{results_dir}/comparison_summary.png", bbox_inches='tight', dpi=150)
        plt.close()
        
        # 2. Signal Comparison with Ratio Plot
        print("2. signal comparison and ratio plots (signal_comparison.png)...")
        fig2, (ax2_top, ax2_bottom) = plt.subplots(2, 1, figsize=(12, 10), 
                                                   gridspec_kw={'height_ratios': [2, 1]})
        
        # Top panel: Signal comparison (时域图)
        ax2_top.plot(t, dataY_only_signal, 'k-', lw=2, label='Injection Signal (dataY_only_signal)', alpha=0.8)
        ax2_top.plot(t, pso_results['best_signal'], 'r--', lw=1.5, 
                    label=f'PSO (SNR: {pso_results["snr"]:.2f})', alpha=0.7)
        ax2_top.plot(t, mcmc_results['best_signal'], 'b--', lw=1.5, 
                    label=f'MCMC (SNR: {mcmc_results["snr"]:.2f})', alpha=0.7)
        ax2_top.set_ylabel('Strain')
        ax2_top.legend()
        ax2_top.grid(True, alpha=0.3)
        ax2_top.set_title('Time-domain diagrams based on the two algorithms')
        
        # Bottom panel: Ratio plot (比值折线图)
        # Add small epsilon to avoid division by zero
        epsilon = 1e-20
        pso_ratio = pso_results['best_signal'] / (dataY_only_signal + epsilon)
        mcmc_ratio = mcmc_results['best_signal'] / (dataY_only_signal + epsilon)
        
        # Clip extreme values for better visualization
        pso_ratio = np.clip(pso_ratio, -5, 5)
        mcmc_ratio = np.clip(mcmc_ratio, -5, 5)
        
        ax2_bottom.plot(t, pso_ratio, 'r-', lw=2, label='PSO / Injection Signal', alpha=0.7)
        ax2_bottom.plot(t, mcmc_ratio, 'b-', lw=2, label='MCMC / Injection Signal', alpha=0.7)
        ax2_bottom.axhline(y=1, color='black', linestyle='--', alpha=0.8, label='Reference line (ratio = 1)', linewidth=2)
        ax2_bottom.set_xlabel('Time (s)')
        ax2_bottom.set_ylabel('Ratio')
        ax2_bottom.legend()
        ax2_bottom.grid(True, alpha=0.3)
        ax2_bottom.set_ylim(-2, 3)
        ax2_bottom.set_title('Ratio Line Chart of Two Algorithms with Injection Signal')
        
        plt.tight_layout()
        plt.savefig(f"{results_dir}/signal_comparison.png", bbox_inches='tight', dpi=150)
        plt.close()
        
        # 3. Comprehensive Results Table (总表格) - FIXED: All rows have 6 columns
        print("3. 生成总表格...")
        fig3 = plt.figure(figsize=(16, 12), dpi=150)
        ax3 = fig3.add_subplot(111)
        ax3.axis('tight')
        ax3.axis('off')
        
        # Prepare comprehensive table data - FIXED: Consistent 6 columns throughout
        comprehensive_data = [
            ['Parameter', 'True Value', 'PSO Estimate', 'PSO Error (%)', 
             'MCMC Estimate', 'MCMC Error (%)'],
            ['Distance', f"{actual_params['source_distance']:.1f}", 
             f"{10**float(pso_results['best_params']['r']):.1f}", f"{pso_errors['r']:.2f}",
             f"{10**float(mcmc_results['best_params']['r']):.1f}", f"{mcmc_errors['r']:.2f}"],
            ['Chirp Mass', f"{actual_params['chirp_mass']:.2f}", 
             f"{10**float(pso_results['best_params']['m_c']):.2f}", f"{pso_errors['m_c']:.2f}",
             f"{10**float(mcmc_results['best_params']['m_c']):.2f}", f"{mcmc_errors['m_c']:.2f}"],
            ['Merger Time', f"{actual_params['merger_time']:.2f}", 
             f"{float(pso_results['best_params']['tc']):.2f}", f"{pso_errors['tc']:.2f}",
             f"{float(mcmc_results['best_params']['tc']):.2f}", f"{mcmc_errors['tc']:.2f}"],
            ['Phase', f"{actual_params['phase']:.4f}", 
             f"{float(pso_results['best_params']['phi_c'])/np.pi:.4f}", f"{pso_errors['phi_c']:.2f}",
             f"{float(mcmc_results['best_params']['phi_c'])/np.pi:.4f}", f"{mcmc_errors['phi_c']:.2f}"],
            ['Flux Ratio', f"{actual_params['flux_ratio']:.4f}", 
             f"{float(pso_results['best_params']['A']):.4f}", f"{pso_errors['A']:.2f}",
             f"{float(mcmc_results['best_params']['A']):.4f}", f"{mcmc_errors['A']:.2f}"],
            ['Time Delay', f"{actual_params['time_delay']:.4f}", 
             f"{float(pso_results['best_params']['delta_t']):.4f}", f"{pso_errors['delta_t']:.2f}",
             f"{float(mcmc_results['best_params']['delta_t']):.4f}", f"{mcmc_errors['delta_t']:.2f}"],
            ['', '', '', '', '', ''],  # Empty row for separation (FIXED: 6 columns)
            ['Performance Metric', 'PSO', 'MCMC', '', '', ''],  # FIXED: 6 columns
            ['Execution Time', f"{pso_results['duration']/8:.2f} s", f"{mcmc_results['duration']:.2f} s", 
             '', '', ''],  # FIXED: 6 columns
            ['SNR', f"{pso_results['snr']:.2f}", f"{mcmc_results['snr']:.2f}", '', '', ''],  # FIXED: 6 columns
            ['Match Score', f"{pso_results['match']:.4f}", f"{mcmc_results['match']:.4f}", '', '', ''],  # FIXED: 6 columns
            ['Avg Parameter Error', f"{np.mean(list(pso_errors.values())):.2f}%", 
             f"{np.mean(list(mcmc_errors.values())):.2f}%", '', '', '']  # FIXED: 6 columns
        ]
        
        table3 = ax3.table(cellText=comprehensive_data[1:], colLabels=comprehensive_data[0],
                          cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table3.auto_set_font_size(False)
        table3.set_fontsize(9)
        table3.scale(1, 1.8)
        
        # Color coding for errors
        for i in range(1, 7):  # Parameter rows
            # PSO error vs MCMC error
            if float(comprehensive_data[i][3]) < float(comprehensive_data[i][5]):
                table3[(i, 3)].set_facecolor('#d4f7d4')  # Green for better PSO
            else:
                table3[(i, 5)].set_facecolor('#d4f7d4')  # Green for better MCMC
        
        # Highlight performance metrics section
        for j in range(6):
            table3[(8, j)].set_facecolor('#e6e6e6')
        
        # Color performance metrics
        # Execution time (lower is better)
        table3[(9, 1)].set_facecolor('#d4f7d4')  # PSO is faster
        # SNR and Match (higher is better)
        if pso_results['snr'] > mcmc_results['snr']:
            table3[(10, 1)].set_facecolor('#d4f7d4')
        else:
            table3[(10, 2)].set_facecolor('#d4f7d4')
            
        if pso_results['match'] > mcmc_results['match']:
            table3[(11, 1)].set_facecolor('#d4f7d4')
        else:
            table3[(11, 2)].set_facecolor('#d4f7d4')
        
        # Average error (lower is better)
        if np.mean(list(pso_errors.values())) < np.mean(list(mcmc_errors.values())):
            table3[(12, 1)].set_facecolor('#d4f7d4')
        else:
            table3[(12, 2)].set_facecolor('#d4f7d4')
        
        plt.title('Algorithm Performance Comprehensive Table - Parameters, True Values, Estimates, Errors, Execution Time, SNR', 
                 fontsize=14, pad=20)
        plt.savefig(f"{results_dir}/comprehensive_results_table.png", bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"✅ 所有要求的图表已生成并保存到 {results_dir}/ 目录")
        print("生成的文件:")
        print("  1. comparison_summary.png - 对比总结表格(无分类结果)")
        print("  2. signal_comparison.png - 时域图和比值折线图")
        print("  3. comprehensive_results_table.png - 总表格")
        
    except Exception as e:
        print(f"生成图表时出错: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Main function - PSO vs CPU-based MCMC comparison"""

    print("=" * 70)
    print("PSO vs CPU-based MCMC Comparison for Gravitational Wave Analysis")
    print("CPU版本 - PSO粒子数100，迭代2000次")
    print("=" * 70)

    try:
        # Load data
        data_dict = load_data()

        # Set up parameters
        param_ranges, pso_params, pso_config, actual_params = setup_parameters(data_dict)

        # Run PSO with modified parameters (100 particles, 3000 iterations)
        pso_results = run_pso(data_dict, pso_params, pso_config, actual_params)

        # Run CPU-based MCMC with equivalent parameters for fair comparison
        mcmc_results = run_bilby_mcmc(data_dict, param_ranges, n_live_points=100, n_iter=3000)

        # Generate CSV files
        print("\n" + "="*70)
        print("📊 GENERATING CSV FILES")
        print("="*70)
        
        # Generate PSO all runs detailed CSV
        pso_csv_file = generate_pso_all_runs_csv(pso_results, data_dict, actual_params, param_ranges)
        
        # Generate PSO vs MCMC comparison CSV
        comparison_csv_file = generate_comparison_csv(pso_results, mcmc_results, actual_params)

        # Generate the three required plots
        generate_required_plots(pso_results, mcmc_results, data_dict, actual_params)

        # Print parameter estimates to console
        print("\n" + "="*80)
        print("📊 PARAMETER ESTIMATION RESULTS")
        print("="*80)
        
        # Print actual parameters
        print("\n🎯 真实参数 (Actual Parameters):")
        print(f"   距离 (Distance):        {actual_params['source_distance']:.1f} Mpc")
        print(f"   Chirp质量 (Chirp Mass): {actual_params['chirp_mass']:.2f} M☉")
        print(f"   并合时间 (Merger Time):   {actual_params['merger_time']:.2f} s")
        print(f"   相位 (Phase):           {actual_params['phase']:.4f} π")
        print(f"   通量比 (Flux Ratio):    {actual_params['flux_ratio']:.4f}")
        print(f"   时延 (Time Delay):      {actual_params['time_delay']:.4f} s")
        
        # Print PSO estimates
        print("\n🔴 PSO算法估计结果 (PSO Estimates):")
        print(f"   距离 (Distance):        {10**float(pso_results['best_params']['r']):.1f} Mpc")
        print(f"   Chirp质量 (Chirp Mass): {10**float(pso_results['best_params']['m_c']):.2f} M☉")
        print(f"   并合时间 (Merger Time):   {float(pso_results['best_params']['tc']):.2f} s")
        print(f"   相位 (Phase):           {float(pso_results['best_params']['phi_c'])/np.pi:.4f} π")
        print(f"   通量比 (Flux Ratio):    {float(pso_results['best_params']['A']):.4f}")
        print(f"   时延 (Time Delay):      {float(pso_results['best_params']['delta_t']):.4f} s")
        
        # Print MCMC estimates
        print("\n🔵 MCMC算法估计结果 (MCMC Estimates):")
        print(f"   距离 (Distance):        {10**float(mcmc_results['best_params']['r']):.1f} Mpc")
        print(f"   Chirp质量 (Chirp Mass): {10**float(mcmc_results['best_params']['m_c']):.2f} M☉")
        print(f"   并合时间 (Merger Time):   {float(mcmc_results['best_params']['tc']):.2f} s")
        print(f"   相位 (Phase):           {float(mcmc_results['best_params']['phi_c'])/np.pi:.4f} π")
        print(f"   通量比 (Flux Ratio):    {float(mcmc_results['best_params']['A']):.4f}")
        print(f"   时延 (Time Delay):      {float(mcmc_results['best_params']['delta_t']):.4f} s")
        
        # Calculate and print errors
        pso_errors_detailed = {
            'distance': abs((10**float(pso_results['best_params']['r']) - actual_params['source_distance']) / actual_params['source_distance']) * 100,
            'chirp_mass': abs((10**float(pso_results['best_params']['m_c']) - actual_params['chirp_mass']) / actual_params['chirp_mass']) * 100,
            'merger_time': abs((float(pso_results['best_params']['tc']) - actual_params['merger_time']) / actual_params['merger_time']) * 100,
            'phase': abs(float(pso_results['best_params']['phi_c']) - actual_params['phase'] * np.pi) / (2 * np.pi) * 100,
            'flux_ratio': abs((float(pso_results['best_params']['A']) - actual_params['flux_ratio']) / actual_params['flux_ratio']) * 100,
            'time_delay': abs((float(pso_results['best_params']['delta_t']) - actual_params['time_delay']) / actual_params['time_delay']) * 100
        }
        
        mcmc_errors_detailed = {
            'distance': abs((10**float(mcmc_results['best_params']['r']) - actual_params['source_distance']) / actual_params['source_distance']) * 100,
            'chirp_mass': abs((10**float(mcmc_results['best_params']['m_c']) - actual_params['chirp_mass']) / actual_params['chirp_mass']) * 100,
            'merger_time': abs((float(mcmc_results['best_params']['tc']) - actual_params['merger_time']) / actual_params['merger_time']) * 100,
            'phase': abs(float(mcmc_results['best_params']['phi_c']) - actual_params['phase'] * np.pi) / (2 * np.pi) * 100,
            'flux_ratio': abs((float(mcmc_results['best_params']['A']) - actual_params['flux_ratio']) / actual_params['flux_ratio']) * 100,
            'time_delay': abs((float(mcmc_results['best_params']['delta_t']) - actual_params['time_delay']) / actual_params['time_delay']) * 100
        }
        
        # Print error comparison
        print("\n📏 参数估计误差对比 (Parameter Estimation Errors):")
        print("   参数                   PSO误差(%)    MCMC误差(%)    胜者")
        print("   " + "-"*55)
        
        params_names = [
            ('距离 (Distance)', 'distance'),
            ('Chirp质量 (Mass)', 'chirp_mass'),
            ('并合时间 (Time)', 'merger_time'),
            ('相位 (Phase)', 'phase'),
            ('通量比 (Flux)', 'flux_ratio'),
            ('时延 (Delay)', 'time_delay')
        ]
        
        pso_wins = 0
        mcmc_wins = 0
        
        for param_display, param_key in params_names:
            pso_err = pso_errors_detailed[param_key]
            mcmc_err = mcmc_errors_detailed[param_key]
            winner = "PSO" if pso_err < mcmc_err else "MCMC"
            if pso_err < mcmc_err:
                pso_wins += 1
            else:
                mcmc_wins += 1
            print(f"   {param_display:<20} {pso_err:>8.2f}%    {mcmc_err:>9.2f}%    {winner}")
        
        print("   " + "-"*55)
        print(f"   总计胜利次数:          PSO: {pso_wins}次      MCMC: {mcmc_wins}次")
        
        # Print performance summary
        print("\n⚡ 性能对比 (Performance Comparison):")
        print(f"   PSO执行时间:     {pso_results['duration']/8:.2f} 秒 (平均每次运行)")
        print(f"   MCMC执行时间:    {mcmc_results['duration']:.2f} 秒")
        print(f"   速度比:         PSO比MCMC快 {mcmc_results['duration']/(pso_results['duration']/8):.1f}倍")
        print(f"   PSO SNR:        {pso_results['snr']:.2f}")
        print(f"   MCMC SNR:       {mcmc_results['snr']:.2f}")
        print(f"   PSO 匹配度:      {pso_results['match']:.4f}")
        print(f"   MCMC 匹配度:     {mcmc_results['match']:.4f}")
        
        # Overall winner
        print("\n🏆 总结 (Overall Summary):")
        if pso_wins > mcmc_wins:
            print(f"   PSO在参数估计精度上获胜 ({pso_wins}/{len(params_names)} 参数)")
        elif mcmc_wins > pso_wins:
            print(f"   MCMC在参数估计精度上获胜 ({mcmc_wins}/{len(params_names)} 参数)")
        else:
            print("   PSO和MCMC在参数估计精度上平分秋色")
        
        
        # Print files generated
        print("\n📁 生成的文件 (Generated Files):")
        print(f"   图表文件夹: {results_dir}/")
        print("   图表文件:")
        print("     - comparison_summary.png")
        print("     - signal_comparison.png") 
        print("     - comprehensive_results_table.png")
        print("   CSV文件:")
        if pso_csv_file:
            print(f"     - {os.path.basename(pso_csv_file)} (PSO所有8次运行详细结果)")
        if comparison_csv_file:
            print(f"     - {os.path.basename(comparison_csv_file)} (PSO最佳结果 vs MCMC对比)")
        
        print("="*80)

    except Exception as e:
        print(f"主程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()