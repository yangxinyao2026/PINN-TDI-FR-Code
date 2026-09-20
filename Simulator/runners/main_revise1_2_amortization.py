# -*- coding: utf-8 -*-
"""Run the R1.2 offline-cost amortization and break-even analysis."""
import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyomo.environ as pyo
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Compatible withPyCharmRun this file directly in ``python -m`` two ways.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Simulator import PROJECT_ROOT
from Simulator.Approximator import PreTrainNet, FullNet
from Simulator.cases import TD_case
import Simulator.cases.DS_case_3phase as DS_case_3phase


CASES = {
    'case33bw_ds': {
        'weights_dir': 'A(36,2)_type3(2,29)_lr1(3e-5)_lr2(1e-5)_rate(1e-4)',
        'three_phase': False,
    },
    'case118zh_ds': {
        'weights_dir': ('A(36,2)_type3(97, 107, 109, 80, 63, 31)_'
                        'lr1(3e-4)_lr2(1e-4)_rate(1e-4)'),
        'three_phase': False,
    },
    'case533mt_hi_ds': {
        'weights_dir': ('A(36,2)_type3(9-36)_lr1(1e-4)_'
                        'lr2(1e-4)_rate(1e-4)'),
        'three_phase': False,
    },
    'case36real_3phase_ds': {
        'weights_dir': ('A(36,2)_type3(8, 11)_lr1(3e-4)_'
                        'lr2(1e-4)_rate(1e-4)'),
        'three_phase': True,
    },
}

# original manuscriptTable 1inm=8andm=36Configured online time (seconds). These values are on the current machine
# Manuscript Experiment Server (Intel Core i9-12900H + NVIDIA RTX 3060 Laptop GPU) on
# To preserve the original data and timing protocol, R1.2 reads these values directly.
# Original values, only new amortization analysis, no retiming on other computers.
ORIGINAL_ONLINE_SECONDS = {
    'case33bw_ds': {
        8: {'optimization': 0.40, 'pinn_gpu': 0.030},
        36: {'optimization': 1.78, 'pinn_gpu': 0.028},
    },
    'case118zh_ds': {
        8: {'optimization': 0.79, 'pinn_gpu': 0.030},
        36: {'optimization': 3.43, 'pinn_gpu': 0.029},
    },
    'case533mt_hi_ds': {
        8: {'optimization': 4.61, 'pinn_gpu': 0.031},
        36: {'optimization': 22.60, 'pinn_gpu': 0.028},
    },
    'case36real_3phase_ds': {
        8: {'optimization': 1.18, 'pinn_gpu': 0.031},
        36: {'optimization': 5.22, 'pinn_gpu': 0.028},
    },
}

ORIGINAL_HARDWARE = {
    'source': 'EPSR-D-26-06663.pdf, Table 1 and Section 4.1',
    'server_relation': 'current server; original-manuscript experiment server',
    'manuscript_note': ('submitted hardware statement should be corrected; '
                        'RTX 5080 belongs to the other revision server'),
    'protocol': 'original_manuscript_single_complete_region_timing',
    'cpu': 'Intel Core i9-12900H',
    'gpu': 'NVIDIA GeForce RTX 3060 Laptop GPU',
    'optimization_device': 'CPU (IPOPT)',
    'pinn_device': 'GPU',
    'reported_polygon_facets': [8, 36],
    'amortization_polygon_facets': 36,
}

N_DIRECTIONS = 36
N_OPT_TRIALS = 30
N_PINN_REPEATS = 1000
N_PINN_WARMUP = 100
DTHETA_RANGE = (-0.5, 0.5)
SEED = 42

OUT = (PROJECT_ROOT / 'results' / 'ds_proj_revise_V1' /
       'comparison' / 'amortization')
PAPER_RESULTS = PROJECT_ROOT / 'results' / 'ds_proj_paper'


SUMMARY_FIELDS = [
    'case', 'n_directions', 'n_opt_trials', 'n_pinn_repeats',
    'pretrain_training_s', 'fullnet_training_s', 'offline_training_s',
    'offline_training_h', 'offline_wall_s',
    'opt_mean_s', 'opt_median_s', 'opt_p95_s', 'opt_success_rate',
    'pinn_cpu_mean_s', 'pinn_cpu_median_s', 'pinn_cpu_p95_s',
    'pinn_gpu_mean_s', 'pinn_gpu_median_s', 'pinn_gpu_p95_s',
    'break_even_calls_cpu', 'break_even_calls_gpu',
    'days_5min_cpu', 'days_5min_gpu',
    'days_15min_cpu', 'days_15min_gpu',
]


def case_data(casename):
    """Only the original physics study is constructed; no thermal constraints, measurement errors, or other rework settings are enabled."""
    if casename == 'case33bw_ds':
        ppc = TD_case.case33bw_ds()
    elif casename == 'case118zh_ds':
        ppc = TD_case.case118zh_ds()
    elif casename == 'case533mt_hi_ds':
        ppc = TD_case.case533mt_hi_ds()
    elif casename == 'case36real_3phase_ds':
        ppc = DS_case_3phase.case36real_3phase_ds()
    else:
        raise ValueError(f'Unsupported calculation examples: {casename}')

    if CASES[casename]['three_phase']:
        case = DS_case_3phase.DScase_3phase_train(
            casedata=ppc, model_type='fullnet', plot_flag=False,
            total_samples=1, batch_size=1, device='cpu')
    else:
        case = TD_case.DScase_train(
            casedata=ppc, model_type='fullnet', plot_flag=False,
            total_samples=1, batch_size=1, device='cpu')
    return ppc, case


def weights_path(casename):
    path = PAPER_RESULTS / casename / CASES[casename]['weights_dir']
    required = [
        'pretrainnet_weights.pth', 'fullnet_weights_feasible.pth',
        'pretrainnet_training_time.npz', 'fullnet_training_time.npz']
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(
            f'{casename} The original results directory is missing {missing}: {path}')
    return path


def read_scalar(npz, preferred):
    for key in preferred:
        if key in npz.files:
            return float(np.asarray(npz[key]))
    raise KeyError(f'Timing file missing fields {preferred}, The actual field is {npz.files}')


def read_offline_times(result_dir):
    pre = np.load(result_dir / 'pretrainnet_training_time.npz')
    full = np.load(result_dir / 'fullnet_training_time.npz')
    pretrain_training = read_scalar(pre, ('phase1_time', 'pretrain_time', 'total_time'))
    fullnet_training = read_scalar(full, ('total_train_time', 'fullnet_total_time', 'total_time'))
    pretrain_wall = read_scalar(pre, ('total_time', 'phase1_time', 'pretrain_time'))
    fullnet_wall = read_scalar(full, ('total_time', 'total_train_time', 'fullnet_total_time'))
    return {
        'pretrain_training_s': pretrain_training,
        'fullnet_training_s': fullnet_training,
        'offline_training_s': pretrain_training + fullnet_training,
        'offline_wall_s': pretrain_wall + fullnet_wall,
    }


def load_fullnet(case, result_dir, device):
    """Press original single layer128 ReLUStructure loading finally worksFullNet."""
    pre = PreTrainNet(
        case['A_hat'], case['b_hat'], is_epigraph=False, device=device)
    pre.load_state_dict(torch.load(
        result_dir / 'pretrainnet_weights.pth', map_location=device))
    pre = pre.to(device)
    with torch.no_grad():
        A_pre, b_pre = pre()
    A_pre = A_pre[0].detach().cpu().numpy()
    b_pre = b_pre[0].detach().cpu().numpy()

    model = FullNet(
        dim_theta=case['params']['count'], A_init=A_pre, b_init=b_pre,
        n_hidden=128, device=device).to(device)
    model.load_state_dict(torch.load(
        result_dir / 'fullnet_weights_feasible.pth', map_location=device))
    model.eval()
    return model


def generate_dthetas(dim_theta, n_trials):
    rng = np.random.RandomState(SEED)
    values = rng.uniform(
        DTHETA_RANGE[0], DTHETA_RANGE[1],
        size=(n_trials, dim_theta))
    values[0] = 0.0
    return values


def update_parameters(error_calculator, init_params_dict, dtheta):
    pd_init = init_params_dict['Pd_meta']['initial_value']
    qd_init = init_params_dict['Qd_meta']['initial_value']
    n_pd = pd_init.size
    error_calculator.update_parameters({
        'Pd_meta': pd_init + dtheta[:n_pd].reshape(pd_init.shape),
        'Qd_meta': qd_init + dtheta[n_pd:].reshape(qd_init.shape),
    })


def benchmark_optimization(error_calculator, init_params_dict, dthetas):
    """Timing parameter update+36aIPOPTDirection solution, complete output36edge reference polyhedron."""
    ec = error_calculator
    ec.solver.options['tol'] = 1e-11
    ec.solver.options['max_iter'] = 10000
    angles = np.linspace(0.0, 2.0 * np.pi, N_DIRECTIONS, endpoint=False)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))

    # An untimed warm-up eliminates the initial initialization overhead of the solver.
    update_parameters(ec, init_params_dict, dthetas[0])
    for direction in directions:
        ec.optimize_direction(direction, in_approx=False)

    timings = np.full(len(dthetas), np.nan)
    success_counts = np.zeros(len(dthetas), dtype=int)
    for idx, dtheta in enumerate(dthetas):
        start = time.perf_counter()
        update_parameters(ec, init_params_dict, dtheta)
        count = 0
        for direction in directions:
            if ec.optimize_direction(direction, in_approx=False) is not None:
                count += 1
        timings[idx] = time.perf_counter() - start
        success_counts[idx] = count
        print(
            f'    optimization {idx + 1:02d}/{len(dthetas)}: '
            f'{timings[idx]:.4f}s, success={count}/{N_DIRECTIONS}')
    return timings, success_counts


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def benchmark_pinn(model, dthetas, device, repeats, warmup):
    """End-to-end single sample latency: numpyinput→tensor→forward→A,btransfer backCPU numpy."""
    with torch.inference_mode():
        for idx in range(warmup):
            tensor = torch.as_tensor(
                dthetas[idx % len(dthetas)], dtype=torch.float32,
                device=device)
            A, b = model(tensor)
            _ = A.detach().cpu().numpy(), b.detach().cpu().numpy()
        synchronize(device)

        timings = np.empty(repeats, dtype=float)
        for idx in range(repeats):
            dtheta = dthetas[idx % len(dthetas)]
            synchronize(device)
            start = time.perf_counter()
            tensor = torch.as_tensor(
                dtheta, dtype=torch.float32, device=device)
            A, b = model(tensor)
            _ = A.detach().cpu().numpy(), b.detach().cpu().numpy()
            synchronize(device)
            timings[idx] = time.perf_counter() - start
    return timings


def statistics(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {'mean': np.nan, 'median': np.nan, 'p95': np.nan}
    return {
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'p95': float(np.percentile(values, 95)),
    }


def break_even(offline_s, optimization_s, inference_s):
    saving = optimization_s - inference_s
    if not np.isfinite(saving) or saving <= 0.0:
        return np.nan
    return int(math.ceil(offline_s / saving))


def days_for_calls(calls, interval_minutes):
    if not np.isfinite(calls):
        return np.nan
    calls_per_day = 24.0 * 60.0 / interval_minutes
    return float(calls / calls_per_day)


def hardware_info():
    info = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python': platform.python_version(),
        'numpy': np.__version__,
        'torch': torch.__version__,
        'torch_cpu_threads': torch.get_num_threads(),
        'cuda_available': torch.cuda.is_available(),
        'timing_clock': 'time.perf_counter',
        'optimization_device': 'CPU (IPOPT)',
        'pinn_cpu_device': 'CPU',
        'pinn_gpu_device': 'CUDA GPU' if torch.cuda.is_available() else 'unavailable',
    }
    # Windowsonplatform.processorSometimes empty, registry reads as fallback.
    if os.name == 'nt':
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r'HARDWARE\DESCRIPTION\System\CentralProcessor\0')
            info['processor'] = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        except Exception:
            pass
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update({
            'gpu_name': props.name,
            'gpu_memory_bytes': int(props.total_memory),
            'cuda_runtime': torch.version.cuda,
        })
    try:
        import psutil
        info['physical_memory_bytes'] = int(psutil.virtual_memory().total)
    except Exception:
        pass
    try:
        executable = str(pyo.SolverFactory('ipopt').executable())
        info['ipopt_executable'] = Path(executable).name
        proc = subprocess.run(
            [executable, '--version'], capture_output=True, text=True,
            timeout=10, check=False)
        info['ipopt_version'] = (proc.stdout or proc.stderr).strip().splitlines()[0]
    except Exception as exc:
        info['ipopt_version'] = f'unavailable: {exc}'
    return info


def save_summary(rows):
    OUT.mkdir(parents=True, exist_ok=True)
        # Resume incomplete cases while allowing an explicitly rerun case to overwrite its prior result.
    merged = {}
    summary_path = OUT / 'summary.csv'
    if summary_path.exists():
        with open(summary_path, 'r', encoding='utf-8') as f:
            merged.update({row['case']: row for row in csv.DictReader(f)})
    merged.update({row['case']: row for row in rows})
    ordered_cases = [name for name in CASES if name in merged]
    ordered_cases.extend(sorted(set(merged) - set(CASES)))
    rows = [merged[name] for name in ordered_cases]
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    np.savez(
        OUT / 'summary.npz',
        case=np.asarray([row['case'] for row in rows]),
        **{field: np.asarray([float(row[field]) for row in rows], dtype=float)
           for field in SUMMARY_FIELDS if field != 'case'})


def original_protocol_row(casename):
    """Read the original finalm=36Online time, only newly added offline time is calculated--Online amortization indicator."""
    result_dir = weights_path(casename)
    offline = read_offline_times(result_dir)
    published = ORIGINAL_ONLINE_SECONDS[casename][36]
    opt_s = float(published['optimization'])
    pinn_gpu_s = float(published['pinn_gpu'])
    offline_s = float(offline['offline_training_s'])
    be_gpu = break_even(offline_s, opt_s, pinn_gpu_s)

    # Keep what you havesummaryFields for easy reading by plotting and historical code. The manuscript reports the complete region only once
    # online time, somean/median/P95Should not be interpreted as a repeated test statistic.
    return {
        'case': casename,
        'n_directions': N_DIRECTIONS,
        'n_opt_trials': 1,
        'n_pinn_repeats': 1,
        'pretrain_training_s': offline['pretrain_training_s'],
        'fullnet_training_s': offline['fullnet_training_s'],
        'offline_training_s': offline_s,
        'offline_training_h': offline_s / 3600.0,
        'offline_wall_s': offline['offline_wall_s'],
        'opt_mean_s': opt_s,
        'opt_median_s': opt_s,
        'opt_p95_s': opt_s,
        'opt_success_rate': 1.0,
        'pinn_cpu_mean_s': np.nan,
        'pinn_cpu_median_s': np.nan,
        'pinn_cpu_p95_s': np.nan,
        'pinn_gpu_mean_s': pinn_gpu_s,
        'pinn_gpu_median_s': pinn_gpu_s,
        'pinn_gpu_p95_s': pinn_gpu_s,
        'break_even_calls_cpu': np.nan,
        'break_even_calls_gpu': be_gpu,
        'days_5min_cpu': np.nan,
        'days_5min_gpu': days_for_calls(be_gpu, 5),
        'days_15min_cpu': np.nan,
        'days_15min_gpu': days_for_calls(be_gpu, 15),
    }


def save_original_online_table(selected):
    """Save originalTable 1ofm=8andm=36Online time without remeasurement."""
    fields = ['case', 'm', 'vertex_generation_s', 'pinn_gpu_s']
    rows = []
    for casename in selected:
        for m in (8, 36):
            values = ORIGINAL_ONLINE_SECONDS[casename][m]
            opt_s = float(values['optimization'])
            pinn_s = float(values['pinn_gpu'])
            rows.append({
                'case': casename,
                'm': m,
                'vertex_generation_s': opt_s,
                'pinn_gpu_s': pinn_s,
            })
    with open(OUT / 'original_online_times.csv', 'w',
              newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    np.savez(
        OUT / 'original_online_times.npz',
        case=np.asarray([row['case'] for row in rows]),
        m=np.asarray([row['m'] for row in rows], dtype=int),
        vertex_generation_s=np.asarray(
            [row['vertex_generation_s'] for row in rows], dtype=float),
        pinn_gpu_s=np.asarray(
            [row['pinn_gpu_s'] for row in rows], dtype=float),
        source=np.asarray('original manuscript Table 1; no remeasurement'))
    return rows


def run_original_protocol(selected):
    """Quickly generate and original manuscriptsTable 1Formal with consistent timing caliberR1.2result."""
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'hardware.json', 'w', encoding='utf-8') as f:
        json.dump(ORIGINAL_HARDWARE, f, ensure_ascii=False, indent=2)

    rows = []
    print('=' * 76)
    print('R1.2 Offline--Online amortization analysis (read directly from the manuscriptm=8andm=36Online time) ')
    print('Platform: current server (original experimental server) ')
    print('Hardware: Intel Core i9-12900H + NVIDIA RTX 3060 Laptop GPU')
    print('=' * 76)
    save_original_online_table(selected)
    for casename in selected:
        row = original_protocol_row(casename)
        rows.append(row)
        print(
            f"{casename}: offline={row['offline_training_h']:.3f} h, "
            f"optimization={row['opt_median_s']:.3f} s, "
            f"PINN-GPU={1e3 * row['pinn_gpu_median_s']:.1f} ms, "
            f"break-even={int(row['break_even_calls_gpu'])} calls")
    save_summary(rows)
    print(f'Manuscript Online Timetable: {OUT / "original_online_times.csv"}')
    return rows


def run_case(casename, n_opt_trials, n_pinn_repeats, n_warmup):
    print('\n' + '=' * 76)
    print(f'Calculation example: {casename}')
    print('=' * 76)
    result_dir = weights_path(casename)
    offline = read_offline_times(result_dir)
    ppc, case = case_data(casename)
    dthetas = generate_dthetas(case['params']['count'], n_opt_trials)

    print(f"  Manuscript weight: {result_dir}")
    print(
        f"  Offline training: pretrain={offline['pretrain_training_s']:.1f}s, "
        f"fullnet={offline['fullnet_training_s']:.1f}s, "
        f"total={offline['offline_training_s'] / 3600.0:.3f}h")
    print(f'  [1/3] Optimization method CPU/IPOPT: {n_opt_trials}Working conditions×{N_DIRECTIONS}direction')
    opt_times, opt_success = benchmark_optimization(
        case['errorcalculator'], case['params']['params_dict'], dthetas)

    print(f'  [2/3] PINN CPU: warmup={n_warmup}, repeats={n_pinn_repeats}')
    cpu = torch.device('cpu')
    model_cpu = load_fullnet(case, result_dir, cpu)
    cpu_times = benchmark_pinn(
        model_cpu, dthetas, cpu, n_pinn_repeats, n_warmup)
    del model_cpu

    gpu_times = np.array([], dtype=float)
    if torch.cuda.is_available():
        print(f'  [3/3] PINN GPU: warmup={n_warmup}, repeats={n_pinn_repeats}')
        gpu = torch.device('cuda')
        model_gpu = load_fullnet(case, result_dir, gpu)
        gpu_times = benchmark_pinn(
            model_gpu, dthetas, gpu, n_pinn_repeats, n_warmup)
        del model_gpu
        torch.cuda.empty_cache()
    else:
        print('  [3/3] not detectedCUDA, GPUColumn asNaN')

    opt_stat = statistics(opt_times)
    cpu_stat = statistics(cpu_times)
    gpu_stat = statistics(gpu_times)
    offline_s = offline['offline_training_s']
    be_cpu = break_even(offline_s, opt_stat['median'], cpu_stat['median'])
    be_gpu = break_even(offline_s, opt_stat['median'], gpu_stat['median'])
    success_rate = float(np.sum(opt_success) / (len(opt_success) * N_DIRECTIONS))

    case_out = OUT / casename
    case_out.mkdir(parents=True, exist_ok=True)
    np.savez(
        case_out / 'timing_raw.npz',
        casename=np.asarray(casename),
        dthetas=dthetas,
        n_directions=np.asarray(N_DIRECTIONS),
        optimization_seconds=opt_times,
        optimization_success_counts=opt_success,
        pinn_cpu_seconds=cpu_times,
        pinn_gpu_seconds=gpu_times,
        pretrain_training_seconds=np.asarray(offline['pretrain_training_s']),
        fullnet_training_seconds=np.asarray(offline['fullnet_training_s']),
        offline_training_seconds=np.asarray(offline_s),
        offline_wall_seconds=np.asarray(offline['offline_wall_s']),
        timing_scope_optimization=np.asarray(
            'parameter_update_plus_36_ipopt_direction_solves'),
        timing_scope_pinn=np.asarray(
            'numpy_input_to_tensor_forward_and_A_b_cpu_output'),
        model_loading_included=np.asarray(False),
        warmup_included=np.asarray(False))

    row = {
        'case': casename,
        'n_directions': N_DIRECTIONS,
        'n_opt_trials': n_opt_trials,
        'n_pinn_repeats': n_pinn_repeats,
        'pretrain_training_s': offline['pretrain_training_s'],
        'fullnet_training_s': offline['fullnet_training_s'],
        'offline_training_s': offline_s,
        'offline_training_h': offline_s / 3600.0,
        'offline_wall_s': offline['offline_wall_s'],
        'opt_mean_s': opt_stat['mean'],
        'opt_median_s': opt_stat['median'],
        'opt_p95_s': opt_stat['p95'],
        'opt_success_rate': success_rate,
        'pinn_cpu_mean_s': cpu_stat['mean'],
        'pinn_cpu_median_s': cpu_stat['median'],
        'pinn_cpu_p95_s': cpu_stat['p95'],
        'pinn_gpu_mean_s': gpu_stat['mean'],
        'pinn_gpu_median_s': gpu_stat['median'],
        'pinn_gpu_p95_s': gpu_stat['p95'],
        'break_even_calls_cpu': be_cpu,
        'break_even_calls_gpu': be_gpu,
        'days_5min_cpu': days_for_calls(be_cpu, 5),
        'days_5min_gpu': days_for_calls(be_gpu, 5),
        'days_15min_cpu': days_for_calls(be_cpu, 15),
        'days_15min_gpu': days_for_calls(be_gpu, 15),
    }
    print(
        f"  result: opt median={opt_stat['median']:.4f}s, "
        f"PINN-CPU={1e3 * cpu_stat['median']:.4f}ms, "
        f"PINN-GPU={1e3 * gpu_stat['median']:.4f}ms")
    print(
        f"        break-even CPU={be_cpu}times/5min={row['days_5min_cpu']:.2f}day, "
        f"GPU={be_gpu}times/5min={row['days_5min_gpu']:.2f}day")
    return row


def main():
    global OUT
    parser = argparse.ArgumentParser(description='R1.2Online timing and break-even analysis')
    parser.add_argument(
        '--remeasure', action='store_true',
        help=('Re-execute30Working condition optimization andGPUSteady state timing, diagnostic only; '
              'The default is to read the original directlyTable 1ofm=8andm=36Online time'))
    parser.add_argument('--quick', action='store_true',
                        help='cooperate--remeasurePerform pipeline debugging; cannot be used for papers')
    parser.add_argument('--cases', default=None,
                        help='Run only specified calculation examples, separated by commas; default all4a')
    args = parser.parse_args()

    # Keep debug and retiming outputs separate from the official result directory.
    if args.quick:
        OUT = OUT / 'quick_smoke'
    elif args.remeasure:
        OUT = OUT / 'remeasured_diagnostic'

    selected = list(CASES)
    if args.cases:
        wanted = [item.strip() for item in args.cases.split(',') if item.strip()]
        unknown = sorted(set(wanted) - set(CASES))
        if unknown:
            raise ValueError(f'Unknown case: {unknown}; Optional: {list(CASES)}')
        selected = wanted

    if not args.remeasure:
        start = time.time()
        run_original_protocol(selected)
        if set(selected) == set(CASES):
            print('\n[plot] Generate manuscript caliber amortized cost comparison chart...')
            from Simulator.draw_pictures.R1_2_amortization_plot import main as plot_main
            plot_main()
        elapsed = time.time() - start
        print(f'\ncompleted, time consuming: {elapsed:.1f}s')
        print(f'Summary: {OUT / "summary.csv"}')
        return

    n_opt_trials = 2 if args.quick else N_OPT_TRIALS
    n_pinn_repeats = 20 if args.quick else N_PINN_REPEATS
    n_warmup = 5 if args.quick else N_PINN_WARMUP
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'hardware.json', 'w', encoding='utf-8') as f:
        json.dump(hardware_info(), f, ensure_ascii=False, indent=2)

    print('=' * 76)
    print('R1.2 Offline--Online amortization retiming diagnostics')
    print(f'mode: {"QUICK (Debug only) " if args.quick else "REMEASURED DIAGNOSTIC"}')
    print(f'Calculation example: {selected}')
    print(f'output: {OUT}')
    print('=' * 76)
    start = time.time()
    rows = []
    for casename in selected:
        rows.append(run_case(
            casename, n_opt_trials, n_pinn_repeats, n_warmup))
        save_summary(rows)  # Place the order every time one is completed, and the completed results will not be lost due to unexpected interruptions.

    elapsed = time.time() - start
    print('\n' + '=' * 76)
    print(f'All completed, total time taken: {elapsed:.1f}s ({elapsed / 60.0:.1f}min) ')
    print(f'Summary: {OUT / "summary.csv"}')
    print('=' * 76)


if __name__ == '__main__':
    main()
