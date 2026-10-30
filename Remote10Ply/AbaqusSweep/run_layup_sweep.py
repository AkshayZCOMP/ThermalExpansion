from __future__ import print_function

import csv
import itertools
import json
import os
import shutil
import subprocess
import sys
import time


# ==============================================================================
# Sweep settings
# ==============================================================================
MIN_PLIES = 3
MAX_PLIES = 10

# Edit this list if you want a different design space.
ANGLE_SET = [-45, 0, 45, 90]

# Balanced means each positive/negative angle pair appears the same number of
# times. Add pairs here if ANGLE_SET includes other angle pairs, e.g. (30, -30).
BALANCE_PAIRS = [(45, -45)]

ABAQUS_SCRIPT = 'ProperlyRotatedFREE.py'
CHECKPOINT_FILE = 'layup_sweep_checkpoint.json'
RUN_LOG_DIR = 'layup_sweep_logs'
# Use a writable, project-local directory by default.  Set an absolute path
# here only when Abaqus needs to run elsewhere.
ABAQUS_WORK_DIR = None
SUMMARY_CSV = 'ABaqusData.csv'
COMPARISON_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'Computational Real',
    'compare_abaqus_to_computational.py'
)

# A case agrees when every generalized strain and curvature component falls
# within these absolute differences (computational model minus Abaqus).
STRAIN_TOLERANCE = 2.0e-4
CURVATURE_TOLERANCE = 5.0e-1

# Change this if you want a specific Abaqus launcher, e.g. 'abq2024.bat'.
ABAQUS_EXECUTABLE = 'abaqus.bat'

# Set True to print the planned cases without running Abaqus.
DRY_RUN = False

# If True, stop the sweep on the first failed Abaqus run.
EXIT_ON_FAILURE = True

# Flag comparison failures in the checkpoint and log, but keep evaluating the
# rest of the design space.
EXIT_ON_DISAGREEMENT = False


def layup_key(layup):
    return '[' + '/'.join(str(int(x)) if float(x).is_integer() else str(x) for x in layup) + ']'


def is_balanced(layup):
    for pos_angle, neg_angle in BALANCE_PAIRS:
        if layup.count(pos_angle) != layup.count(neg_angle):
            return False
    return True


def make_symmetric_layup(seed, nplies):
    if nplies % 2 == 0:
        return list(seed) + list(reversed(seed))
    return list(seed) + list(reversed(seed[:-1]))


def generate_symmetric_balanced_layups(nplies):
    seed_len = (nplies + 1) // 2
    seen = set()
    layups = []

    for seed in itertools.product(ANGLE_SET, repeat=seed_len):
        layup = make_symmetric_layup(seed, nplies)
        key = layup_key(layup)
        if key in seen:
            continue
        seen.add(key)

        if is_balanced(layup):
            layups.append(layup)

    return layups


def all_cases():
    """Return the original balanced cases plus full-unidirectional cases.

    The unidirectional cases are appended after the existing design space so
    their addition does not renumber any in-progress balanced-sweep jobs.
    [0]n and [90]n already satisfy the symmetric-balanced rule, so suppress
    them here if they are already present rather than scheduling duplicates.
    """
    cases = []
    case_keys = set()
    for nplies in range(MIN_PLIES, MAX_PLIES + 1):
        for layup in generate_symmetric_balanced_layups(nplies):
            cases.append((nplies, layup))
            case_keys.add(layup_key(layup))

    for nplies in range(MIN_PLIES, MAX_PLIES + 1):
        for angle in ANGLE_SET:
            layup = [angle] * nplies
            key = layup_key(layup)
            if key not in case_keys:
                cases.append((nplies, layup))
                case_keys.add(key)

    return cases


def load_checkpoint(path):
    if not os.path.exists(path):
        return {
            'completed': [],
            'compared': [],
            'failed': [],
            'mismatches': [],
            'last_started': None
        }

    with open(path, 'r') as f:
        data = json.load(f)

    data.setdefault('completed', [])
    data.setdefault('compared', [])
    data.setdefault('failed', [])
    data.setdefault('mismatches', [])
    data.setdefault('last_started', None)
    return data


def save_checkpoint(path, checkpoint):
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(checkpoint, f, indent=2, sort_keys=True)
    if os.path.exists(path):
        os.remove(path)
    os.rename(tmp_path, path)


def make_job_name(nplies, index):
    return 'LaminateShellIRJob_p%03d_i%06d' % (nplies, index)


def run_abaqus_case(script_dir, work_dir, summary_csv_path, layup, job_name):
    env = os.environ.copy()
    env['ABAQUS_LAYUP'] = repr(layup)
    env['ABAQUS_JOB_NAME'] = job_name
    env['ABAQUS_SUMMARY_CSV'] = summary_csv_path

    abaqus_executable = shutil.which(ABAQUS_EXECUTABLE)
    if abaqus_executable is None:
        for candidate in ['abaqus.bat', 'abq2024.bat', 'abq2023.bat', 'abq2022.bat', 'abaqus']:
            abaqus_executable = shutil.which(candidate)
            if abaqus_executable is not None:
                break

    if abaqus_executable is None:
        print('Could not find Abaqus on PATH.')
        print('Set ABAQUS_EXECUTABLE in this script to the full path of abaqus.bat.')
        return 2

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    abaqus_script_path = os.path.join(script_dir, ABAQUS_SCRIPT)
    abaqus_command_args = ['cae', 'noGUI=' + abaqus_script_path]

    log_dir = os.path.join(script_dir, RUN_LOG_DIR)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, job_name + '.log')

    print('Running %s with layup %s' % (job_name, layup_key(layup)))
    try:
        with open(log_path, 'w') as log:
            process = subprocess.Popen(
                [abaqus_executable] + abaqus_command_args,
                cwd=work_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
            saw_error = False
            for raw_line in iter(process.stdout.readline, b''):
                try:
                    line = raw_line.decode('utf-8')
                except UnicodeDecodeError:
                    line = raw_line.decode('mbcs', 'replace')
                sys.stdout.write(line)
                log.write(line)
                if 'Abaqus Error:' in line or 'Traceback (most recent call last)' in line:
                    saw_error = True
            process.wait()

        if saw_error and process.returncode == 0:
            return 1
        return process.returncode
    except OSError as err:
        print('Could not launch Abaqus: %s' % err)
        print('Tried executable: %s' % abaqus_executable)
        return 2


def run_comparison(
    script_dir,
    summary_csv_path,
    layup=None,
    fail_on_disagreement=False,
):
    output_csv_path = os.path.join(
        script_dir,
        'ABaqusData_with_computational_results.csv'
    )
    command = [
        sys.executable,
        COMPARISON_SCRIPT,
        '--input-csv', summary_csv_path,
        '--output-csv', output_csv_path,
        '--strain-tolerance', str(STRAIN_TOLERANCE),
        '--curvature-tolerance', str(CURVATURE_TOLERANCE),
    ]
    if fail_on_disagreement:
        command.append('--fail-on-disagreement')
    if layup is not None:
        command.extend(['--layup', layup])

    if layup is None:
        print('Comparing Abaqus results with the computational model...')
    else:
        print('Comparing new Abaqus case %s with the computational model...' % layup)
    return subprocess.call(command, cwd=script_dir), output_csv_path


def get_case_agreement(comparison_csv_path, layup):
    """Return the latest comparison result for one layup from the CSV."""
    result = None
    with open(comparison_csv_path, 'r') as comparison_file:
        for row in csv.DictReader(comparison_file):
            if row['Layup'] == layup:
                result = row

    if result is None:
        raise RuntimeError('No comparison result found for %s' % layup)
    return result['Agreement'], float(result['CaseMaxNormalizedError'])


def record_mismatch(checkpoint, layup, normalized_error):
    """Keep one current mismatch record per layup in the resume checkpoint."""
    checkpoint['mismatches'] = [
        item for item in checkpoint.setdefault('mismatches', [])
        if item['layup'] != layup
    ]
    checkpoint['mismatches'].append({
        'layup': layup,
        'max_normalized_error': normalized_error,
        'flagged_at': time.strftime('%Y-%m-%d %H:%M:%S')
    })


def clear_mismatch(checkpoint, layup):
    checkpoint['mismatches'] = [
        item for item in checkpoint.setdefault('mismatches', [])
        if item['layup'] != layup
    ]


def sync_mismatches(checkpoint, comparison_csv_path):
    """Synchronize all existing comparison flags into the checkpoint."""
    flagged_layups = []
    with open(comparison_csv_path, 'r') as comparison_file:
        for row in csv.DictReader(comparison_file):
            layup = row['Layup']
            if row['Agreement'] == 'FAIL':
                record_mismatch(
                    checkpoint, layup, float(row['CaseMaxNormalizedError'])
                )
                flagged_layups.append(layup)
            else:
                clear_mismatch(checkpoint, layup)
    return flagged_layups


def compared_layups(comparison_csv_path):
    """Return layups with a completed computational result in the CSV."""
    layups = set()
    with open(comparison_csv_path, 'r') as comparison_file:
        for row in csv.DictReader(comparison_file):
            if row.get('Comp_e0_11', '') != '':
                layups.add(row['Layup'])
    return layups


def mark_compared(checkpoint, layup):
    if layup not in checkpoint['compared']:
        checkpoint['compared'].append(layup)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    work_dir = ABAQUS_WORK_DIR or os.path.join(script_dir, 'abaqus_work')
    summary_csv_path = os.path.join(script_dir, SUMMARY_CSV)
    checkpoint_path = os.path.join(script_dir, CHECKPOINT_FILE)
    checkpoint = load_checkpoint(checkpoint_path)
    completed = set(checkpoint['completed'])

    cases = all_cases()
    pending = [(i, n, l) for i, (n, l) in enumerate(cases, start=1) if layup_key(l) not in completed]

    print('Total generated cases: %d' % len(cases))
    print('Already completed: %d' % len(completed))
    print('Pending: %d' % len(pending))

    if DRY_RUN:
        for index, nplies, layup in pending:
            print('%06d: %d plies %s' % (index, nplies, layup_key(layup)))
        return 0

    comparison_csv_path = os.path.join(
        script_dir,
        'ABaqusData_with_computational_results.csv'
    )
    if os.path.exists(comparison_csv_path):
        existing_flags = sync_mismatches(checkpoint, comparison_csv_path)
        for layup in compared_layups(comparison_csv_path):
            mark_compared(checkpoint, layup)
        save_checkpoint(checkpoint_path, checkpoint)
        print(
            'Reusing existing comparison CSV; existing flags: %d' %
            len(existing_flags)
        )

    # Only reconcile cases that finished in Abaqus but were interrupted before
    # their single-case computational comparison could be written.
    if os.path.exists(summary_csv_path):
        deferred_comparisons = [
            layup for layup in checkpoint['completed']
            if layup not in checkpoint['compared']
        ]
        for layup in deferred_comparisons:
            print('Completing deferred comparison for %s...' % layup)
            comparison_code, comparison_csv_path = run_comparison(
                script_dir, summary_csv_path, layup=layup
            )
            if comparison_code != 0:
                return comparison_code
            mark_compared(checkpoint, layup)
            sync_mismatches(checkpoint, comparison_csv_path)
            save_checkpoint(checkpoint_path, checkpoint)

    try:
        for index, nplies, layup in pending:
            key = layup_key(layup)
            job_name = make_job_name(nplies, index)

            checkpoint['last_started'] = {
                'index': index,
                'nplies': nplies,
                'layup': key,
                'job_name': job_name,
                'started_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            save_checkpoint(checkpoint_path, checkpoint)

            return_code = run_abaqus_case(
                script_dir, work_dir, summary_csv_path, layup, job_name
            )
            if return_code != 0:
                failure = {
                    'index': index,
                    'nplies': nplies,
                    'layup': key,
                    'job_name': job_name,
                    'return_code': return_code,
                    'failed_at': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                checkpoint['failed'].append(failure)
                save_checkpoint(checkpoint_path, checkpoint)
                print('Abaqus failed with return code %s for %s' % (return_code, key))
                if EXIT_ON_FAILURE:
                    return return_code
                continue

            checkpoint['completed'].append(key)
            checkpoint['last_started'] = None
            save_checkpoint(checkpoint_path, checkpoint)
            print('Completed %s' % key)

            comparison_code, comparison_csv_path = run_comparison(
                script_dir, summary_csv_path, layup=key
            )
            if comparison_code != 0:
                print('Comparison failed for %s with return code %s' % (
                    key, comparison_code
                ))
                return comparison_code

            agreement, max_normalized_error = get_case_agreement(
                comparison_csv_path, key
            )
            mark_compared(checkpoint, key)
            sync_mismatches(checkpoint, comparison_csv_path)
            if agreement == 'PASS':
                print('Comparison PASS for %s (max normalized error %.3f)' % (
                    key, max_normalized_error
                ))
            else:
                print('COMPARISON FLAGGED for %s (max normalized error %.3f)' % (
                    key, max_normalized_error
                ))
                if EXIT_ON_DISAGREEMENT:
                    save_checkpoint(checkpoint_path, checkpoint)
                    return 1
            save_checkpoint(checkpoint_path, checkpoint)

    except KeyboardInterrupt:
        save_checkpoint(checkpoint_path, checkpoint)
        print('\nStopped. Checkpoint saved to %s' % checkpoint_path)
        return 130

    print('Sweep complete.')
    if not os.path.exists(summary_csv_path):
        print('No Abaqus summary CSV was produced, so no comparison was run.')
        return 1

    mismatch_count = len(checkpoint.get('mismatches', []))
    print('Sweep complete. Comparison results were updated case by case.')
    return 1 if mismatch_count else 0


if __name__ == '__main__':
    sys.exit(main())
