"""Compact status of the 3 entry-edge DeepMimic runs: latest Iteration/Samples/
Test_Return/Test_EpLen(+trend)/Pose_R/Key_Pos_R, plus a stdout NaN/error scan.
Run repeatedly to watch progress.  python tools/monitor_edges.py"""
import os, glob
RUNS = [
    ('W2  tadasana->warrior2  ', 'output/yoga_edges_v3/tadasana_to_warrior2_deepmimic'),
    ('W3  tadasana->warrior3  ', 'output/yoga_edges_v3/tadasana_to_warrior3_deepmimic'),
    ('HS  downdog3L->handstand', 'output/yoga_edges_v3/downdog3L_to_handstand_deepmimic'),
    ('DD  tadasana->downdog   ', 'output/yoga_edges_v3/tadasana_to_downdog_deepmimic'),
]
WANT = ['Iteration', 'Samples', 'Test_Return', 'Test_Episode_Length', 'Pose_R', 'Key_Pos_R', 'Reward_Total']

def parse(run):
    f = os.path.join(run, 'log.txt')
    if not os.path.exists(f):
        return None
    data = open(f, newline='').read().replace('\r', '\n')
    rows = [r for r in data.split('\n') if r.strip()]
    if len(rows) < 2:
        return None
    hdr = rows[0].split()
    idx = {h: i for i, h in enumerate(hdr)}
    def val(row, col):
        t = row.split()
        return t[idx[col]] if col in idx and idx[col] < len(t) else '?'
    data_rows = rows[1:]
    return hdr, idx, data_rows, val

def main():
    for label, run in RUNS:
        p = parse(run)
        # error scan
        so = os.path.join(run, 'stdout.txt')
        errs = 0
        if os.path.exists(so):
            txt = open(so, errors='ignore').read()
            for k in ('Traceback', 'AssertionError', 'nan', 'NaN', 'Killed', 'OOM', 'CUDA error', 'RuntimeError'):
                errs += txt.count(k)
        alive = ''
        if p is None:
            print(f'{label} | no log yet   | stdout_errs={errs}')
            continue
        hdr, idx, rows, val = p
        last = rows[-1]
        smp = float(val(last, 'Samples')) / 1e6 if val(last, 'Samples') not in ('?',) else -1
        el = val(last, 'Test_Episode_Length')
        # EpLen trend over last up-to-5 rows
        trend = [val(r, 'Test_Episode_Length') for r in rows[-5:]]
        tr = '->'.join(t if t == '?' else f'{float(t):.0f}' for t in trend)
        print(f'{label} | it {val(last,"Iteration"):>5} {smp:6.1f}M | '
              f'Test_Ret {float(val(last,"Test_Return")):6.1f} | '
              f'EpLen {float(el):5.1f} [{tr}] | '
              f'Pose_R {float(val(last,"Pose_R")):.2f} Key_R {float(val(last,"Key_Pos_R")):.2f} | '
              f'errs={errs}')

if __name__ == '__main__':
    main()
