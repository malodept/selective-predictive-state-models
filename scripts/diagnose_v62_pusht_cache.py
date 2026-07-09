from pathlib import Path
import argparse
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=str, required=True)
    args = ap.parse_args()

    d = dict(np.load(args.cache))
    state = d["state"]
    fut = d["future_state"]
    sid = d["state_id"]
    aid = d["action_id"]
    action = d["action"]
    coverage = d.get("coverage", np.zeros(len(sid)))

    delta = fut - state
    agent_disp = np.linalg.norm(delta[:, :2], axis=1)
    block_disp = np.linalg.norm(delta[:, 2:4], axis=1)
    block_rot = np.abs(delta[:, 4])
    total = len(sid)

    print("rows", total)
    print("states", len(np.unique(sid)))
    print("actions", sorted(np.unique(aid).tolist()))
    print()
    print("agent_disp mean/med/max", agent_disp.mean(), np.median(agent_disp), agent_disp.max())
    print("block_disp mean/med/max", block_disp.mean(), np.median(block_disp), block_disp.max())
    print("block_rot  mean/med/max", block_rot.mean(), np.median(block_rot), block_rot.max())
    print("coverage   mean/med/max", coverage.mean(), np.median(coverage), coverage.max())
    print()
    for thr in [0.5, 1, 2, 5, 10]:
        print(f"block_disp > {thr:4.1f} px:", float((block_disp > thr).mean()))
    print()
    print("by action:")
    rows = []
    for a in sorted(np.unique(aid)):
        m = aid == a
        rows.append({
            "action": int(a),
            "n": int(m.sum()),
            "agent_disp": float(agent_disp[m].mean()),
            "block_disp": float(block_disp[m].mean()),
            "block_disp>2": float((block_disp[m] > 2).mean()),
            "block_disp>5": float((block_disp[m] > 5).mean()),
            "block_rot": float(block_rot[m].mean()),
            "coverage": float(coverage[m].mean()),
            "action_dx": float(action[m,0].mean()),
            "action_dy": float(action[m,1].mean()),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    print()
    print("within-state action diversity:")
    divs = []
    block_divs = []
    for s in np.unique(sid):
        idx = np.where(sid == s)[0]
        if len(idx) < 2:
            continue
        # pairwise future full-state and block-only distances
        fs = fut[idx]
        bd = fut[idx, 2:5]
        ds = []
        bs = []
        for i in range(len(idx)):
            for j in range(i+1, len(idx)):
                ds.append(np.linalg.norm(fs[i] - fs[j]))
                bs.append(np.linalg.norm(bd[i] - bd[j]))
        divs.append(np.mean(ds))
        block_divs.append(np.mean(bs))
    print("future_state pairwise mean/med/max", np.mean(divs), np.median(divs), np.max(divs))
    print("block_future pairwise mean/med/max", np.mean(block_divs), np.median(block_divs), np.max(block_divs))

if __name__ == "__main__":
    main()
