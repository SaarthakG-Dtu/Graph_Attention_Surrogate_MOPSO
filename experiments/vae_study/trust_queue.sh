#!/bin/bash
cd /e/SUTD_BENCHMARK_NOVEL_GNN
export PYTHONIOENCODING=utf-8
S=experiments/vae_study
python -u $S/search_sweep.py --vae_tag rdn_d200 --bounds 0.5 --maxvels 0.02 --name tr_d200_a > $S/sweep_tr_d200_a.log 2>&1
python -u $S/search_sweep.py --vae_tag rdn_d200 --bounds 1.0 --maxvels 0.05 0.1 --name tr_d200_b > $S/sweep_tr_d200_b.log 2>&1
python -u $S/search_sweep.py --vae_tag rdn_d200 --bounds 2.0 3.0 --maxvels 0.1 0.3 --name tr_d200_c > $S/sweep_tr_d200_c.log 2>&1
python -u $S/search_sweep.py --space binary --bounds 3.0 --maxvels 0.1 0.25 0.5 --name tr_binary > $S/sweep_tr_binary.log 2>&1
python - <<'PY'
import json, glob
best = lambda fs: max(sum((json.load(open(f)) for f in fs), []), key=lambda r: r["mean"])
L = best(glob.glob("experiments/vae_study/sweep_tr_d200_*.json")); B = best(["experiments/vae_study/sweep_tr_binary.json"])
json.dump({"vae": "rdn_d200", "latent": [L["bound"], L["maxvel"], L["mean"]], "binary": [B["bound"], B["maxvel"], B["mean"]]},
          open("experiments/vae_study/scaling_settings.json", "w"))
print("settings", L["bound"], L["maxvel"], L["mean"], B["maxvel"], B["mean"])
PY
python -u $S/scaling.py --seeds 0 1 2 > $S/scaling_trust.log 2>&1
touch $S/trust.done
