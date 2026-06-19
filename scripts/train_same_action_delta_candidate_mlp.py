from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class MLP(nn.Module):
    def __init__(self, din, dout, hidden, layers, dropout):
        super().__init__()
        mods=[]; d=din
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        mods.append(nn.Linear(d, dout))
        self.net = nn.Sequential(*mods)
    def forward(self, x): return self.net(x)

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--groups", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--mode", choices=["full","action_only","state_only","no_context"], required=True)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-groups", type=int, default=64)
    p.add_argument("--eval-batch-groups", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=0.02)
    p.add_argument("--lambda-mse", type=float, default=0.01)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def cond(z,a,mode):
    if mode=="full": return torch.cat([z,a],-1)
    if mode=="action_only": return a
    if mode=="state_only": return z
    if mode=="no_context": return torch.ones((z.shape[0],1),device=z.device,dtype=z.dtype)
    raise ValueError(mode)

def rank_metrics(pred_delta, cand_delta):
    # pred_delta [G,D], cand_delta [G,K,D], correct index = 0
    dist=((pred_delta[:,None,:]-cand_delta)**2).mean(-1)
    order=np.argsort(dist,axis=1)
    top1=(order[:,0]==0)
    ranks=np.array([np.where(order[i]==0)[0][0]+1 for i in range(len(order))])
    margin=dist[:,1:].min(1)-dist[:,0]
    return {
        "top1":float(top1.mean()),
        "mean_rank":float(ranks.mean()),
        "positive_margin_frac":float((margin>0).mean()),
        "mean_margin":float(margin.mean())
    }

@torch.no_grad()
def evaluate(model,z,y,a,anchor,cand,gids,mode,variant,bg,device,seed):
    model.eval(); rng=np.random.default_rng(seed)
    preds=[]; tgts=[]
    delta_all=y-z
    for s in range(0,len(gids),bg):
        ids=gids[s:s+bg]; anch=anchor[ids]; c=cand[ids]
        zc=z[anch].copy(); aa=a[anch].copy()
        if variant=="original": pass
        elif variant=="action_zero": aa=np.zeros_like(aa)
        elif variant=="cond_state_shuffle": zc=zc[rng.permutation(len(zc))]
        elif variant=="cond_state_zero": zc=np.zeros_like(zc)
        else: raise ValueError(variant)
        zt=torch.from_numpy(zc).float().to(device)
        at=torch.from_numpy(aa).float().to(device)
        pred=model(cond(zt,at,mode))
        preds.append(pred.cpu().numpy())
        tgts.append(delta_all[c])
    return rank_metrics(np.concatenate(preds), np.concatenate(tgts))

def main():
    a=args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    if a.device=="cuda" and not torch.cuda.is_available(): a.device="cpu"
    a.out_dir.mkdir(parents=True,exist_ok=True); a.report.parent.mkdir(parents=True,exist_ok=True)

    d=np.load(a.data,allow_pickle=True); g=np.load(a.groups,allow_pickle=True)
    z=d["z_current"].astype(np.float32).reshape(len(d["z_current"]),-1)
    y=d["z_future"].astype(np.float32).reshape(len(d["z_future"]),-1)
    act=d["action"].astype(np.float32)
    anchor=g["anchor_indices"].astype(np.int64)
    cand=g["candidate_indices"].astype(np.int64)
    delta=y-z

    G,K=cand.shape
    rng=np.random.default_rng(a.seed); perm=rng.permutation(G)
    ntr=int(a.train_frac*G); nv=int(a.val_frac*G)
    tr,va,te=perm[:ntr],perm[ntr:ntr+nv],perm[ntr+nv:]

    din=cond(torch.zeros(1,z.shape[1]),torch.zeros(1,act.shape[1]),a.mode).shape[1]
    model=MLP(din,z.shape[1],a.hidden,a.layers,a.dropout).to(a.device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    ce=nn.CrossEntropyLoss()
    loader=DataLoader(TensorDataset(torch.from_numpy(tr).long()),batch_size=a.batch_groups,shuffle=True)

    best=-1; best_state=None; train_rows=[]
    for ep in range(1,a.epochs+1):
        model.train(); losses=[]
        for (gid_t,) in loader:
            gids=gid_t.numpy(); anch=anchor[gids]; c=cand[gids]
            zc=torch.from_numpy(z[anch]).float().to(a.device)
            aa=torch.from_numpy(act[anch]).float().to(a.device)
            tgt=torch.from_numpy(delta[c]).float().to(a.device)
            pred=model(cond(zc,aa,a.mode))
            dist=((pred[:,None,:]-tgt)**2).mean(-1)
            logits=-dist/a.temperature
            loss_ce=ce(logits,torch.zeros(len(gids),dtype=torch.long,device=a.device))
            loss_mse=((pred-tgt[:,0,:])**2).mean()
            loss=loss_ce+a.lambda_mse*loss_mse
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(float(loss.item()))
        vm=evaluate(model,z,y,act,anchor,cand,va,a.mode,"original",a.eval_batch_groups,a.device,a.seed)
        row={"epoch":ep,"loss":float(np.mean(losses)),"val_top1":vm["top1"],"val_rank":vm["mean_rank"]}
        train_rows.append(row)
        if vm["top1"]>best:
            best=vm["top1"]; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if ep==1 or ep%20==0 or ep==a.epochs:
            print(f"mode={a.mode} epoch={ep:03d} loss={row['loss']:.6f} val_top1={row['val_top1']:.6f} val_rank={row['val_rank']:.3f}",flush=True)

    if best_state is not None: model.load_state_dict(best_state)
    torch.save({"model":model.state_dict(),"args":vars(a),"input_dim":int(din),"output_dim":int(z.shape[1]),"best_val_top1":float(best)}, a.out_dir/"checkpoint.pt")

    rows=[]
    for split,gids in [("train",tr),("val",va),("test",te)]:
        for var in ["original","action_zero","cond_state_shuffle","cond_state_zero"]:
            m=evaluate(model,z,y,act,anchor,cand,gids,a.mode,var,a.eval_batch_groups,a.device,a.seed)
            m.update({"split":split,"variant":var}); rows.append(m)

    rep={"data":str(a.data),"groups":str(a.groups),"mode":a.mode,"n_groups":int(G),"candidates":int(K),"train_val_test":[len(tr),len(va),len(te)],"chance_top1":1.0/K,"best_val_top1":float(best),"training_rows":train_rows,"eval_rows":rows}
    a.report.write_text(json.dumps(rep,indent=2)+"\n")

    md=a.report.with_suffix(".md")
    lines=[f"# Same-action delta-candidate MLP: {a.mode}","",f"- data: `{a.data}`",f"- groups: `{a.groups}`",f"- mode: `{a.mode}`",f"- groups count: `{G}`",f"- candidates: `{K}`",f"- train/val/test groups: `{len(tr)}/{len(va)}/{len(te)}`",f"- chance top-1: `{1.0/K:.6f}`",f"- best val top-1: `{best:.6f}`","","| split | variant | top-1 | mean rank | positive margin frac | mean margin |","| --- | --- | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['split']} | {r['variant']} | {r['top1']:.6f} | {r['mean_rank']:.6f} | {r['positive_margin_frac']:.6f} | {r['mean_margin']:.6f} |")
    lines += ["","## Interpretation","","This ranks predicted latent displacements against candidate latent displacements, removing the absolute-state shortcut from future-space ranking."]
    md.write_text("\n".join(lines)+"\n")
    print(md); print(md.read_text())

if __name__=="__main__":
    main()
