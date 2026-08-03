#!/usr/bin/env python
"""Complete the TITAN attention audit with persistence, head, spatial and rank diagnostics."""
from __future__ import annotations
import argparse
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from tqdm.auto import tqdm

def resolve(value, manifest):
    p = Path(value)
    if p.is_absolute(): return p
    candidates = [(manifest.parent / p).resolve(), (manifest.parent.parent / p).resolve(), (Path.cwd() / p).resolve()]
    return next((x for x in candidates if x.exists()), candidates[0])

def norm(x):
    x = np.clip(x.astype(np.float64), 0, None); return x / np.clip(x.sum(-1, keepdims=True), 1e-12, None)

def top(x, frac):
    n = max(1, int(np.ceil(len(x)*frac))); return np.argpartition(x, -n)[-n:]

def recall(a, b, frac):
    return len(set(top(a, frac)) & set(top(b, frac))) / len(top(b, frac))

def effective_rank(matrix):
    values = np.linalg.eigvalsh(np.corrcoef(matrix)); values = np.clip(values, 0, None)
    return float(values.sum()**2 / np.square(values).sum()) if values.sum() else np.nan

def spatial_metrics(x, coords, frac, k):
    n = len(x); tree = cKDTree(coords[:, :2]); _, nn = tree.query(coords[:, :2], k=min(k+1, n)); nn = nn[:, 1:]
    centered = x - x.mean(); denom = np.square(centered).sum(); w = n * nn.shape[1]
    moran = float(n / w * (centered[:, None] * centered[nn]).sum() / denom) if denom else np.nan
    geary = float((n - 1) / (2*w) * np.square(x[:, None] - x[nn]).sum() / denom) if denom else np.nan
    chosen = top(x, frac); chosen_set = set(chosen); edges = {int(i): [int(j) for j in nn[i] if int(j) in chosen_set] for i in chosen}
    components = 0; seen = set()
    for start in chosen:
        if int(start) in seen: continue
        components += 1; stack = [int(start)]; seen.add(int(start))
        while stack:
            node = stack.pop()
            for nxt in edges[node]:
                if nxt not in seen: seen.add(nxt); stack.append(nxt)
    extent = coords.max(0) - coords.min(0); bbox = coords[chosen].max(0) - coords[chosen].min(0)
    coverage = float(np.prod(bbox[:2]) / max(np.prod(extent[:2]), 1))
    sample = chosen if len(chosen) <= 1000 else np.random.default_rng(17).choice(chosen, 1000, replace=False)
    distance = float(cKDTree(coords[sample, :2]).query(coords[sample, :2], k=2)[0][:, 1].mean())
    return moran, geary, components, coverage, distance

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wsi-output-manifest', type=Path, required=True); p.add_argument('--metadata', type=Path)
    p.add_argument('--output-dir', type=Path, required=True); p.add_argument('--top-fraction', type=float, default=.10)
    p.add_argument('--spatial-k', type=int, default=8); p.add_argument('--noise-scale', type=float, default=.05); p.add_argument('--max-slides', type=int)
    args = p.parse_args(); outputs = pd.read_csv(args.wsi_output_manifest); outputs = outputs[outputs.status.isin(['complete','skipped'])]
    if args.max_slides: outputs = outputs.sort_values('slide_id').head(args.max_slides)
    meta = pd.read_csv(args.metadata) if args.metadata else pd.DataFrame(columns=['slide_id','project'])
    if 'tcga_project' in meta and 'project' not in meta: meta = meta.rename(columns={'tcga_project':'project'})
    projects = dict(zip(meta.slide_id, meta.get('project', pd.Series(dtype=str))))
    slides=[]; persistence=[]; heads=[]; rank_rows=[]; errors=[]
    for item in tqdm(outputs.itertuples(index=False), total=len(outputs), desc='Complete TITAN audit', unit='slide'):
        try:
            with h5py.File(resolve(item.path, args.wsi_output_manifest)) as f:
                glob=norm(np.asarray(f['attention']['global_to_tiles_mass_share'])); recv=norm(np.asarray(f['attention']['received_by_tiles_broadcast'])); coords=np.asarray(f['coords'])
            L,H,N=glob.shape; final=L-1; base={'slide_id':item.slide_id,'case_id':'-'.join(item.slide_id.split('-')[:3]),'project':projects.get(item.slide_id),'n_tiles':N}
            final_mean=glob[final].mean(0); final_mean/=final_mean.sum(); final_top=top(final_mean,args.top_fraction)
            first=[]; persistent=[]
            for idx in final_top:
                membership=np.array([idx in set(top(glob[layer].mean(0),args.top_fraction)) for layer in range(L)])
                first.append(int(np.argmax(membership))); persistent.append(float(membership.mean()))
            persistence.append({**base,'top_fraction':args.top_fraction,'final_top_tiles':len(final_top),'first_entry_layer_median':float(np.median(first)),'entered_by_layer1_fraction':float(np.mean(np.array(first)<=1)),'persistence_fraction_mean':float(np.mean(persistent))})
            layer_head=[]
            for layer in range(L):
                corr=np.corrcoef(glob[layer]); pair=corr[np.triu_indices(H,1)]
                overlaps=[recall(glob[layer,i],glob[layer,j],args.top_fraction) for i in range(H) for j in range(i+1,H)]
                heads.append({**base,'layer':layer,'head_agreement_spearman_mean':float(np.nanmean(pair)),'head_effective_rank':effective_rank(glob[layer]),'head_top_overlap_mean':float(np.mean(overlaps)),'global_received_spearman_mean':float(np.mean([spearmanr(glob[layer,h],recv[layer,h]).statistic for h in range(H)]))})
                layer_head.append(glob[layer].mean(0))
            for early in range(L-1):
                rho=float(spearmanr(layer_head[early],layer_head[final]).statistic); logits=np.log(np.clip(layer_head[early],1e-12,None)); sigma=args.noise_scale*np.std(logits)
                rng=np.random.default_rng(17); stability=np.mean([recall(logits+rng.normal(0,sigma,N),logits,args.top_fraction) for _ in range(10)])
                rank_rows.append({**base,'early_layer':early,'target_layer':final,'spearman':rho,'top10_recall':recall(layer_head[early],layer_head[final],args.top_fraction),'rank_gap_at_top_boundary':float(np.sort(layer_head[early])[::-1][max(0,int(np.ceil(N*args.top_fraction))-1)]-np.sort(layer_head[early])[::-1][min(N-1,int(np.ceil(N*args.top_fraction)))]),'top_set_noise_stability':float(stability)})
            moran,geary,components,coverage,distance=spatial_metrics(final_mean,coords,args.top_fraction,args.spatial_k)
            slides.append({**base,'spatial_moran_final':moran,'spatial_geary_final':geary,'top_components_final':components,'top_bbox_coverage_final':coverage,'top_nearest_neighbor_distance_final':distance,'final_top10_mass':float(final_mean[top(final_mean,args.top_fraction)].sum())})
        except Exception as exc: errors.append({'slide_id':item.slide_id,'error':repr(exc)})
    out=args.output_dir; out.mkdir(parents=True,exist_ok=True)
    per_slide=pd.DataFrame(slides).merge(pd.DataFrame(persistence),on=list(slides[0].keys() & persistence[0].keys()) if slides else ['slide_id'],how='left')
    per_slide.to_csv(out/'attention_structure_extended_per_slide.csv',index=False); pd.DataFrame(heads).to_csv(out/'head_agreement_extended.csv',index=False); pd.DataFrame(rank_rows).to_csv(out/'early_final_stability_extended.csv',index=False); pd.DataFrame(errors).to_csv(out/'errors_extended.csv',index=False)
    if len(per_slide):
        per_slide['n_tiles_quartile']=pd.qcut(per_slide.n_tiles,4,duplicates='drop'); per_slide.groupby('project',dropna=False).mean(numeric_only=True).to_csv(out/'by_project.csv'); per_slide.groupby('n_tiles_quartile',observed=True).mean(numeric_only=True).to_csv(out/'by_tile_count_quartile.csv')
    print(f'slides={len(slides)} errors={len(errors)} results={out}')
if __name__ == '__main__': main()
