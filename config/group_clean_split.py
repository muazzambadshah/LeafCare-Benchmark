"""Exact all-pairs descriptor threshold graph, connected groups, and grouped resplit."""
import argparse,csv,json,hashlib,time
from pathlib import Path
import numpy as np
import torch

def main(a):
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False
    with (a.audit/'image_index.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    x=np.load(a.audit/'features.npy',mmap_mode='r');n=len(rows);assert x.shape==(n,576)
    parent=np.arange(n);size=np.ones(n,dtype=int)
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=int(parent[i])
        return i
    def union(i,j):
        i,j=find(i),find(j)
        if i==j:return
        if size[i]<size[j]:i,j=j,i
        parent[j]=i;size[i]+=size[j]
    for key in ['image_sha256','crop_rgb224_sha256']:
        lookup={}
        for i,r in enumerate(rows):
            if r[key] in lookup:union(i,lookup[r[key]])
            else:lookup[r[key]]=i
    tensor=torch.nn.functional.normalize(torch.as_tensor(np.array(x),device='cuda'),dim=1)
    edge_count=0;started=time.time()
    with (a.output/'cosine_edges.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['index_a','index_b','cosine'])
        for start in range(0,n,256):
            scores=tensor[start:start+256]@tensor.T
            rr,cc=torch.where(scores>=.99);mask=cc>(rr+start);rr=rr[mask];cc=cc[mask]
            values=scores[rr,cc].cpu().tolist();left=(rr+start).cpu().tolist();right=cc.cpu().tolist()
            for i,j,v in zip(left,right,values):union(i,j);w.writerow([i,j,v])
            edge_count+=len(left)
            if start%12800==0:print(f'All-pairs rows {start}/{n}; edges {edge_count}; seconds {time.time()-started:.1f}',flush=True)
    groups={}
    for i in range(n):groups.setdefault(find(i),[]).append(i)
    labels=sorted({r['label'] for r in rows});li={v:i for i,v in enumerate(labels)}
    total=np.bincount([li[r['label']] for r in rows],minlength=len(labels))
    targets=np.asarray([.7,.1,.1,.1])[:,None]*total[None,:];counts=np.zeros_like(targets)
    splits=['train','val','calibration','test'];assignment={};rng=np.random.default_rng(20261110)
    records=[]
    for members in groups.values():
        hist=np.bincount([li[rows[i]['label']] for i in members],minlength=len(labels))
        gid=hashlib.sha256('\n'.join(sorted(rows[i]['image_sha256'] for i in members)).encode()).hexdigest()
        records.append((members,hist,gid,float(rng.random())))
    # Rare-label groups first; then larger groups. Whole components are indivisible.
    records.sort(key=lambda g:(min(total[np.flatnonzero(g[1])]),-len(g[0]),g[3]))
    for members,hist,gid,_ in records:
        costs=[]
        for split in range(4):
            before=((counts-targets)**2/np.maximum(targets,1)).sum()
            candidate=counts.copy();candidate[split]+=hist
            after=((candidate-targets)**2/np.maximum(targets,1)).sum()
            costs.append(after-before)
        selected=int(np.argmin(costs));counts[selected]+=hist
        for i in members:assignment[i]=(splits[selected],gid)
    fields=list(rows[0])+['clean_split','duplicate_group']
    with (a.output/'grouped_manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i,r in enumerate(rows):w.writerow(dict(r,clean_split=assignment[i][0],duplicate_group=assignment[i][1]))
    # Verify every recorded graph edge stays inside one new split.
    with (a.output/'cosine_edges.csv').open(newline='') as f:
        assert all(assignment[int(r['index_a'])][0]==assignment[int(r['index_b'])][0] for r in csv.DictReader(f))
    summary={'images':n,'groups':len(groups),'cosine_edges':edge_count,'threshold':.99,'maximum_group_images':max(map(len,groups.values())),'split_counts':dict(zip(splits,counts.sum(1).astype(int).tolist())),'missing_labels':{s:[labels[i] for i in np.flatnonzero(counts[j]==0)] for j,s in enumerate(splits)},'precision':'Normalized descriptors, strict float32 matrix products; exact all-pairs threshold graph','limitations':'Similarity groups do not establish plant/session identity. Label conflicts and large components require review. No checkpoint was trained on this manifest yet. All historical files unchanged.'}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
