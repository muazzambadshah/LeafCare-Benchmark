"""One PyTorch loop for five model families and the four PhytoEdgeNet variants.

Fixed ten-epoch budget, three seeds, validation-only checkpoint selection. This
is a new experiment, not a modification or resumption of historical checkpoints.
"""
import argparse,csv,json,hashlib,random,time,os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import models,transforms
from PIL import Image
from sklearn.metrics import f1_score
from phytoedge_net import PhytoEdgeNet

MODELS=['mobilenet_v3_small','phytoedge_head_only','phytoedge_attention','phytoedge_fusion','phytoedge_net','efficientnet_b0','convnext_tiny','yolo26n']
class Images(Dataset):
    def __init__(self,root,rows,mapping,training,model):
        self.root,self.rows,self.mapping=root,rows,mapping
        geometry=[transforms.RandomResizedCrop(224,scale=(.8,1.)),transforms.RandomHorizontalFlip()] if training else [transforms.Resize(256),transforms.CenterCrop(224)]
        geometry+=[transforms.ToTensor()]
        if model!='yolo26n':geometry+=[transforms.Normalize([.485,.456,.406],[.229,.224,.225])]
        self.transform=transforms.Compose(geometry)
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        with Image.open(self.root/r['relative_path']) as im:image=self.transform(im.convert('RGB'))
        return image,self.mapping[r['label']]
def build(name,classes,project,pretrained=True):
    if name.startswith('phytoedge'):
        variant={'phytoedge_head_only':'head_only','phytoedge_attention':'attention','phytoedge_fusion':'fusion','phytoedge_net':'full'}[name]
        return PhytoEdgeNet(classes,pretrained,variant)
    if name=='yolo26n':
        from ultralytics import YOLO
        net=YOLO(str(project/'research_results/yolo26/pretrained/yolo26n-cls.pt')).model.float()
        layer=net.model[-1];layer.linear=nn.Linear(layer.linear.in_features,classes)
        return net.requires_grad_(True)
    ctor=getattr(models,name);net=ctor(weights='DEFAULT' if pretrained else None)
    net.classifier[-1]=nn.Linear(net.classifier[-1].in_features,classes);return net
def evaluate(net,loader,device):
    net.eval();truth=[];parts=[]
    with torch.inference_mode():
        for x,y in loader:
            out=net(x.to(device));out=out[-1] if isinstance(out,(tuple,list)) else out
            parts.append(out.float().cpu().numpy());truth.extend(y.tolist())
    logits=np.concatenate(parts);y=np.asarray(truth)
    return logits,y,float(f1_score(y,logits.argmax(1),labels=range(147),average='macro',zero_division=0))
def main(a):
    a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    with a.manifest.open(newline='') as f:rows=list(csv.DictReader(f))
    mapping=json.loads((a.project/'output/Leaf_Dataset_Combined/training_disease/class_to_idx.json').read_text())
    root=a.project/'output/Leaf_Dataset_Combined/training_disease'
    splits={s:[r for r in rows if r['clean_split']==s] for s in ['train','val','calibration','test']}
    group_splits={}
    for r in rows:group_splits.setdefault(r['duplicate_group'],set()).add(r['clean_split'])
    assert all(len(s)==1 for s in group_splits.values())
    protocol={'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),'seeds':[42,43,44],'models':MODELS,'epochs':a.epochs,'optimizer':'AdamW','lr':.0001,'weight_decay':.0001,'batch_size':a.batch_size,'augmentation':'random resized crop224 scale .8-1 and horizontal flip','frozen_layers':False,'checkpoint_selection':'maximum validation macro F1, first epoch on tie','early_stopping':'disabled; same fixed epoch cap for every run','ema':False,'amp':True,'eval_precision':'float32, TF32 disabled','normalization':'ImageNet mean/std except YOLO native RGB[0,1]','split_counts':{k:len(v) for k,v in splits.items()},'class_to_idx':mapping,'scope':'All five trained in this loop; no Ultralytics trainer/optimizer/EMA. Pretrained backbone initializations differ by family. Four ablations share dual-pool head; only attention and fusion vary.'}
    pf=a.output/'protocol.json'
    if pf.exists() and json.loads(pf.read_text())!=protocol:raise ValueError('Existing protocol differs')
    pf.write_text(json.dumps(protocol,indent=2))
    device='cuda'
    for seed in [42,43,44]:
      for name in MODELS:
        out=a.output/f'{name}_seed{seed}'
        if (out/'complete.json').exists():continue
        out.mkdir(exist_ok=True)
        with (out/'runtime_events.jsonl').open('a') as runtime_log:runtime_log.write(json.dumps({'started_unix':time.time(),'workers':a.workers,'prefetch_factor':1,'resume_requested':a.resume})+'\n')
        if (out/'last.pt').exists() and not a.resume:raise RuntimeError('Use --resume for an interrupted run')
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        net=build(name,147,a.project).to(device).requires_grad_(True)
        assert all(p.requires_grad for p in net.parameters()),'All backbones must be unfrozen'
        optimizer=torch.optim.AdamW(net.parameters(),lr=.0001,weight_decay=.0001)
        scaler=torch.amp.GradScaler('cuda')
        gen=torch.Generator().manual_seed(seed)
        train=DataLoader(Images(root,splits['train'],mapping,True,name),batch_size=a.batch_size,shuffle=True,generator=gen,num_workers=a.workers,prefetch_factor=1,pin_memory=True,persistent_workers=True)
        val=DataLoader(Images(root,splits['val'],mapping,False,name),batch_size=a.batch_size,num_workers=a.workers,prefetch_factor=1,pin_memory=True,persistent_workers=True)
        best=-1.;history=[];started=time.time();first_epoch=1
        if (out/'last.pt').exists():
            saved=torch.load(out/'last.pt',weights_only=False,map_location='cpu')
            net.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
            best=saved['best'];history=saved['history'];first_epoch=saved['epoch']+1
            torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng']);gen.set_state(saved['sampler_rng'])
            print('Resuming at epoch boundary; worker augmentation RNG restarts are recorded',name,seed,first_epoch,flush=True)
            (out/'resume.json').write_text(json.dumps({'first_resumed_epoch':first_epoch,'augmentation_rng_note':'DataLoader worker RNG restarted; not bit-exact continuation'},indent=2))
        for epoch in range(first_epoch,a.epochs+1):
            net.train();total=0.;n=0;epstart=time.time()
            for step,(x,y) in enumerate(train,1):
                x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True);optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    output=net(x);logits=output[-1] if isinstance(output,(tuple,list)) else output
                    loss=nn.functional.cross_entropy(logits,y)
                scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
                total+=float(loss.detach())*len(y);n+=len(y)
                if step%200==0:
                    status={'model':name,'seed':seed,'epoch':epoch,'step':step,'steps':len(train),'elapsed_epoch_seconds':time.time()-epstart,'training_loss':total/n}
                    (a.output/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
            _,_,score=evaluate(net,val,device)
            history.append({'epoch':epoch,'train_loss':total/n,'val_macro_f1':score,'seconds':time.time()-epstart})
            if score>best:
                best=score;torch.save({'state_dict':net.state_dict(),'model':name,'seed':seed,'epoch':epoch,'class_to_idx':mapping},out/'best.pt')
            torch.save({'epoch':epoch,'state_dict':net.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'best':best,'history':history,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'sampler_rng':gen.get_state()},out/'last.pt')
            (out/'history.json').write_text(json.dumps(history,indent=2));print(name,seed,history[-1],flush=True)
        net.load_state_dict(torch.load(out/'best.pt',weights_only=True)['state_dict'])
        for split in ['calibration','test']:
            loader=DataLoader(Images(root,splits[split],mapping,False,name),batch_size=a.batch_size,num_workers=a.workers,prefetch_factor=1,pin_memory=True)
            logits,y,f1=evaluate(net,loader,device)
            np.savez_compressed(out/f'{split}_logits.npz',logits=logits,true_indices=y,image_sha256=np.asarray([r['image_sha256'] for r in splits[split]]))
            (out/f'{split}_metrics.json').write_text(json.dumps({'accuracy':float((logits.argmax(1)==y).mean()),'macro_f1':f1,'images':len(y)},indent=2))
        (out/'complete.json').write_text(json.dumps({'complete':True,'seconds':time.time()-started},indent=2))
        del net,train,val;torch.cuda.empty_cache()
    (a.output/'complete.json').write_text(json.dumps({'complete':True,'runs':24},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--workers',type=int,default=2)
    main(p.parse_args())
