"""Evaluate the four frozen models on all content and style combinations."""
import argparse
import csv
import json
from functools import partial
from pathlib import Path
from model_assets import ensure_asset
import sys

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

root = Path(__file__).resolve().parent
repo = root / "StyTR-2-main"
sys.path.insert(0, str(repo))
from models import StyTR, transformer

parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+", choices=("A","B","control","C"), default=["A","B"])
parser.add_argument("--output", type=Path, default=root / "evaluation_full_grid")
args = parser.parse_args()
out = args.output
out.mkdir(exist_ok=True)
models = {"A":"step1_256_5000","B":"step2_content10","control":"step2_local/control","C":"step2_local/experiment_c"}
models = {name: models[name] for name in args.models}
for name in models:
    if name in ("A","B"):
        ensure_asset(name)
ensure_asset('vgg')
StyTR.vgg.load_state_dict(torch.load(repo/"experiments/vgg_normalised.pth",weights_only=True))
net = StyTR.StyTrans(StyTR.vgg,StyTR.decoder,StyTR.PatchEmbed(),transformer.Transformer(),None).cuda().eval()
for module in net.modules():
    if isinstance(module,torch.nn.MultiheadAttention):
        module.forward = partial(module.forward,need_weights=False)
torch.set_float32_matmul_precision("high")

def images(kind,split):
    result = {}
    for path in sorted((root/"datasets/baseline"/kind/split).glob("*.jpg")):
        with Image.open(path) as im:
            result[path.name] = to_tensor(im.convert("RGB").resize((256,256),Image.Resampling.BILINEAR))[None].cuda()
    return result

metrics = ("content_loss","style_loss","local_loss")
summary = {}
with torch.inference_mode(), (out/"pairs.csv").open("w",newline="") as f, (out/"by_style.csv").open("w",newline="") as g:
    pairs,styles = csv.writer(f),csv.writer(g)
    pairs.writerow(["model","split","content","style",*metrics])
    styles.writerow(["model","split","style",*metrics])
    for name,folder in models.items():
        missing,extra=net.load_state_dict(torch.load(root/folder/"stytr2.pth",map_location="cpu",weights_only=True),strict=False)
        assert not extra and all(key.startswith("enc_") for key in missing)
        summary[name]={}
        for split in ("test","val"):
            content,refs=images("content",split),images("style",split)
            assert (len(content),len(refs))==(32,12)
            scores=[]
            for ref,s in refs.items():
                rows=[]
                for photo,c in content.items():
                    _,lc,ls,_,_,ll=net(c,s,return_local_loss=True)
                    row=[v.item() for v in (lc,ls,ll)]
                    rows.append(row)
                    pairs.writerow([name,split,photo,ref,*row])
                scores.extend(rows)
                styles.writerow([name,split,ref,*[sum(x)/len(x) for x in zip(*rows)]])
            summary[name][split]=dict(zip(metrics,[sum(x)/len(x) for x in zip(*scores)]))
            f.flush()
            print(name,split,summary[name][split],flush=True)
(out/"summary.json").write_text(json.dumps(summary,indent=2))
