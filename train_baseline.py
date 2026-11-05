"""Step 1 single-GPU baseline using the downloaded StyTr2 authors' model."""
from pathlib import Path
from model_assets import ensure_asset
from functools import partial
import argparse
import csv
import json
import random
import sys
import time

import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.utils import save_image

root = Path(__file__).resolve().parent
repo = root / "StyTR-2-main"
sys.path.insert(0, str(repo))
from models import StyTR, transformer

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=512)
parser.add_argument("--output", type=Path, default=root / "step1_results")
parser.add_argument("--crop256", action="store_true", help="Resize to 512 square then randomly crop 256 during training")
parser.add_argument("--content-weight", type=float, default=7.0)
parser.add_argument("--local-style-weight", type=float, default=0.0)
parser.add_argument("--initial-state", type=Path)
args = parser.parse_args()
args.output.mkdir(exist_ok=True, parents=True)
torch.manual_seed(9444)
rng = random.Random(9444)
torch.set_float32_matmul_precision("high")
data = root / "datasets/baseline"


def files(kind, split):
    return sorted((data / kind / split).glob("*.jpg"))


crop_rng = random.Random(9444)


def read(path, training=False):
    with Image.open(path) as im:
        size = (512 if training else 256) if args.crop256 else 128
        im = im.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        if training and args.crop256:
            x, y = crop_rng.randint(0, 256), crop_rng.randint(0, 256)
            im = im.crop((x, y, x+256, y+256))
        return ToTensor()(im)


def shuffled_images(kind):
    paths = files(kind, "train")
    while True:
        rng.shuffle(paths)
        for path in paths:
            yield read(path, training=True)


ensure_asset('vgg')
StyTR.vgg.load_state_dict(torch.load(repo / "experiments/vgg_normalised.pth", map_location="cpu", weights_only=True))
net = StyTR.StyTrans(StyTR.vgg, StyTR.decoder, StyTR.PatchEmbed(), transformer.Transformer(), None).cuda()
if args.initial_state:
    if args.initial_state.exists():
        net.load_state_dict(torch.load(args.initial_state, map_location="cpu", weights_only=True))
    else:
        torch.save(net.state_dict(), args.initial_state)
if args.crop256:
    for module in net.modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            module.forward = partial(module.forward, need_weights=False)
opt = torch.optim.Adam((p for p in net.parameters() if p.requires_grad), lr=0.00005)
content = shuffled_images("content")
style = shuffled_images("style")
start = time.perf_counter()
with (args.output / "train.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["step","content","style","identity_pixel","identity_feature","total","seconds","local","weighted_local"])
    for step in range(1, args.steps + 1):
        c = torch.stack([next(content), next(content)]).cuda()
        s = torch.stack([next(style), next(style)]).cuda()
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=args.crop256):
            outputs = net(c, s, return_local_loss=args.local_style_weight > 0)
        _, lc, ls, lp, lf = outputs[:5]
        loss = args.content_weight*lc + 10*ls + 70*lp + lf
        local = outputs[5] if args.local_style_weight > 0 else None
        if local is not None:
            loss = loss + args.local_style_weight * local
        loss.backward()
        opt.step()
        values = [v.item() for v in (lc, ls, lp, lf, loss)]
        elapsed = time.perf_counter()-start
        writer.writerow([step, *values, elapsed, local.item() if local is not None else "",
                         args.local_style_weight * local.item() if local is not None else 0])
        if step in (1,10) or step % 32 == 0:
            handle.flush()
            print(step, values, elapsed, flush=True)
        if args.crop256 and step in (1000, 2500):
            torch.save({k:v for k,v in net.state_dict().items() if not k.startswith("enc_")}, args.output / f"step_{step}.pth")
torch.save({k:v for k,v in net.state_dict().items() if not k.startswith("enc_")}, args.output / "stytr2.pth")
net.eval()
summary = {"local_style_weight":args.local_style_weight,"content_weight":args.content_weight,"steps":args.steps,"train_seconds":elapsed}
with torch.inference_mode(), (args.output/"evaluation.csv").open("w",newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["split","content_file","style_file","content_loss","style_loss","local_loss"])
    for split in ("val","test"):
        styles = files("style",split)
        rows, pictures = [], []
        for i, cp in enumerate(files("content",split)):
            sp = styles[i % len(styles)]
            c,s = read(cp)[None].cuda(),read(sp)[None].cuda()
            out,lc,ls,_,_,local = net(c,s,return_local_loss=True)
            rows.append([lc.item(),ls.item(),local.item()])
            writer.writerow([split,cp.name,sp.name,*rows[-1]])
            pair_dir = args.output / "pairs" / split
            pair_dir.mkdir(parents=True, exist_ok=True)
            save_image(torch.cat([c,s,out.clamp(0,1)]),pair_dir / f"{cp.stem}_{sp.stem}.png",nrow=3)
            if i < 8:
                pictures.extend([c[0],s[0],out[0].clamp(0,1)])
        summary[split] = dict(zip(["content_loss","style_loss","local_loss"],torch.tensor(rows).mean(0).tolist()))
        save_image(torch.stack(pictures), args.output / (split+".jpg"), nrow=3)
print(summary)
(args.output/"summary.json").write_text(json.dumps(summary,indent=2))
