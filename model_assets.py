"""Download required model assets on first use."""
from pathlib import Path
import hashlib
import os
import tempfile
from urllib.request import urlopen

ASSETS = {'vgg': ('StyTR-2-main/experiments/vgg_normalised.pth', 'https://drive.usercontent.google.com/download?id=1BinnwM5AmIcVubr16tPTqxMjUCE8iu5M&export=download&confirm=t', '804ca2835ecf7539f0cd2a7ac3c18ce81e6f8468969ae7117ac0c148d286bb4a'), 'A': ('step1_256_5000/stytr2.pth', 'https://github.com/qcxyt/comp9444-style-transfer/releases/download/trained-models-v1/baseline.pth', '6c38b994d87bdacc2a5ec93e6659f16ddcecb2ee9a702980bd5de06c4541a912'), 'B': ('step2_content10/stytr2.pth', 'https://github.com/qcxyt/comp9444-style-transfer/releases/download/trained-models-v1/improvement1.pth', '4a21d3eed6cb585ed1191bc44c48c616deba5b5cd8da619778be57c23d6a4371')}

def ensure_asset(name):
    relative, url, expected = ASSETS[name]
    path = Path(__file__).resolve().parent / relative
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {name} weights...", flush=True)
    temporary = None
    try:
        digest = hashlib.sha256()
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            with urlopen(url, timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"Downloaded {name} weights failed verification")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
