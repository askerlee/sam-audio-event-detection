# make_sb.py
import modal
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default="B200")
parser.add_argument("--image_id", type=str, default="im-QmP3Htx6h1f9BvIIlriEDM")
args = parser.parse_args()

if args.image_id is not None:
    image = modal.Image.from_id(args.image_id)
    print("Using provided image:", args.image_id)
else:
    cuda_version = "12.8.1"  # should be no greater than host CUDA version
    flavor = "devel"  # includes full CUDA toolkit
    operating_sys = "ubuntu24.04"
    tag = f"{cuda_version}-{flavor}-{operating_sys}"
    HF_CACHE_PATH = "/cache"

    image = modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    '''
    image = (
        modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
        .entrypoint([])  # remove verbose logging by base image on entry
        .apt_install(["fish", "clang", "libopenmpi-dev"])  # required for tensorrt
        .pip_install("tensorrt-llm==0.19.0", "pynvml", extra_index_url="https://pypi.nvidia.com")
        .pip_install("hf-transfer", "huggingface_hub[hf_xet]")
        .env({"HF_HUB_CACHE": HF_CACHE_PATH, "HF_HUB_ENABLE_HF_TRANSFER": "1", "PMIX_MCA_gds": "hash"})    
    )
    '''

app_name = f"sam-audio-{args.gpu.lower()}"
app = modal.App.lookup(app_name, create_if_missing=True)
hf_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
sam_vol = modal.Volume.from_name("sam-audio", create_if_missing=True)

with modal.enable_output():
    sb = modal.Sandbox.create(
        "bash", "-lc", "sleep infinity",
        app=app,
        image=image,
        gpu=args.gpu,
        timeout=6 * 60 * 60,
        idle_timeout=6 * 60 * 60,
        volumes={"/hf_cache": hf_vol, "/sam_audio": sam_vol},
    )

print(sb.object_id)  # sb-...
