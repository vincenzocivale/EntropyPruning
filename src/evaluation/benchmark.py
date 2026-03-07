import time
import torch
from fvcore.nn import FlopCountAnalysis


def benchmark_model(model, loader, device, n_warmup=10, label="model"):
    """Measure average inference time per image and FLOPs."""
    model.eval()

    # FLOPs on a single sample
    dummy = next(iter(loader))[0][:1].to(device)
    try:
        flops = FlopCountAnalysis(model, dummy)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        total_flops = flops.total()
    except Exception as e:
        print(f"FLOPs not computable: {e}")
        total_flops = None

    # GPU warmup
    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            model(imgs.to(device))
            if i >= n_warmup:
                break

    # Timing
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_imgs = 0
    with torch.no_grad():
        for imgs, _ in loader:
            model(imgs.to(device))
            torch.cuda.synchronize()
            n_imgs += len(imgs)
    t1 = time.perf_counter()

    ms_per_img = (t1 - t0) / n_imgs * 1000

    print(f"\n-- Benchmark: {label} --")
    print(f"  Average time per image: {ms_per_img:.2f} ms")
    if total_flops:
        print(f"  FLOPs per image:        {total_flops/1e9:.2f} GFLOPs")

    return {
        "ms_per_img": ms_per_img,
        "gflops": total_flops / 1e9 if total_flops else None,
    }
