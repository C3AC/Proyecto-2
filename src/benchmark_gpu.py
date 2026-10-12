# Benchmark de entrenamiento en la GTX 1650 (datos sintéticos, batch 32, 224 px). Resultado en results/gpu_benchmark.csv
import time, torch, timm, sys
torch.backends.cudnn.benchmark = True
def bench(arch, amp, cl, bs=32, steps=8):
    m = timm.create_model(arch, pretrained=False, num_classes=6).cuda()
    x = torch.randn(bs, 3, 224, 224, device="cuda"); y = torch.randint(0, 6, (bs,), device="cuda")
    if cl: m = m.to(memory_format=torch.channels_last); x = x.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(m.parameters()); sc = torch.amp.GradScaler(enabled=amp)
    for i in range(steps + 3):
        if i == 3: torch.cuda.synchronize(); t = time.time()
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = torch.nn.functional.cross_entropy(m(x), y)
        opt.zero_grad(); sc.scale(loss).backward(); sc.step(opt); sc.update()
    torch.cuda.synchronize()
    print(f"{arch:16s} amp={amp!s:5} channels_last={cl!s:5} {bs*steps/(time.time()-t):6.1f} img/s", flush=True)
    del m, opt; torch.cuda.empty_cache()
for arch in ("efficientnet_b0", "convnext_tiny"):
    for amp, cl in ((False, False), (True, False), (True, True)):
        bench(arch, amp, cl)
