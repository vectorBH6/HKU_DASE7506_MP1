"""Training loop for MP1 with EMA and cosine LR schedule."""
import argparse
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score
import copy  # 用于深拷贝出 EMA 影子模型


def main():
    total_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', default='student')
    p.add_argument('--config', type=Path, default=ROOT/'configs/baseline.json')
    p.add_argument('--run-dir', type=Path, default=ROOT/'runs/baseline-s17')
    # 默认值改为 GPU、长训练、并每隔 400 步做一次中间验证
    p.add_argument('--device', default='cuda')
    p.add_argument('--precision', choices=['auto','fp32','bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=24000)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--eval-every', type=int, default=400,
                   help='Optional validation-curve interval; 0 evaluates only after training.')
    # 新增可配置项：学习率 / warmup 步数 / EMA 衰减（从硬编码改为命令行参数）
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--warmup', type=int, default=0, help='0=steps//10 auto')
    p.add_argument('--ema-decay', type=float, default=0.99)
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        p.error('Batch size and step count must be positive.')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Run directory already contains results. Use a new --run-dir.')
    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()
    data = load_data()
    config = json.loads(args.config.read_text())
    model, implementation_sha = make_model(args.implementation, config, device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    # 参数分组：矩阵权重走 weight_decay，bias / LayerNorm 等不衰减
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{'params': decay, 'weight_decay': .1}, {'params': no_decay, 'weight_decay': 0.}],
        lr=args.lr, betas=(0.9, 0.95))
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter() - prepared
    started = time.perf_counter()
    history = []
    validation_history = []
    intermediate_validation_seconds = 0.
    # 跟踪训练过程中出现的最低验证 bpb（用于保存最佳 checkpoint）
    best = {'bpb': float('inf'), 'step': 0}
    # EMA 影子模型：深拷贝主模型并冻结梯度，每步用主模型参数做指数移动平均
    ema_model = copy.deepcopy(model)
    for pe in ema_model.parameters():
        pe.requires_grad_(False)
    warmup = args.warmup or max(100, args.steps // 10)
    for step in range(args.steps):
        starts = torch.randint(len(tokens)-257, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:, None] + torch.arange(257, device=device)]
        # 学习率：前 warmup 步线性升到峰值，之后余弦衰减到峰值的 10%
        learning_rate = args.lr * min(1., (step+1)/warmup) * (.1 + .9*.5*(1+math.cos(math.pi*step/args.steps)))
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            loss = F.cross_entropy(model(batch[:, :-1]).flatten(0, 1).float(), batch[:, 1:].flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        # EMA 更新：pe = decay * pe + (1 - decay) * pm
        with torch.no_grad():
            for pe, pm in zip(ema_model.parameters(), model.parameters()):
                pe.mul_(args.ema_decay).add_(pm, alpha=1-args.ema_decay)
        if (step+1) % 100 == 0 or step+1 == args.steps:
            row = {'step': step+1, 'loss': loss.item(),
                   'seconds': time.perf_counter()-started-intermediate_validation_seconds}
            history.append(row)
            print(json.dumps(row), flush=True)
        # 中间验证用 EMA 模型（更平滑，泛化更好）
        if args.eval_every > 0 and (step+1) % args.eval_every == 0:
            intermediate = score(ema_model, *data['validation'], device, 'fp32')
            intermediate.pop('window_nll_nats')
            intermediate_validation_seconds += intermediate['seconds']
            validation_history.append({'step': step+1, **intermediate})
            print(json.dumps({'validation': validation_history[-1]}), flush=True)
            # 验证 bpb 创新低 -> 把 EMA 权重快照到 checkpoint_best.pt
            if intermediate['bpb'] < best['bpb']:
                best = {'bpb': intermediate['bpb'], 'step': step+1}
                snapshot = {k: v.detach().cpu().clone() for k, v in ema_model.state_dict().items()}
                torch.save({'protocol': PROTOCOL, 'implementation': args.implementation, 'config': config,
                            'model': snapshot, 'seed': args.seed, 'ema_decay': args.ema_decay,
                            'selected_step': step+1, 'train_tokens': (step+1)*args.batch_size*256},
                           args.run_dir/'checkpoint_best.pt')
                print(json.dumps({'new_best': best}), flush=True)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started - intermediate_validation_seconds
    # 最终验证用 EMA 模型；同时也用原始模型验证一次，便于对比 EMA 的收益
    validation = score(ema_model, *data['validation'], device, 'fp32')
    validation_raw = score(model, *data['validation'], device, 'fp32')
    validation_raw.pop('window_nll_nats')
    validation.pop('window_nll_nats')
    # 若训练末步的 EMA bpb 仍低于历史最佳，再覆盖写 best_checkpoint.pt
    if validation['bpb'] < best['bpb']:
        best = {'bpb': validation['bpb'], 'step': args.steps}
        torch.save({'protocol': PROTOCOL, 'implementation': args.implementation, 'config': config,
                    'model': {k: v.detach().cpu().clone() for k, v in ema_model.state_dict().items()},
                    'seed': args.seed, 'ema_decay': args.ema_decay, 'selected_step': args.steps,
                    'train_tokens': args.steps*args.batch_size*256},
                   args.run_dir/'best_checkpoint.pt')
    checkpoint = args.run_dir/'checkpoint.pt'
    # 最终 checkpoint 存 EMA 权重（而非原始模型），用于获得最佳验证表现
    torch.save({'protocol': PROTOCOL, 'implementation': args.implementation, 'config': config,
                'model': ema_model.cpu().state_dict(), 'seed': args.seed,
                'train_tokens': args.steps*args.batch_size*256}, checkpoint)
    result = {'protocol': PROTOCOL, 'implementation': args.implementation, 'config': config, 'seed': args.seed,
              'parameters': sum(p.numel() for p in model.parameters()), 'precision': precision,
              'train_tokens': args.steps*args.batch_size*256, 'preparation_seconds': preparation_seconds,
              'train_seconds': train_seconds, 'validation': validation, 'history': history,
              'validation_history': validation_history, 'best_validation': best,
              'intermediate_validation_seconds': intermediate_validation_seconds,
              'process_seconds': time.perf_counter()-total_started, 'validation_raw': validation_raw,
              'torch_version': str(torch.__version__), 'threads': args.threads,
              'checkpoint_sha256': sha(checkpoint), 'implementation_sha256': implementation_sha,
              **device_metrics(device)}
    (args.run_dir/'metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result | {'history': []}, indent=2), flush=True)


if __name__ == '__main__':
    main()