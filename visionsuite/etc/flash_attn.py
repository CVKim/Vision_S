import torch
import torch.nn.functional as F
import time
import torch.profiler 

# GPU 사용 강제
device = 'cuda'
dtype = torch.float16  # FlashAttention은 float16/bfloat16에서 더 빠름

# 입력 텐서 정의 (배치 크기, 헤드 수, 시퀀스 길이, 임베딩 차원)
B, H, L, D = 8, 12, 512, 64  # B: batch size, H: heads, L: seq length, D: dim per head

q = torch.randn(B, H, L, D, device=device, dtype=dtype)
k = torch.randn(B, H, L, D, device=device, dtype=dtype)
v = torch.randn(B, H, L, D, device=device, dtype=dtype)

# Warmup
for _ in range(10):
    _ = F.scaled_dot_product_attention(q, k, v, is_causal=False)

# FlashAttention 사용 (PyTorch가 내부적으로 적용, CUDA + float16 필요)
start = time.time()
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True
) as prof:
    for _ in range(50):
        out_flash = F.scaled_dot_product_attention(q, k, v, is_causal=False)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
torch.cuda.synchronize()
end = time.time()
print(f"[FlashAttention 사용] 평균 소요 시간: {(end - start)/50:.6f}초")

# 일반 Attention 함수 직접 구현 (softmax → matmul 방식)
def standard_attention(q, k, v):
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
    attn_probs = torch.softmax(attn_scores, dim=-1)
    return torch.matmul(attn_probs, v)

# Warmup
for _ in range(10):
    _ = standard_attention(q, k, v)

start = time.time()
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True
) as prof:
    for _ in range(50):
        out_standard = standard_attention(q, k, v)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
torch.cuda.synchronize()
end = time.time()
print(f"[FlashAttention 미사용] 평균 소요 시간: {(end - start)/50:.6f}초")

# 결과 비교
diff = (out_flash - out_standard).abs().max()
print(f"출력 차이 (max abs diff): {diff.item():.6f}")
