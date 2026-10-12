"""S8: repeated calls are bit-identical; CPU and GPU agree (GPU cases skip without CUDA)."""
import pytest
import torch

import aes_tensor
import cdc_problems as P

N, C = 500, 12
CASES = [("ces", 1.0), ("ces", -0.3), ("pairwise", 1), ("pairwise", -1)]


def build(family, setting):
    if family == "ces":
        data, params = P.make_ces(N, C, 40, setting, 0.8 if setting < 0 else 1.0)
        return data, params, P.ces_profit, P.ces_marginal, P.mode_for(setting)
    data, params = P.make_pairwise(N, C, 40, setting)
    return data, params, P.pairwise_profit, P.pairwise_marginal, ("complements" if setting > 0 else "substitutes")


@pytest.mark.parametrize("family, setting", CASES)
def test_repeated_calls_identical(family, setting):
    data, params, profit, marginal, mode = build(family, setting)
    for m in (None, marginal):
        first = aes_tensor.solve(profit, data, params, n_choices=C, mode=mode, marginal=m)
        second = aes_tensor.solve(profit, data, params, n_choices=C, mode=mode, marginal=m)
        assert torch.equal(first.menu, second.menu)
        assert torch.equal(first.value, second.value)
        assert first.stats == second.stats


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("family, setting", CASES)
def test_cpu_and_gpu_agree(family, setting):
    data, params, profit, marginal, mode = build(family, setting)
    value, menu, unique = P.brute_force(profit, data, params, C)
    gpu = torch.device("cuda")
    data_gpu = {k: v.to(gpu) for k, v in data.items()}
    params_gpu = {k: v.to(gpu) for k, v in params.items()}
    for m in (None, marginal):
        cpu = aes_tensor.solve(profit, data, params, n_choices=C, mode=mode, marginal=m)
        on_gpu = aes_tensor.solve(profit, data_gpu, params_gpu, n_choices=C, mode=mode, marginal=m)
        assert on_gpu.menu.device.type == "cuda"
        assert torch.equal(on_gpu.menu.cpu(), cpu.menu)
        assert P.close(on_gpu.value.cpu(), cpu.value).all()
        P.check(cpu, profit, data, params, value, menu, unique)
