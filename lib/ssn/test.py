import torch
from .pair_wise_distance import PairwiseDistFunction


# naive implementation for debug
def naive_pair_wise_dist(pix, spix, idx, n_spix_w, n_spix_h):
    device = pix.device
    ba, ch, pi = pix.shape
    outputs = []
    for b in range(ba):
        batch_out = []
        for p in range(pi):
            pix_out = []
            pix_v = pix[b, :, p]
            sp_i = idx[b, p]
            sp_i_x = sp_i % n_spix_w
            sp_i_y = sp_i // n_spix_w
            for i in range(9):
                if sp_i_x == 0 and (i % 3) == 0:
                    d_dist = pix.new(1).fill_(0)
                    pix_out.append(d_dist[0])
                elif sp_i_x == (n_spix_w - 1) and (i % 3) == 2:
                    d_dist = pix.new(1).fill_(0)
                    pix_out.append(d_dist[0])
                elif sp_i_y == 0 and (i // 3) == 0:
                    d_dist = pix.new(1).fill_(0)
                    pix_out.append(d_dist[0])
                elif sp_i_y == (n_spix_h - 1) and (i // 3) == 2:
                    d_dist = pix.new(1).fill_(0)
                    pix_out.append(d_dist[0])
                else:
                    offset_x = i % 3 - 1
                    offset_y = (i // 3 - 1) * n_spix_w
                    s = int(sp_i + offset_y + offset_x)
                    pix_out.append((pix_v - spix[b, :, s]).pow(2).sum())
            batch_out.append(torch.stack(pix_out))
        outputs.append(torch.stack(batch_out, 1))
    return torch.stack(outputs, 0)


def test(eps=1e-4):
    func = PairwiseDistFunction.apply

    pix = torch.randn(2, 20, 81).double().to("cuda")
    spix = torch.randn(2, 20, 9).double().to("cuda")
    idx = torch.randint(0, 9, (2, 81)).double().to("cuda")
    wid = 3
    hei = 3

    pix.requires_grad = True
    spix.requires_grad = True

    res = torch.autograd.gradcheck(func, (pix, spix, idx, wid, hei), eps=eps, raise_exception=False)
    print(res)

    o = PairwiseDistFunction.apply(pix, spix, idx, wid, hei)
    o.sum().backward()

    cuda_p_grad = pix.grad
    cuda_sp_grad = spix.grad

    pix.grad.zero_()
    spix.grad.zero_()

    naive_o = naive_pair_wise_dist(pix, spix, idx, wid, hei)
    naive_o.sum().backward()

    print("output diff between GPU and naive", torch.abs(o - naive_o).mean())
    print("pix grad diff between GPU and naive", torch.abs(cuda_p_grad - pix.grad).mean())
    print("spix grad diff between GPU and naive", torch.abs(cuda_sp_grad - spix.grad).mean())
