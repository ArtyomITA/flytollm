"""GPU numerical tests; run through verify_phase23.py watchdog."""
import unittest
import torch
from torch import nn
from fly_attention import AttentionConfig, AttentionCache, CausalAttention, MathSDPA


class ManualAttention(nn.Module):
    def forward(self,q,k,v,mask):
        scores=(q@k.transpose(-1,-2))/q.shape[-1]**.5
        return scores.masked_fill(~mask,float('-inf')).softmax(-1)@v


def model(window=3):
    return CausalAttention(AttentionConfig(dim=16,heads=2,window=window)).cuda()


class AttentionTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(109)

    def test_sdpa_against_manual_forward_backward(self):
        q=torch.randn(2,2,1,8,device='cuda',requires_grad=True)
        k=torch.randn(2,2,5,8,device='cuda',requires_grad=True)
        v=torch.randn_like(k,requires_grad=True)
        mask=torch.tensor([[1,0,1,0,1],[0,1,0,0,0]],device='cuda',dtype=torch.bool)[:,None,None,:]
        actual=MathSDPA()(q,k,v,mask);expected=ManualAttention()(q,k,v,mask)
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
        ga=torch.autograd.grad(actual.square().sum(),(q,k,v),retain_graph=True)
        ge=torch.autograd.grad(expected.square().sum(),(q,k,v))
        for a,e in zip(ga,ge):torch.testing.assert_close(a,e,atol=2e-6,rtol=2e-5)

    def test_rope_complex_reference_and_relative_position(self):
        m=model();x=torch.randn(2,2,1,8,device='cuda')
        positions=torch.tensor([0,131],device='cuda')
        complex_x=torch.view_as_complex(x.reshape(2,2,1,4,2))
        angle=positions[:,None,None,None]*10000.0**(-torch.arange(4,device='cuda')/4)
        expected=torch.view_as_real(complex_x*torch.polar(torch.ones_like(angle),angle)).flatten(-2)
        torch.testing.assert_close(m.rotate(x,positions),expected)
        y=torch.randn_like(x);a=m.rotate(x,positions);b=m.rotate(y,positions+7)
        torch.testing.assert_close((a*b).sum(-1),(x*m.rotate(y,torch.full_like(positions,7))).sum(-1),atol=3e-5,rtol=3e-5)

    def test_functional_eviction_padding_reset_and_detach(self):
        m=model();cache=m.empty(2);snapshot=[x.clone() for x in cache]
        x=torch.randn(2,16,device='cuda',requires_grad=True)
        active=torch.tensor([True,False],device='cuda')
        updated=m.append(x,cache,active)
        for old,saved,new in zip(cache,snapshot,updated):
            torch.testing.assert_close(old,saved)
            torch.testing.assert_close(new[1],old[1])
        for _ in range(4):updated=m.append(x,updated,active)
        torch.testing.assert_close(updated.positions[0],torch.tensor([2,3,4],device='cuda'))
        self.assertEqual(updated.next_position.tolist(),[5,0])
        detached=m.detach(updated)
        for old,new in zip(updated,detached):
            torch.testing.assert_close(old,new);self.assertEqual(old.data_ptr(),new.data_ptr())
            self.assertIsNone(new.grad_fn)
        reset=m.reset(updated,active)
        self.assertFalse(reset.valid.any());self.assertEqual(reset.next_position.sum().item(),0)
        reset.k.sum().backward();self.assertEqual(x.grad.abs().sum().item(),0)

    def test_empty_inactive_future_and_invalid_slots(self):
        m=model();q=torch.randn(2,16,device='cuda',requires_grad=True)
        cache=m.empty(2);out=m.read(q,cache)
        self.assertEqual(out.abs().sum().item(),0)
        out.sum().backward();self.assertTrue(torch.isfinite(q.grad).all())
        cache=m.append(torch.randn_like(q),cache)
        inactive=torch.zeros(2,device='cuda',dtype=torch.bool)
        self.assertEqual(m.read(q,cache,inactive).abs().sum().item(),0)
        future=cache._replace(positions=cache.next_position[:,None].expand_as(cache.positions))
        self.assertEqual(m.read(q,future).abs().sum().item(),0)
        # Corrupt invalid slots: neither output nor valid-slot gradients may depend on them.
        poison=cache._replace(k=torch.where(cache.valid[:,None,:,None],cache.k,torch.full_like(cache.k,float('nan'))),
                              v=torch.where(cache.valid[:,None,:,None],cache.v,torch.full_like(cache.v,float('nan'))))
        torch.testing.assert_close(m.read(q,poison),m.read(q,cache))

    def test_prefix_causality_and_memory_gradient(self):
        m=model();x=torch.randn(6,2,16,device='cuda',requires_grad=True)
        def run(sequence):
            cache=m.empty(2);outputs=[]
            for final in sequence:
                outputs.append(m.read(final,cache));cache=m.append(final,cache)
            return torch.stack(outputs)
        output=run(x);changed=x.detach().clone();changed[3:]*=-5
        torch.testing.assert_close(output[:3],run(changed)[:3])
        output[2].square().sum().backward()
        self.assertGreater(x.grad[:2].abs().sum().item(),0)
        self.assertEqual(x.grad[3:].abs().sum().item(),0)
        for name,p in m.named_parameters():
            self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)
            self.assertGreater(p.grad.abs().max().item(),0,name)
        old=torch.randn(2,16,device='cuda',requires_grad=True)
        cache=m.detach(m.append(old,m.empty(2)))
        grad=torch.autograd.grad(m.read(x[0],cache).sum(),old,allow_unused=True)[0]
        self.assertIsNone(grad)

    def test_full_window_absolute_positions(self):
        m=model(window=128)
        with torch.no_grad():
            cache=m.empty(1);x=torch.randn(1,16,device='cuda')
            for _ in range(131):cache=m.append(x,cache)
            torch.testing.assert_close(cache.positions[0],torch.arange(3,131,device='cuda'))
            self.assertTrue(cache.valid.all());self.assertEqual(cache.next_position.item(),131)
            torch.testing.assert_close(cache.k[:,:,-1:],m.rotate(m._heads(m.k(m.norm(x))),torch.tensor([130],device='cuda')))

    def test_backend_swap_and_serialization(self):
        a=model();b=CausalAttention(a.config,backend=ManualAttention()).cuda()
        b.load_state_dict(a.state_dict());x=torch.randn(2,16,device='cuda')
        cache=a.append(x,a.empty(2));cache=a.append(x*.2,cache)
        torch.testing.assert_close(a.read(x*.4,cache),b.read(x*.4,cache),atol=2e-6,rtol=2e-5)


if __name__=='__main__':unittest.main(verbosity=2)
