"""Substep API: independent old equations, phase composition, resets and TBPTT."""
import unittest
import torch
from torch.utils.checkpoint import checkpoint
from fly_core import Core, Spike


def make_core():
    return Core(torch.tensor([0,1,2,3,0,4,4]), torch.tensor([1,2,3,0,4,3,0]),
                torch.full((7,), .3, dtype=torch.double),
                torch.tensor([1,1,-1,1,-1.], dtype=torch.double), 6, 3)


def old_equations(core, current, steps, state):
    v, s = state
    total = torch.zeros_like(v)
    w = core.weights()
    for _ in range(steps):
        syn = torch.zeros_like(s).index_add(1, core.dst, s[:, core.src] * w)
        u = .95*v + syn + current
        s = Spike.apply(u)
        v = u*(1-s)
        total = total+s
    return (v,s), total/steps


class CoreAPITests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(77)

    def test_equations_and_phase_composition_gradients(self):
        base = torch.rand(2,6,dtype=torch.double)*1.6
        expected = None
        for mode in ('old', 'advance', 'phases', 'checkpoint'):
            m=make_core(); x=base.clone().requires_grad_(); state=m.initial_state(2)
            if mode == 'old':
                state, rate=old_equations(m,x,8,state)
            elif mode == 'phases':
                w=m.weights()
                state,r1=m.advance(x,3,state,weights=w)
                state,r2=m.advance(x,5,state,weights=w)
                rate=(3*r1+5*r2)/8
            elif mode == 'checkpoint':
                def segment(v,s,c,w):
                    (v,s),r=m.advance(c,8,(v,s),weights=w)
                    return v,s,r
                v,s,rate=checkpoint(segment,*state,x,m.weights(),use_reentrant=False)
                state=(v,s)
            else:
                state,rate=m.advance(x,8,state)
            loss=state[0].square().mean()+rate.square().sum()+state[1].sum()*.03
            loss.backward()
            actual=[*state,rate,x.grad,m.raw.grad]
            self.assertGreater(m.raw.grad.abs().max().item(),0)
            if expected is None: expected=[a.detach().clone() for a in actual]
            else:
                for a,b in zip(actual,expected): torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-12)

    def test_reset_freeze_and_gradients(self):
        m=make_core(); x=torch.rand(3,6,dtype=torch.double,requires_grad=True)
        v=torch.rand(3,6,dtype=torch.double,requires_grad=True)
        s=torch.zeros_like(v,requires_grad=True)
        before=[a.detach().clone() for a in (v,s)]
        active=torch.tensor([True,False,True]); reset=torch.tensor([True,False,False])
        state,rate=m.advance(x,4,(v,s),active=active,reset=reset)
        fresh,r=m.advance(x[:1],4)
        for a,b,old in zip(state,(v,s),before):
            torch.testing.assert_close(a[1],b[1])
            torch.testing.assert_close(b,old)
        for a,b in zip(state,fresh): torch.testing.assert_close(a[:1],b)
        torch.testing.assert_close(rate[:1],r)
        self.assertEqual(rate[1].abs().sum().item(),0)
        (state[0].sum()+state[1].sum()+rate.sum()).backward()
        self.assertEqual(x.grad[1].abs().sum().item(),0)
        self.assertEqual(v.grad[0].abs().sum().item(),0)
        self.assertEqual(s.grad[0].abs().sum().item(),0)
        torch.testing.assert_close(v.grad[1],torch.ones(6,dtype=torch.double))

    def test_detach_and_changed_current_between_phases(self):
        m=make_core(); x=torch.rand(2,6,dtype=torch.double,requires_grad=True)
        state,_=m.advance(x,3)
        carried=m.detach_state(state)
        for a,b in zip(carried,state):
            torch.testing.assert_close(a,b)
            self.assertFalse(a.requires_grad)
        y=torch.full_like(x,.8,requires_grad=True)
        final,rate=m.advance(y,5,carried)
        (final[0].sum()+rate.sum()).backward()
        self.assertIsNone(x.grad)
        self.assertGreater(y.grad.abs().max().item(),0)
        self.assertTrue(torch.isfinite(m.raw.grad).all())

    def test_invalid_contract(self):
        m=make_core(); x=torch.zeros(2,6,dtype=torch.double)
        for kwargs in ({'steps':0},{'active':torch.ones(2)}, {'reset':torch.tensor([True])},
                       {'weights':torch.zeros(1,dtype=torch.double)}):
            with self.assertRaises(ValueError): m.advance(x,**kwargs)
        with self.assertRaises(ValueError): m.advance(x.float())


if __name__ == '__main__': unittest.main(verbosity=2)
