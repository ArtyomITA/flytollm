"""CPU: exact surrogate reference, checkpoint, signs, stable IDs."""
import tempfile
import pathlib
import unittest
import torch
from fly_core import Core, LIFReset, Spike


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(18)

    def model(self):
        return Core(torch.tensor([0,1,2,3,0,4,4]), torch.tensor([1,2,3,0,4,3,0]),
                    torch.full((7,), .3, dtype=torch.double), torch.tensor([1,1,-1,1,-1.],dtype=torch.double), 6, 3)

    def test_reset(self):
        u = torch.tensor([-.4,.2,.99,1.,1.01,1.8],dtype=torch.double,requires_grad=True)
        gv, gs = torch.randn_like(u), torch.randn_like(u)
        v,s = LIFReset.apply(u)
        actual = torch.autograd.grad((v,s), u, (gv,gs))[0]
        sr=Spike.apply(u); vr=u*(1-sr)
        expected=torch.autograd.grad((vr,sr),u,(gv,gs))[0]
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)

    def test_window_and_checkpoint(self):
        drive = torch.rand(4,2,6,dtype=torch.double) * 1.6
        expected = None
        for reference, ck in [(True,0),(False,0),(False,2)]:
            m=self.model(); x=drive.clone().requires_grad_()
            loss,rate,state=m.window(x,steps=3,reference=reference,checkpoint_chars=ck)
            loss.backward()
            values=[loss.detach(), rate, *[v.detach() for v in state],m.raw.grad,x.grad]
            self.assertGreater(rate.item(),0)
            self.assertGreater(m.raw.grad.abs().max().item(),0)
            if expected is None: expected=values
            else:
                for a,b in zip(values,expected): torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-12)

    def test_dale(self):
        m=self.model()
        with torch.no_grad(): m.raw.copy_(torch.linspace(-80,80,7))
        self.assertTrue((m.weights()*m.signs >= 0).all())

    def test_saved_memory(self):
        sizes=[]
        for reference in [True,False]:
            saved={}
            def pack(t):
                storage=t.untyped_storage()
                saved[storage.data_ptr()]=(storage.nbytes(),t)
                return t
            m=self.model();drive=torch.rand(4,2,6,dtype=torch.double)
            with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
                loss,_,_=m.window(drive,steps=3,reference=reference)
            sizes.append(sum(size for size,t in saved.values()))
            loss.backward()
        self.assertLess(sizes[1],sizes[0]*.6)

    def test_carry_state(self):
        m=self.model();x=torch.rand(4,2,6,dtype=torch.double)
        full=m.window(x,steps=2)
        first=m.window(x[:2],steps=2)
        second=m.window(x[2:],steps=2,state=first[2])
        torch.testing.assert_close(full[0],(first[0]+second[0])/2)
        for a,b in zip(full[2],second[2]): torch.testing.assert_close(a,b)

    def test_fixed_ids(self):
        import pyarrow as pa
        import pyarrow.feather as pf
        import numpy as np
        from fly_graph import load_graph
        with tempfile.TemporaryDirectory() as td:
            p=pathlib.Path(td)
            pf.write_feather(pa.table({'bodyId':[30,10,20,40], 'superclass':['cb_sensory','cb_intrinsic','cb_motor',None]}),p/'body-annotations-male-cns-v1.0.feather')
            pf.write_feather(pa.table({'body':[10,20,30], 'consensus_nt':['gaba','unclear','acetylcholine']}),p/'body-neurotransmitters-male-cns-v1.0.feather')
            pf.write_feather(pa.table({'body_pre':[10,20,40], 'body_post':[20,30,10], 'weight':[12,6,99]}),p/'connectome-weights-male-cns-v1.0.feather')
            a=load_graph(10,p,progress=lambda _:None);b=load_graph(5,p,progress=lambda _:None)
            np.testing.assert_array_equal(a['body_ids'],[10,20,30])
            np.testing.assert_array_equal(a['body_ids'],b['body_ids'])
            self.assertEqual(len(a['src']),1);self.assertEqual(len(b['src']),2)
            np.testing.assert_array_equal(load_graph(10,p,progress=lambda _:None)['src'],a['src'])


if __name__=='__main__': unittest.main(verbosity=2)
