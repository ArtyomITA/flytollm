import tempfile
import unittest
from pathlib import Path
import torch
from fly_core import Core
from fly_interfaces import InterfaceConfig,TextInterfaces,SparseInjector,PopulationReadout,anatomical_ports


def fixture():
    c=InterfaceConfig(vocab=32,dim=16,fan_in=4,pool_size=2)
    m=TextInterfaces(12,torch.arange(4),torch.arange(4,12),torch.arange(8)//2,c)
    src=torch.arange(12); dst=(src+4)%12
    core=Core(src,dst,torch.ones(12)*.4,torch.ones(12),12,8)
    return m.cuda(),core.cuda()


class InterfaceTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)

    def test_sparse_reference_and_gradients(self):
        p=SparseInjector(12,torch.arange(4),16,4,17).cuda().double()
        x=torch.randn(2,16,device="cuda",dtype=torch.double,requires_grad=True)
        y=p(x)
        matrix=torch.zeros(16,12,device="cuda",dtype=torch.double).index_put(
            (p.channels.flatten(),p.nodes[:,None].expand_as(p.channels).flatten()),p.weight.flatten(),accumulate=True)
        ref=x@matrix
        expected=torch.zeros_like(y).index_copy(1,p.nodes,p.bias+p.gain*torch.tanh(ref[:,p.nodes]))
        torch.testing.assert_close(y,expected)
        gs=torch.autograd.grad(y.square().sum(),(x,p.weight,p.bias),retain_graph=True)
        gr=torch.autograd.grad(expected.square().sum(),(x,p.weight,p.bias))
        for a,b in zip(gs,gr): torch.testing.assert_close(a,b)
        self.assertEqual(y[:,4:].abs().sum().item(),0)
        self.assertTrue(all(len(r.unique())==4 for r in p.channels))

    def test_pooling_reference(self):
        p=PopulationReadout(torch.tensor([1,3,4]),torch.tensor([0,0,1]),4).cuda().double()
        v=torch.randn(2,6,device="cuda",dtype=torch.double,requires_grad=True)
        r=torch.rand_like(v)
        expected=torch.stack(((v[:,1]+v[:,3])/2,v[:,4],(r[:,1]+r[:,3])/2,r[:,4]),dim=1)
        torch.testing.assert_close(p.features((v,torch.zeros_like(v)),r),expected)

    def test_core_path_tying_gradients_and_roundtrip(self):
        m,core=fixture(); ids=torch.tensor([4,5],device="cuda"); target=torch.tensor([6,7],device="cuda")
        recalled=torch.randn(2,16,device="cuda",requires_grad=True)
        state,rate=core.advance(m.input_current(ids)+m.feedback_current(recalled),8)
        logits=m.logits(state,rate)
        torch.nn.functional.cross_entropy(logits,target).backward()
        for name,p in list(m.named_parameters())+list(core.named_parameters()):
            self.assertIsNotNone(p.grad,name)
            self.assertTrue(torch.isfinite(p.grad).all(),name)
            self.assertGreater(p.grad.abs().max().item(),0,name)
        self.assertGreater(recalled.grad.abs().max().item(),0)
        self.assertEqual(sum(p is m.embedding.weight for p in m.parameters()),1)
        torch.testing.assert_close(logits,torch.nn.functional.linear(m.representation(state,rate),m.embedding.weight))
        replica,_=fixture(); replica.load_state_dict(m.state_dict())
        torch.testing.assert_close(replica.input_current(ids),m.input_current(ids))

    def test_no_edges_no_input_output_shortcut(self):
        m,core=fixture()
        ids=torch.tensor([4,7],device="cuda")
        state,rate=core.advance(m.input_current(ids),8,weights=torch.zeros_like(core.raw))
        torch.testing.assert_close(m.logits(state,rate)[0],m.logits(state,rate)[1])
        self.assertEqual(m.representation(state,rate).abs().sum().item(),0)

    def test_anatomy_mapping_and_rng(self):
        import pyarrow as pa
        import pyarrow.feather as pf
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'a.feather'
            pf.write_feather(pa.table({'bodyId':[40,10,20,30], 'superclass':['cb_motor','cb_sensory','cb_motor','cb_intrinsic'],
                                      'rootSide':['R','L','R',None]}),p)
            sensory,read,groups,labels=anatomical_ports([10,20,30,40],p,pool_size=1)
            self.assertEqual(sensory.tolist(),[0]); self.assertEqual(read.tolist(),[1,3])
            self.assertEqual(groups.tolist(),[0,1]); self.assertEqual(len(labels),2)
        rng=torch.random.get_rng_state().clone()
        a,_=fixture(); torch.random.set_rng_state(rng); b,_=fixture()
        for x,y in zip(a.parameters(),b.parameters()): torch.testing.assert_close(x,y)


if __name__=='__main__': unittest.main(verbosity=2)
