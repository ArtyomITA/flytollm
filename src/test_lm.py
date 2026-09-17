"""Integration checks, GPU-only; invoke via verify_phase24.py."""
import unittest
import tempfile
from pathlib import Path
import torch
from fly_lm import FlyLM,state_tensors,select_token,CUDATokenStep
from fly_attention import CausalAttention,AttentionConfig
from test_interfaces import fixture


def toy():
    interfaces,core=fixture()
    return FlyLM(core,interfaces,CausalAttention(AttentionConfig(dim=16,heads=2,window=4)).cuda())


class LMTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(73)

    def test_sequence_token_and_causality(self):
        m=toy();ids=torch.tensor([[1,1],[4,5],[6,7],[8,9]],device='cuda')
        out,state=m(ids);seq=[];s=None
        for token in ids:
            logits,s=m.step(token,s);seq.append(logits)
        torch.testing.assert_close(out,torch.stack(seq))
        for x,y in zip(state_tensors(state),state_tensors(s)):torch.testing.assert_close(x,y)
        changed=ids.clone();changed[2:]+=3
        torch.testing.assert_close(out[:2],m(changed)[0][:2])

    def test_reset_padding_and_masked_loss(self):
        m=toy();ids=torch.tensor([[1,1],[4,5],[0,0]],device='cuda')
        _,previous=m(ids[:2]);out,after=m(ids[2:],previous)
        for x,y in zip(state_tensors(previous),state_tensors(after)):torch.testing.assert_close(x,y)
        self.assertEqual(out.abs().sum().item(),0)
        bos=torch.ones(2,device='cuda',dtype=torch.long)
        fresh=m.step(bos);reset=m.step(bos,previous)
        torch.testing.assert_close(fresh[0],reset[0])
        for x,y in zip(state_tensors(fresh[1]),state_tensors(reset[1])):torch.testing.assert_close(x,y)
        logits,_=m(ids);target=torch.tensor([[4,5],[2,2],[0,0]],device='cuda')
        loss,count=m.loss(logits,target,ids)
        expected=torch.nn.functional.cross_entropy(logits[:2].flatten(0,1),target[:2].flatten())
        torch.testing.assert_close(loss,expected);self.assertEqual(count.item(),4)
        zero,n=m.loss(logits,target,ids,torch.zeros_like(ids,dtype=torch.bool))
        self.assertEqual(zero.item(),0);self.assertEqual(n.item(),0)

    def test_gradients_and_tbptt(self):
        m=toy();ids=torch.tensor([[1,1],[4,5],[6,7],[8,9]],device='cuda')
        logits,state=m(ids);loss,_=m.loss(logits,ids+3,ids);loss.backward()
        for name,p in m.named_parameters():
            self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)
            self.assertGreater(p.grad.abs().max().item(),0,name)
        detached=m.detach(state)
        for x,y in zip(state_tensors(state),state_tensors(detached)):
            torch.testing.assert_close(x,y);self.assertIsNone(y.grad_fn)
        m.zero_grad(set_to_none=True)
        logits,_=m(ids[1:],detached);m.loss(logits,ids[1:]+3,ids[1:])[0].backward()

    def test_checkpoint_and_story_boundaries(self):
        from lm_io import save_model,load_model,story_batch
        import numpy as np
        m=toy();ids=torch.tensor([[1],[4]],device='cuda')
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.pt';save_model(path,m,{'status':'untrained_test'})
            restored,meta=load_model(path)
            torch.testing.assert_close(m(ids)[0],restored(ids)[0]);self.assertEqual(meta['status'],'untrained_test')
        data=[np.array([1,4,2]),np.array([1,5,6,2])]
        x,y=story_batch(data,[0,1],[1,0],4)
        torch.testing.assert_close(x,torch.tensor([[4,1],[0,5],[0,6],[0,0]],device='cuda'))
        torch.testing.assert_close(y,torch.tensor([[2,5],[0,6],[0,2],[0,0]],device='cuda'))

    def test_sampling_and_eos_graph(self):
        m=toy();logits=torch.zeros(2,32,device='cuda');logits[:,0]=100;logits[:,1]=100
        logits[:,2]=10
        self.assertEqual(select_token(logits,m.config).tolist(),[2,2])
        sample=select_token(logits,m.config,1.,torch.zeros(2,device='cuda'))
        self.assertEqual(sample.tolist(),[2,2])
        # Force EOS as argmax using existing parameters, solely in this fixture.
        with torch.no_grad():
            m.interfaces.output_norm.weight.zero_();m.interfaces.output_norm.bias.fill_(1)
            m.interfaces.embedding.weight.zero_();m.interfaces.embedding.weight[2].fill_(1)
        decoder=CUDATokenStep(m,2)
        try:
            generated=decoder.generate(torch.ones(1,2,device='cuda',dtype=torch.long),4)
            torch.testing.assert_close(generated,torch.tensor([[2,2],[0,0],[0,0],[0,0]],device='cuda'))
            self.assertEqual(decoder.state.cache.next_position.tolist(),[1,1])
        finally:decoder.close()


if __name__=='__main__':unittest.main(verbosity=2)
