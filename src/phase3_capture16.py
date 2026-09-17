"""TBPTT16 capture; derived from phase3_extended.py, only training length changed.
Source SHA256: e7e751e6758b2fa94c67a333673f80e462dbf9b1e0b5fa112e84dbbecd036e6f
"""
from bench_runtime import emit

class Captured16:
    def __init__(self,model,opt):
        import torch
        from fly_lm import copy_state_
        self.model=model;self.opt=opt
        self.ids=torch.ones((16,2),device='cuda',dtype=torch.long)
        self.targets=self.ids.clone();self.state=model.initial_state(2);self.empty=model.initial_state(2)
        self.eval_ids=torch.ones(2,device='cuda',dtype=torch.long);self.eval_targets=self.eval_ids.clone()
        self.graph=torch.cuda.CUDAGraph();self.eval_graph=torch.cuda.CUDAGraph()
        observed={}
        hooks=[model.interfaces.input.register_forward_hook(lambda m,args,out:observed.__setitem__('token',out)),
               model.interfaces.feedback.register_forward_hook(lambda m,args,out:observed.__setitem__('feedback',out)),
               model.interfaces.readout.register_forward_pre_hook(lambda m,args:observed.__setitem__('rate',args[1]))]
        def forward():
            logits,new=model.step(self.eval_ids,self.state)
            loss,count=model.loss(logits[None],self.eval_targets[None],self.eval_ids[None])
            valid=self.eval_ids.ne(0)&self.eval_targets.ne(0)
            correct=((logits.argmax(-1)==self.eval_targets)&valid).sum()
            restricted=((logits[:,400:408].argmax(-1)+400==self.eval_targets)&valid).sum()
            copy_state_(self.state,model.detach(new))
            rate=observed['rate'];nodes=model.interfaces.input.nodes
            stats=torch.stack([new.voltage.square().mean().sqrt(),new.voltage.abs().max(),rate.mean(),
                               (rate==0).float().mean(),(rate==1).float().mean(),model.interfaces.gate.clone(),
                               observed['token'][:,nodes].square().mean().sqrt(),
                               (model.interfaces.gate*observed['feedback'][:,nodes]).square().mean().sqrt()])
            observed.clear()
            return loss,count,correct,restricted,stats
        def train_function():
            opt.zero_grad(set_to_none=False)
            logits,new=model(self.ids,self.state)
            loss,count=model.loss(logits,self.targets,self.ids)
            with torch.no_grad():
                rate=observed['rate'];nodes=model.interfaces.input.nodes
                stats=torch.stack([new.voltage.square().mean().sqrt(),new.voltage.abs().max(),rate.mean(),
                                   (rate==0).float().mean(),(rate==1).float().mean(),model.interfaces.gate.clone(),
                                   observed['token'][:,nodes].square().mean().sqrt(),
                                   (model.interfaces.gate*observed['feedback'][:,nodes]).square().mean().sqrt()])
            observed.clear();loss.backward()
            finite=torch.stack([torch.isfinite(p.grad).all() for p in model.parameters()]).all()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=False)
            opt.step();copy_state_(self.state,model.detach(new))
            return loss,count,stats,norm,finite & torch.isfinite(stats).all() & torch.isfinite(loss)
        initial=[p.detach().clone() for p in model.parameters()]
        def restore():
            with torch.no_grad():
                for p,v in zip(model.parameters(),initial):p.copy_(v)
                for st in opt.state.values():
                    for v in st.values():v.zero_()
            self.reset();opt.zero_grad(set_to_none=False)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        emit('warmup')
        with torch.cuda.stream(stream):
            for _ in range(3):train_function()
            with torch.no_grad():forward()
        torch.cuda.current_stream().wait_stream(stream);restore()
        emit('capture')
        with torch.cuda.graph(self.graph):self.train_out=train_function()
        restore()
        with torch.no_grad(),torch.cuda.graph(self.eval_graph):self.eval_out=forward()
        restore()
        for hook in hooks:hook.remove()
        observed.clear()

    def reset(self):
        from fly_lm import copy_state_
        copy_state_(self.state,self.empty)

    def stage(self,x,y):
        self.ids.copy_(x);self.targets.copy_(y)

    def update(self,x,y):
        import torch
        if not bool((x.ne(0)&y.ne(0)).any()):
            # Unscored prefix still supplies context; Adam is skipped, state is advanced.
            with torch.no_grad():
                for token,target in zip(x,y):
                    self.eval_ids.copy_(token);self.eval_targets.copy_(target);self.eval_graph.replay()
            return None
        self.stage(x,y);self.graph.replay()
        loss,count,stats,norm,finite=self.train_out
        if not bool(finite):raise RuntimeError('Nonfinite train state or gradients')
        return dict(loss=loss.item(),count=count.item(),stats=stats.tolist(),grad_norm=norm.item())

    def evaluate(self,pairs):
        import torch
        total=0.;count=correct=restricted=0;states=[];per_pair=[]
        with torch.no_grad():
            for pair in pairs:
                self.reset()
                pair_total=0.;pair_count=0
                for x,y in pair:
                    for token,target in zip(x,y):
                        self.eval_ids.copy_(token);self.eval_targets.copy_(target);self.eval_graph.replay()
                        loss,n,c,r,stats=self.eval_out
                        if not torch.isfinite(stats).all() or not torch.isfinite(loss):raise RuntimeError('Nonfinite eval')
                        total+=loss.item()*n.item();count+=n.item();correct+=c.item();restricted+=r.item()
                        pair_total+=loss.item()*n.item();pair_count+=n.item()
                    states.append(stats.tolist())
                per_pair.append(dict(ce=pair_total/max(pair_count,1),targets=pair_count))
        return dict(ce=total/max(count,1),accuracy=correct/max(count,1),restricted_accuracy=restricted/max(count,1),
                    targets=count,stats=states,per_pair=per_pair)

    def close(self):
        self.graph.reset();self.eval_graph.reset();self.model.zero_grad(set_to_none=True)

