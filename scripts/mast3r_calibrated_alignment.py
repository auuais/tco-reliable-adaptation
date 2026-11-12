"""Add an explicit known-intrinsics comparator to geometric sparse alignment."""
import torch
import torch.nn.functional as F
from mast3r_joint_prior_loss import JointPriors


class CalibratedPriors(JointPriors):
    def __init__(self,module,arguments,samples,poses,k,pose_weight):
        super().__init__(module,arguments,samples,poses,.01,pose_weight)
        self.k=torch.as_tensor(k,dtype=torch.float32,device=arguments['device'])

    def __call__(self,k,cameras,depthmaps,phase,iteration):
        base=super().__call__(k,cameras,depthmaps,phase,iteration)
        if phase!='loss_2d': return base
        predicted=torch.stack([k[:,0,0],k[:,1,1],k[:,0,2],k[:,1,2]],dim=-1)
        target=torch.stack([self.k[0,0],self.k[1,1],self.k[0,2],self.k[1,2]])
        normalizer=.02*torch.stack([self.k[0,0],self.k[1,1],self.k.new_tensor(512.),self.k.new_tensor(384.)])
        residual=(predicted-target)/normalizer
        intrinsic=F.huber_loss(residual,torch.zeros_like(residual),delta=1.)
        total=base+.01*intrinsic
        if not torch.isfinite(total):raise FloatingPointError('Nonfinite calibrated alignment objective')
        self.trace[-1].update(intrinsic=float(intrinsic.detach()),total_weighted=float(total.detach()))
        return total
