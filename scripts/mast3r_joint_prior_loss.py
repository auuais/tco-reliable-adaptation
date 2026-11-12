"""Gauge-consistent sparse range and relative-camera terms for MASt3R fine alignment."""
import numpy as np
import torch
import torch.nn.functional as functional


def terms(predicted_depth,cameras,measured_depth,reference_cameras,scale,unit):
    depth=torch.cat(predicted_depth)
    measured=torch.cat(measured_depth)
    depth_residual=(scale*depth-measured)/(.02*unit)
    loss_depth=functional.huber_loss(depth_residual,torch.zeros_like(depth_residual),delta=1.)
    rotation=cameras[0,:3,:3].T@cameras[1:,:3,:3]
    target_rotation=reference_cameras[0,:3,:3].T@reference_cameras[1:,:3,:3]
    translation=(cameras[0,:3,:3].T@(cameras[1:,:3,3]-cameras[0,:3,3]).T).T
    target_translation=(reference_cameras[0,:3,:3].T@(reference_cameras[1:,:3,3]-reference_cameras[0,:3,3]).T).T
    rotation_residual=(rotation-target_rotation)/.02
    translation_residual=(scale*translation-target_translation)/(.02*unit)
    loss_rotation=functional.huber_loss(rotation_residual,torch.zeros_like(rotation_residual),delta=1.)
    loss_translation=functional.huber_loss(translation_residual,torch.zeros_like(translation_residual),delta=1.)
    return dict(depth=loss_depth,rotation=loss_rotation,translation=loss_translation)


class JointPriors:
    def __init__(self,module,arguments,samples,supplied_world_to_camera,depth_weight=.01,pose_weight=.01):
        self.module=module
        self.anchors={}
        self.base_focals=[]
        self.measurements=[]
        self.counts=[]
        self.trace=[]
        self.scale=None
        self.depth_weight=depth_weight
        self.pose_weight=pose_weight
        device=arguments['device']
        for i,(path,values) in enumerate(zip(arguments['canonical_paths'],samples)):
            (_,canonical,_),focal=torch.load(path,map_location=device,weights_only=True)
            height,width=canonical.shape
            pixels=np.rint((values[:,:2]+.5)*[width/518,height/392]-.5).astype(int)
            inside=((pixels>=0)&(pixels<[width,height])).all(axis=1)
            nearest={}
            for j in np.argsort(values[:,2]):
                if inside[j]:
                    nearest.setdefault(tuple(pixels[j]),j)
            selected=np.array(list(nearest.values()))
            if len(selected)<8:
                raise ValueError('Insufficient measurements after native-grid projection')
            xy=torch.as_tensor(pixels[selected],dtype=torch.float32,device=device)
            ids,offset=module.anchor_depth_offsets(canonical,{i:(xy,None)},subsample=arguments['subsample'])
            self.anchors[i]=(xy,ids[i],offset[i])
            self.base_focals.append(focal)
            self.measurements.append(torch.as_tensor(values[selected,2],dtype=torch.float32,device=device))
            self.counts.append(len(selected))
        if len(self.anchors)!=len(samples) or len(samples)<2:
            raise ValueError('Missing canonical camera')
        self.reference=torch.linalg.inv(torch.as_tensor(supplied_world_to_camera,dtype=torch.float32,device=device))
        self.unit=torch.cat(self.measurements).median().detach()

    def __call__(self,k,cameras,depthmaps,phase,iteration):
        if phase!='loss_2d':
            return k.sum()*0
        _,predicted=self.module.make_pts3d(self.anchors,k,cameras,[d.ravel() for d in depthmaps],
                                         base_focals=self.base_focals,ret_depth=True)
        if self.scale is None:
            ratios=[]
            for z,measured in zip(predicted,self.measurements):
                valid=torch.isfinite(z)&(z>0)
                if valid.sum()<8:
                    raise ValueError('Insufficient positive initial depths to anchor metric scale')
                ratios.append(measured[valid]/z[valid].detach())
            self.scale=torch.cat(ratios).median().detach()
        components=terms(predicted,cameras,self.measurements,self.reference,self.scale,self.unit)
        loss=self.depth_weight*components['depth']+self.pose_weight*(components['rotation']+components['translation'])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite joint-prior objective')
        event=dict(iteration=iteration,metric_scale=float(self.scale),unit_m=float(self.unit),
                   depth=float(components['depth'].detach()),rotation=float(components['rotation'].detach()),
                   translation=float(components['translation'].detach()),weighted=float(loss.detach()))
        def backward(gradient):
            event['received_gradient']=float(gradient.detach())
            event['finite_gradient']=bool(torch.isfinite(gradient))
        loss.register_hook(backward)
        self.trace.append(event)
        return loss
