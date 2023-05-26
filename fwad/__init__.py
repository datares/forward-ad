import torch
import torch.autograd.forward_ad as fwAD

class ForwardModule():
    def __init__(self, module):
        self.module = module
    
    def __call__(self, *args):
        return self.module(*args)
    
    def _set_parameters(self, params):
        tangents = {name: torch.rand_like(p) for name, p in params.items()}
        for name, p in params.items():
            namelist = name.split('.')
            parent = self.module
            while len(namelist) > 1:
                parent = getattr(parent, namelist.pop(0))
            delattr(parent, namelist[0])
            setattr(parent, namelist[0], fwAD.make_dual(p, tangents[name]))
    
    def jvp(self, *args):
        with fwAD.dual_level(), torch.no_grad():
            self._set_parameters({name: p for name, p in self.module.named_parameters()})
            out = self.module(*args)
            return fwAD.unpack_dual(out)
