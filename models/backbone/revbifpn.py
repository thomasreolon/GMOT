import torch

try:
    from .RevBiFPN.revbifpn import RevBiFPN_S
except:
    from RevBiFPN.revbifpn import RevBiFPN_S


# note: should comment 
#   print(f"Note: {arch} is pretrained using an input image size of {img_size}.")
#   print(f"Note: revbifpn_s1 is pretrained using an input image size of 256.")
# in RevBiFPN repository>>revbifpn.py

class Revbifpn(RevBiFPN_S):
    def __init__(self, n_feat_levels=3, fat=True, **kwargs_overrides):
        arch = 'revbifpn_s3' if fat else 'revbifpn_s1'
        self.n_feat_levels = n_feat_levels
        super().__init__(arch, pretrained=False, progress=False, strict=True, classes=None, **kwargs_overrides)

    def forward(self, x):
        x = super().forward(x)
        return x



if __name__ == "__main__":
    # small test
    dev = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net, input = Revbifpn().to(dev),  torch.rand(1,3,64,128, device=dev)
    if dev=='cuda':
        import torchvision.transforms.functional  as T
        input = T.pad(input, (0,0,0,0))
    print(input.shape)
    print([v.shape for v in net(input)])
