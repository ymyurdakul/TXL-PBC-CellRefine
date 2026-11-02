# Analytic parameter and FLOP count of the CellRefine network for a 128 x 128 crop (FLOPs = 2 x multiply-accumulates).
def conv(cin, cout, k, h, w): return cout * h * w * cin * k * k, cout * cin * k * k
def resnet18_body(size=128, upto=4):
    macs = params = 0
    s = size // 2; m, p = conv(3, 64, 7, s, s); macs += m; params += p + 128  # conv1 + bn
    s //= 2  # maxpool
    cin = 64
    for li, cout in enumerate([64, 128, 256, 512][:upto]):
        stride = 1 if li == 0 else 2
        so = s // stride
        for b in range(2):
            ci = cin if b == 0 else cout
            m1, p1 = conv(ci, cout, 3, so, so); m2, p2 = conv(cout, cout, 3, so, so)
            macs += m1 + m2; params += p1 + p2 + 4 * cout
            if b == 0 and (stride != 1 or ci != cout):
                m3, p3 = conv(ci, cout, 1, so, so); macs += m3; params += p3 + 2 * cout
        s = so; cin = cout
    return macs, params, cin, s
for name, upto in (('ResNet-18', 4), ('ResNet-18 without layer4', 3)):
    m, p, ch, s = resnet18_body(128, upto)
    mr, pr = conv(ch, 128, 1, s, s); pr += 256
    d = 128 * 4 * 4 + 3 + 2
    mlp_m = d * 512 + 512 * 512 + 512 * 68 + 512; mlp_p = (d + 1) * 512 + 513 * 512 + 513 * 68 + 513
    M = m + mr + mlp_m; Pp = p + pr + mlp_p
    print(f'{name}: params {Pp/1e6:.3f} M, MACs {M/1e6:.1f} M, GFLOPs per box {2*M/1e9:.3f}')
