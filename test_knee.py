# Smallest checks that fail if orientation or tensor plumbing breaks.
import numpy as np, torch
import knee

def test_orient():
    img = np.zeros((4, 6)); img[0, 0] = 1  # marker at top-left
    # coronal, standard IOP (row +x, col -z): unchanged
    assert knee.orient(img, [1, 0, 0, 0, 0, -1], 'Coronal')[0, 0] == 1
    # row -x: mirrored left-right
    assert knee.orient(img, [-1, 0, 0, 0, 0, -1], 'Coronal')[0, -1] == 1
    # col +z (upside down): flipped vertically
    assert knee.orient(img, [1, 0, 0, 0, 0, 1], 'Coronal')[-1, 0] == 1
    # rows/cols swapped (row along z): transposed back to (6,4)
    assert knee.orient(img, [0, 0, -1, 1, 0, 0], 'Coronal').shape == (6, 4)
    # axial standard (row +x, col +y): unchanged
    assert knee.orient(img, [1, 0, 0, 0, 1, 0], 'Axial')[0, 0] == 1

def test_model_shapes():
    m = torch.tensor([[1, 1, 1, 0, 1], [1, 1, 1, 1, 1]]).float()
    mdls = [knee.make_model(False, head=h).eval() for h in ('slot', 'mil')]
    for n, train, S in ((16, True, 8), (16, False, 8), (31, True, (7, 8)), (31, False, 15)):
        x = torch.randint(0, 255, (2, 5, n, knee.SIZE, knee.SIZE), dtype=torch.uint8)
        xi, pos = knee.to_input(x, train)
        assert xi.shape[2] in np.atleast_1d(S) and xi.shape[:2] + xi.shape[3:] == (2, 5, 3, knee.IMG, knee.IMG)
        assert pos.shape == (xi.shape[2],) and 0 <= pos.min() and pos.max() <= 1
        with torch.no_grad():  # CPU autograd over 150 images segfaults on this laptop
            for mdl in mdls:
                assert mdl(xi, m, pos).shape == (2, 12)

def test_mil_head():  # position-aware, and a missing slot's pixels never matter
    torch.manual_seed(0)
    mdl = knee.make_model(False, head='mil').eval()
    with torch.no_grad():
        mdl.pos_emb.weight.normal_()  # trained embeddings are non-zero; zeros at init would hide order
        x = torch.randint(0, 255, (1, 5, 16, 64, 64), dtype=torch.uint8)
        m = torch.tensor([[1, 1, 1, 0, 1]]).float()
        xi, pos = knee.to_input(x, False)
        out = mdl(xi, m, pos)
        assert not torch.allclose(out, mdl(xi.flip(2), m, pos), atol=1e-5)  # slice order now matters
        xg = xi.clone(); xg[:, 3] = torch.randn_like(xg[:, 3]) * 100
        assert torch.allclose(out, mdl(xg, m, pos), atol=1e-5)  # masked slot ignored
        mdl.train(); knee.SLOT_DROP = 1.0  # dropout of every series must still keep one per study
        one = torch.tensor([[0, 0, 1, 0, 0]]).float()
        assert torch.isfinite(mdl(xi, one, pos)).all() and torch.isfinite(mdl(xi, m, pos)).all()
        knee.SLOT_DROP = 0.15

def test_dense_compat():  # a 16-slice model on the 31-slice test prep sees exactly its training slices
    for n in range(5, 60):
        for side in 'LR':
            for plane in ('Sagittal', 'Coronal'):
                knee.N_SLICE = 31; a = knee.pick_idx(n, side, plane)[::2]
                knee.N_SLICE = 16; b = knee.pick_idx(n, side, plane)
                assert (a == b).all(), (n, side, plane, a, b)
    knee.N_SLICE = 31

def _series(d, row, col, rows, cols, ps, n, ball=None, copies=1):
    """Write a synthetic DICOM series (optionally a 25mm-radius ball at x=100) into directory d."""
    import os, pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    os.makedirs(d, exist_ok=True)
    row, col = np.array(row, float), np.array(col, float)
    nrm = np.cross(row, col)
    for c in range(copies):  # copies > 1 = several echoes at the same positions
        for k in range(n):
            ipp = np.array([100., 0, 0]) + nrm * (k - (n - 1) / 2) * 4 - row * ps * cols / 2 - col * ps * rows / 2
            jj, ii = np.meshgrid(np.arange(cols), np.arange(rows))
            p = ipp + row * ps * jj[..., None] + col * ps * ii[..., None]
            img = ((np.linalg.norm(p - [100, 0, 0], axis=-1) < 25) * 900).astype(np.uint16) if ball else np.zeros((rows, cols), np.uint16)
            ds = Dataset(); ds.file_meta = FileMetaDataset()
            ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.4'
            ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID = generate_uid()
            ds.Rows, ds.Columns, ds.PixelSpacing = rows, cols, [ps, ps]
            ds.ImageOrientationPatient, ds.ImagePositionPatient = list(row) + list(col), list(ipp)
            ds.InstanceNumber, ds.EchoNumbers, ds.Laterality = k + 1, c + 1, 'L'
            ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, 'MONOCHROME2'
            ds.BitsAllocated = ds.BitsStored = 16; ds.HighBit = 15; ds.PixelRepresentation = 0
            ds.PixelData = img.tobytes()
            ds.save_as(os.path.join(d, f'{c}_{k}.dcm'), enforce_file_format=True)

def test_prep_v2():  # true-scale crop, one image per position, same-plane slot fallback
    import os, tempfile, shutil
    root = tempfile.mkdtemp()
    try:
        knee.PREP, knee.N_SLICE = 2, 31
        COR, SAG = ([1, 0, 0], [0, 0, -1]), ([0, 1, 0], [0, 0, -1])
        for rows, cols, ps in ((384, 512, .3), (320, 320, .4)):  # non-square / FOV under 150mm
            d = os.path.join(root, f'c{rows}{cols}'); _series(d, *COR, rows, cols, ps, 12, ball=True)
            h = knee.series_headers(d, 'Coronal')
            ys, xs = np.where(knee.read_series(d, 'Coronal', h, 'L')[15] > 128)
            assert abs(np.ptp(ys) - np.ptp(xs)) <= 2 and abs(np.ptp(ys) + 1 - 50 / 150 * knee.SIZE) <= 3
        d = os.path.join(root, 'echo'); _series(d, *SAG, 32, 32, 4, 10, copies=2)
        assert len(knee.series_headers(d, 'Sagittal')) == 10
        recs = []
        for uid in ('a', 'b'):  # two non-fat-sat sagittal series, no fat-sat one
            _series(os.path.join(root, 'S', uid), *SAG, 32, 32, 4, 10)
            recs.append({'Anatomical_Plane': 'Sagittal', 'Fat_Suppression': 0, 'SeriesInstanceUID': uid})
        assert knee.prep_study(('S', root, recs))[2].tolist() == [1, 0, 0, 1, 0]
    finally:
        shutil.rmtree(root)

def test_pick_idx():  # right-knee sagittal must actually be reversed (regression: it was a no-op)
    for n in (12, 17, 20, 33, 40):
        l, r = knee.pick_idx(n, 'L', 'Sagittal'), knee.pick_idx(n, 'R', 'Sagittal')
        assert (np.diff(l) >= 0).all() and (np.diff(r) <= 0).all() and r[0] > r[-1]
        assert (knee.pick_idx(n, 'R', 'Coronal') == l).all()

def test_aux_loss():  # max-over-tokens aux logits exist in training, ignore missing slots, and stay off by default
    knee.AUX_W = 0.3
    try:
        mdl = knee.make_model(False, 'efficientnet_b0.ra_in1k', 'mil').train()
        x = torch.randint(0, 255, (2, 5, 31, 64, 64), dtype=torch.uint8)
        m = torch.tensor([[1, 1, 0, 1, 0], [1, 0, 0, 0, 0]]).float()
        knee.IMG = 64
        xi, pos = knee.to_input(x, True)
        out = mdl(xi, m, pos)
        assert out.shape == mdl.aux.shape == (2, 12) and torch.isfinite(mdl.aux).all() and (mdl.aux > -1e3).all()
    finally:
        knee.AUX_W, knee.IMG = 0.0, 224

def test_masked_bce():  # NaN targets carry no loss; with no NaN it is the plain BCE mean
    import torch.nn.functional as F
    lg, t = torch.randn(3, 12), torch.rand(3, 12)
    assert torch.allclose(knee.masked_bce(lg, t), F.binary_cross_entropy_with_logits(lg, t))
    tn = t.clone(); tn[0, :5] = float('nan')
    ok = ~torch.isnan(tn)
    assert torch.allclose(knee.masked_bce(lg, tn), F.binary_cross_entropy_with_logits(lg[ok], t[ok]))
    lg2 = lg.clone(); lg2[0, :5] = 50.0
    assert torch.allclose(knee.masked_bce(lg2, tn), knee.masked_bce(lg, tn))  # masked logits don't matter

def test_mirror():  # contralateral knee: sagittal slice order reversed, coronal/axial flipped on width, labels swapped
    x = torch.randint(0, 255, (2, 5, 31, 8, 8), dtype=torch.uint8)
    xm = knee.mirror(x)
    assert torch.equal(knee.mirror(xm), x)  # its own inverse
    for k, (plane, _) in enumerate(knee.SLOTS):
        assert torch.equal(xm[:, k], x[:, k].flip(1) if plane == 'Sagittal' else x[:, k].flip(-1))
    L = knee.LABELS
    assert [L[i] for i in knee.PERM][2:6] == ['Lateral Meniscus', 'Medial Meniscus', 'Lateral OA', 'Medial OA']
    assert [knee.PERM[i] for i in knee.PERM] == list(range(12))
    t = torch.rand(3, 12)
    tm = knee.mirror_targets(t)
    assert torch.isnan(tm[:, knee.MCL]).all() and torch.equal(tm[:, L.index('Medial OA')], t[:, L.index('Lateral OA')])
    assert torch.equal(tm[:, L.index('ACL')], t[:, L.index('ACL')])
    z = torch.randn(3, 12)
    assert torch.allclose(knee.flip_merge(z, z[:, knee.PERM]), z)  # an equivariant model is unchanged by flip TTA
    zm = torch.randn(3, 12)
    assert torch.equal(knee.flip_merge(z, zm)[:, knee.MCL], z[:, knee.MCL])  # MCL from the plain pass only


test_aux_loss(); test_orient(); test_model_shapes(); test_mil_head(); test_prep_v2(); test_pick_idx(); test_dense_compat(); test_masked_bce(); test_mirror(); print("ok")
