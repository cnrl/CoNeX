"""End-to-end tests for the conex.helpers utilities.

Local build_network helper: the shared tests/e2e/conftest.py is owned by
another worker and not yet merged, so each file carries its own builder.

GridEraseMask and GridCropMask are NOT covered: their __init__ imports
torchvision.transforms.functional, and no torchvision release imports cleanly
under the pinned torch==1.13.1 (torchvision 0.14.1 fails on
torch.onnx.symbolic_opset9._cast_Long). GridKeepMask uses pure torch slicing
and is covered.
"""

import pytest
import torch

from conex.helpers.data import LocationDataset
from conex.helpers.filters import DoGFilter, GaborFilter
from conex.helpers.transforms.encoders import SimplePoisson, Poisson, Intensity2Latency
from conex.helpers.transforms.masks import GridKeepMask
from conex.helpers.transforms.misc import (
    UnsqueezeTransform,
    SqueezeTransform,
    SwapTransform,
    DeviceTransform,
    Conv2dFilter,
    AbsoluteTransform,
    DivideSignPolarity,
)


class TestDoGFilter:
    def test_shape(self):
        assert DoGFilter(5, 1.0, 2.0).shape == (5, 5)

    def test_center_is_positive(self):
        f = DoGFilter(5, 1.0, 2.0)
        assert f[2, 2] > 0  # sigma_1 < sigma_2 -> positive center

    def test_zero_mean(self):
        f = DoGFilter(5, 1.0, 2.0, zero_mean=True)
        assert f.sum().item() == pytest.approx(0.0, abs=1e-6)

    def test_one_sum(self):
        f = DoGFilter(5, 1.0, 2.0, one_sum=True)
        assert f.abs().sum().item() == pytest.approx(1.0)


class TestGaborFilter:
    def test_shape(self):
        assert GaborFilter(5, 2.0, 0.0, 1.0, 0.5).shape == (5, 5)

    def test_zero_mean(self):
        f = GaborFilter(5, 2.0, 0.0, 1.0, 0.5, zero_mean=True)
        assert f.sum().item() == pytest.approx(0.0, abs=1e-5)

    def test_orientation_rotates(self):
        a = GaborFilter(7, 3.0, 0.0, 1.0, 0.5)
        b = GaborFilter(7, 3.0, 1.5708, 1.0, 0.5)  # 90 degrees
        assert not torch.allclose(a, b)


class TestSimplePoisson:
    def test_shape_and_dtype(self):
        out = SimplePoisson(time_window=5, ratio=1.0)(torch.tensor([0.0, 1.0]))
        assert out.shape == (5, 2)
        assert out.dtype == torch.bool

    def test_zero_intensity_never_spikes(self):
        out = SimplePoisson(time_window=20, ratio=1.0)(torch.tensor([0.0]))
        assert not out.any()

    def test_tuple_input(self):
        out = SimplePoisson(time_window=3, ratio=1.0)((torch.tensor([1.0]), torch.tensor([0.0])))
        assert isinstance(out, tuple)
        assert out[0].shape == (3, 1)


class TestPoisson:
    def test_shape(self):
        out = Poisson(time_window=20, ratio=1.0)(torch.tensor([0.0, 1.0]))
        assert out.shape == (20, 2)
        assert out.dtype == torch.bool

    def test_zero_intensity_no_spikes(self):
        out = Poisson(time_window=20, ratio=1.0)(torch.tensor([0.0]))
        assert not out.any()

    def test_nonzero_spikes(self):
        torch.manual_seed(0)
        out = Poisson(time_window=50, ratio=1.0)(torch.tensor([1.0]))
        assert out.any()


class TestIntensity2Latency:
    def test_shape(self):
        out = Intensity2Latency(time_window=10)(torch.tensor([0.0, 0.5, 1.0]))
        assert out.shape == (10, 3)

    def test_stronger_spikes_earlier(self):
        # row 0 is the earliest time slot (output is flipped)
        out = Intensity2Latency(time_window=10)(torch.tensor([0.0, 0.5, 1.0]))
        first = [out[:, i].nonzero().flatten()[0].item() for i in range(3)]
        # value 1.0 spikes first (row 0), value 0.0 spikes last
        assert first[2] < first[1] < first[0]

    def test_each_element_spikes_once(self):
        out = Intensity2Latency(time_window=10)(torch.tensor([0.2, 0.8]))
        assert out.sum(dim=0).tolist() == [1, 1]


class TestGridKeepMask:
    def test_shape_and_location(self):
        img = torch.rand(1, 8, 8)
        result, location = GridKeepMask(2, 2)(img)
        assert result.shape == (4, 1, 8, 8)
        assert location.shape == (4, 2, 2)
        assert location.sum().item() == 4  # one True per cell

    def test_keeps_only_one_cell(self):
        img = torch.arange(64.0).reshape(1, 8, 8)
        result, _ = GridKeepMask(2, 2)(img)
        # cell (0,0) keeps the top-left 4x4 and zeros the rest
        assert torch.equal(result[0][0, :4, :4], img[0, :4, :4])
        assert (result[0][0, 4:, :4] == 0).all()
        assert (result[0][0, :4, 4:] == 0).all()

    def test_random_permutes(self):
        torch.manual_seed(0)
        img = torch.rand(1, 8, 8)
        result, location = GridKeepMask(2, 2, random=True)(img)
        assert result.shape == (4, 1, 8, 8)
        assert location.sum().item() == 4


class TestMiscTransforms:
    def test_unsqueeze(self):
        assert UnsqueezeTransform(0)(torch.arange(6.).reshape(2, 3)).shape == (1, 2, 3)

    def test_squeeze(self):
        assert SqueezeTransform(0)(torch.rand(1, 4)).shape == (4,)

    def test_swap(self):
        assert SwapTransform(0, 1)(torch.arange(6.).reshape(2, 3)).shape == (3, 2)

    def test_device(self):
        assert DeviceTransform("cpu")(torch.rand(2)).device.type == "cpu"

    def test_conv2d(self):
        out = Conv2dFilter(torch.rand(1, 1, 3, 3))(torch.rand(1, 1, 5, 5))
        assert out.shape == (1, 1, 3, 3)

    def test_absolute(self):
        assert AbsoluteTransform()(torch.tensor([-1.0, 2.0])).tolist() == [1.0, 2.0]

    def test_divide_sign_polarity(self):
        out = DivideSignPolarity()(torch.tensor([-1.0, 2.0, -3.0]))
        # first half: positives kept, negatives zeroed; second half: |negatives|
        assert out.tolist() == [0.0, 2.0, 0.0, 1.0, 0.0, 3.0]


class TestLocationDataset:
    def test_returns_triple(self):
        class Tiny(torch.utils.data.Dataset):
            def __len__(self):
                return 3

            def __getitem__(self, i):
                return torch.rand(4, 4), i

        ds = LocationDataset(Tiny(), pre_transform=lambda img: (img, torch.tensor([0.0, 0.])))
        assert len(ds) == 3
        img, loc, label = ds[0]
        assert img.shape == (4, 4)
        assert loc.shape == (2,)
        assert label == 0

    def test_transforms_applied(self):
        class Tiny(torch.utils.data.Dataset):
            def __len__(self):
                return 1

            def __getitem__(self, i):
                return torch.ones(2, 2), 5

        ds = LocationDataset(
            Tiny(),
            post_transform=lambda img: img * 2,
            target_transform=lambda y: y + 1,
        )
        img, loc, label = ds[0]
        assert (img == 2.0).all()
        assert label == 6
        assert loc is None
