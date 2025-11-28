import pytest
import torch
import numpy as np
import pandas as pd
from scipy import stats

from drn import GLM, CANN, MDN, Constant
from drn.distributions import Histogram, ExtendedHistogram


class TestQuantilesConsistency:
    """Test that quantiles work consistently across all models."""

    def test_quantiles_method_exists_all_models(self):
        """Test that all models have quantiles method."""
        models = [
            GLM("gamma"),
            GLM("gaussian"),
            GLM("inversegaussian"),
            Constant("gamma"),
            Constant("gaussian"),
            Constant("inversegaussian"),
            CANN(GLM("gamma")),
            CANN(Constant("gamma")),
            MDN("gamma"),
            MDN("gaussian"),
        ]

        for model in models:
            assert hasattr(
                model, "quantiles"
            ), f"{type(model).__name__} missing quantiles method"
            assert callable(
                getattr(model, "quantiles")
            ), f"{type(model).__name__} quantiles not callable"
            assert hasattr(model, "icdf"), f"{type(model).__name__} missing icdf method"
            assert callable(
                getattr(model, "icdf")
            ), f"{type(model).__name__} icdf not callable"

    def test_quantiles_shape_consistency(self):
        """Test that quantiles return correct shape across all models."""
        np.random.seed(42)
        n_samples = 10
        n_features = 3
        X_test = np.random.randn(n_samples, n_features)
        percentiles = [10, 25, 50, 75, 90]

        # Test GLM models
        for dist in ["gamma", "gaussian", "inversegaussian"]:
            model = GLM(dist)
            quantiles = model.quantiles(X_test, percentiles)
            assert quantiles.shape == (
                len(percentiles),
                n_samples,
            ), f"GLM {dist}: expected {(len(percentiles), n_samples)}, got {quantiles.shape}"

        # Test Constant models
        for dist in ["gamma", "gaussian", "inversegaussian"]:
            model = Constant(dist)
            quantiles = model.quantiles(X_test, percentiles)
            assert quantiles.shape == (
                len(percentiles),
                n_samples,
            ), f"Constant {dist}: expected {(len(percentiles), n_samples)}, got {quantiles.shape}"

    def test_quantiles_monotonicity(self):
        """Test that quantiles are monotonically increasing."""
        np.random.seed(42)

        # Create training data
        X_train = np.random.randn(50, 2)
        y_train = np.random.gamma(2, 2, 50)  # Positive values for gamma

        # Test different models
        models = [GLM("gamma"), Constant("gamma")]

        X_test = np.random.randn(5, 2)
        percentiles = [5, 25, 50, 75, 95]

        for model in models:
            # Fit model
            if hasattr(model, "fit"):
                model.fit(X_train, y_train)

            quantiles = model.quantiles(X_test, percentiles)

            # Check monotonicity for each sample
            for j in range(quantiles.shape[1]):
                col = quantiles[:, j]
                for i in range(len(col) - 1):
                    assert (
                        col[i] <= col[i + 1]
                    ), f"{type(model).__name__}: quantiles not monotonic at sample {j}"

    def test_quantiles_vs_icdf_consistency(self):
        """Test that quantiles method gives same results as calling icdf multiple times."""
        np.random.seed(42)

        X_train = np.random.randn(30, 2)
        y_train = np.random.gamma(2, 2, 30)
        X_test = np.random.randn(3, 2)
        percentiles = [25, 50, 75]

        model = Constant("gamma")
        model.fit(X_train, y_train)

        # Get quantiles using quantiles method
        quantiles_method = model.quantiles(X_test, percentiles)

        # Get quantiles using icdf method
        quantiles_icdf = torch.stack(
            [
                model.icdf(X_test, p / 100.0).squeeze(0)  # Remove the extra dimension
                for p in percentiles
            ],
            dim=0,
        )

        assert torch.allclose(
            quantiles_method, quantiles_icdf, atol=1e-6
        ), "quantiles() and icdf() methods give different results"


class TestQuantilesBounds:
    """Test the effect of bounds on quantile calculation."""

    def test_bounds_effect_on_gamma(self):
        """Test how different bounds affect gamma distribution quantiles."""
        model = Constant("gamma")
        y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        X_train = np.zeros((5, 1))
        model.fit(X_train, y_train)

        X_test = np.zeros((1, 1))

        # Test with default bounds (should be 0 and 200)
        q_default = model.icdf(X_test, 0.5)

        # Test with tight bounds around expected value
        q_tight = model.icdf(X_test, 0.5, l=2.0, u=4.0)

        # Test with very wide bounds
        q_wide = model.icdf(X_test, 0.5, l=0.001, u=1000.0)

        # All should be close (within tolerance)
        assert (
            torch.abs(q_default - q_tight) < 0.1
        ), "Tight bounds give very different result"
        assert (
            torch.abs(q_default - q_wide) < 0.1
        ), "Wide bounds give very different result"

    def test_bounds_effect_on_gaussian(self):
        """Test how different bounds affect Gaussian distribution quantiles."""
        model = Constant("gaussian")
        y_train = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        X_train = np.zeros((5, 1))
        model.fit(X_train, y_train)

        X_test = np.zeros((1, 1))

        # Test with default bounds
        q_default = model.icdf(X_test, 0.5)

        # Test with symmetric bounds around mean
        q_symmetric = model.icdf(X_test, 0.5, l=-10.0, u=10.0)

        # Should be very close since Gaussian is well-behaved
        assert (
            torch.abs(q_default - q_symmetric) < 0.01
        ), "Different bounds significantly affect Gaussian quantiles"

    def test_pytorch_vs_binary_search_comparison(self):
        """Compare PyTorch's built-in icdf with our binary search for distributions that have it."""
        model = Constant("gaussian")
        y_train = np.array([2.0, 4.0, 6.0, 8.0])
        X_train = np.zeros((4, 1))
        model.fit(X_train, y_train)

        X_test = np.zeros((1, 1))

        # Get distribution and use PyTorch's icdf directly
        dist = model.predict(X_test)
        pytorch_quantile = dist.icdf(torch.tensor(0.5))

        # Get quantile using our method (which should try PyTorch first)
        our_quantile = model.icdf(X_test, 0.5)[0]

        # Should be essentially identical
        assert (
            torch.abs(pytorch_quantile - our_quantile) < 1e-6
        ), "PyTorch icdf and our method give different results"


class TestQuantilesEdgeCases:
    """Test edge cases and potential issues with quantiles."""

    def test_extreme_percentiles(self):
        """Test very low and high percentiles."""
        model = Constant("gamma")
        y_train = np.array([1.0, 2.0, 3.0])
        X_train = np.zeros((3, 1))
        model.fit(X_train, y_train)

        X_test = np.zeros((2, 1))

        # Test extreme percentiles
        extreme_percentiles = [0.1, 1, 99, 99.9]
        quantiles = model.quantiles(X_test, extreme_percentiles)

        # Should still be monotonic
        for j in range(quantiles.shape[1]):
            col = quantiles[:, j]
            for i in range(len(col) - 1):
                assert col[i] <= col[i + 1], "Extreme percentiles not monotonic"

        # Should be positive for gamma
        assert torch.all(quantiles > 0), "Gamma quantiles should be positive"

    def test_single_percentile(self):
        """Test with single percentile."""
        model = Constant("gaussian")
        y_train = np.array([0.0, 1.0, 2.0])
        X_train = np.zeros((3, 1))
        model.fit(X_train, y_train)

        X_test = np.zeros((3, 1))

        # Single percentile should work
        quantiles = model.quantiles(X_test, [50])
        assert quantiles.shape == (1, 3), f"Expected (1, 3), got {quantiles.shape}"

    def test_different_input_types(self):
        """Test quantiles with different input types."""
        model = Constant("gamma")
        y_train = np.array([1.0, 2.0, 3.0])
        X_train = np.zeros((3, 1))
        model.fit(X_train, y_train)

        percentiles = [25, 50, 75]

        # Test with numpy
        X_np = np.zeros((2, 1))
        q_np = model.quantiles(X_np, percentiles)

        # Test with torch
        X_torch = torch.zeros(2, 1)
        q_torch = model.quantiles(X_torch, percentiles)

        # Test with pandas
        X_pd = pd.DataFrame(np.zeros((2, 1)), columns=["feat"])
        q_pd = model.quantiles(X_pd, percentiles)

        # All should give same results
        assert torch.allclose(q_np, q_torch, atol=1e-6)
        assert torch.allclose(q_np, q_pd, atol=1e-6)


class TestHistogramQuantiles:
    """Test quantiles for histogram distributions which should have their own implementation."""

    def test_histogram_uses_own_quantiles(self):
        """Test that Histogram uses its own quantiles method, not the binary search."""
        # Create a simple histogram
        cutpoints = torch.linspace(0, 10, 11)
        frequencies = (
            torch.ones(1, 10) / 10.0
        )  # Shape: (batch_size, num_bins), normalized
        hist = Histogram(cutpoints, frequencies)

        # This should use the histogram's own quantiles method
        percentiles = [25, 50, 75]
        quantiles = hist.quantiles(percentiles)

        # For uniform histogram from 0 to 10, quantiles should be straightforward
        expected_25 = 2.5  # 25% of 10
        expected_50 = 5.0  # 50% of 10
        expected_75 = 7.5  # 75% of 10

        assert (
            abs(quantiles[0] - expected_25) < 0.5
        ), f"25th percentile: expected ~{expected_25}, got {quantiles[0]}"
        assert (
            abs(quantiles[1] - expected_50) < 0.5
        ), f"50th percentile: expected ~{expected_50}, got {quantiles[1]}"
        assert (
            abs(quantiles[2] - expected_75) < 0.5
        ), f"75th percentile: expected ~{expected_75}, got {quantiles[2]}"


class TestBoundsIssue:
    """Test to understand and document the bounds issue you mentioned."""

    def test_default_bounds_investigation(self):
        """Investigate what happens with default bounds for different distributions."""

        # Test gamma distribution
        gamma_model = Constant("gamma")
        y_train = np.array([2.0, 4.0, 6.0])
        X_train = np.zeros((3, 1))
        gamma_model.fit(X_train, y_train)
        X_test = np.zeros((1, 1))

        # Check what the default bounds are in the binary search
        # The utils.py binary_search_icdf uses l=0, u=200 for non-histogram distributions
        print(f"Gamma median with default bounds: {gamma_model.icdf(X_test, 0.5)}")
        print(
            f"Gamma median with tight bounds: {gamma_model.icdf(X_test, 0.5, l=1.0, u=10.0)}"
        )

        # Test gaussian distribution
        gauss_model = Constant("gaussian")
        y_train = np.array([0.0, 2.0, 4.0])
        X_train = np.zeros((3, 1))
        gauss_model.fit(X_train, y_train)

        print(f"Gaussian median with default bounds: {gauss_model.icdf(X_test, 0.5)}")
        print(
            f"Gaussian median with wide bounds: {gauss_model.icdf(X_test, 0.5, l=-100.0, u=100.0)}"
        )

        # For comparison, what does PyTorch give directly?
        gamma_dist = gamma_model.predict(X_test)
        gauss_dist = gauss_model.predict(X_test)

        try:
            print(f"Gamma PyTorch icdf: {gamma_dist.icdf(torch.tensor(0.5))}")
        except NotImplementedError:
            print(
                "Gamma PyTorch icdf: Not implemented (hence the need for binary search)"
            )

        try:
            print(f"Gaussian PyTorch icdf: {gauss_dist.icdf(torch.tensor(0.5))}")
        except NotImplementedError:
            print("Gaussian PyTorch icdf: Not implemented")

    def test_convergence_with_bad_bounds(self):
        """Test what happens when bounds are poorly chosen."""
        model = Constant("gamma")
        y_train = np.array([1.0, 2.0, 3.0])
        X_train = np.zeros((3, 1))
        model.fit(X_train, y_train)
        X_test = np.zeros((1, 1))

        # Get the "true" quantile with default (hopefully good) bounds
        true_quantile = model.icdf(X_test, 0.5)

        # Test with bounds that are too tight (don't contain the true quantile)
        # The improved binary search should adapt these bounds
        q_high = model.icdf(X_test, 0.5, l=10.0, u=20.0)
        q_low = model.icdf(X_test, 0.5, l=0.001, u=0.1)

        print(f"True quantile: {true_quantile}")
        print(f"Quantile with too-high bounds: {q_high}")
        print(f"Quantile with too-low bounds: {q_low}")

        # With adaptive bounds, these should be closer to the true value
        # Allow some tolerance, but they shouldn't be completely wrong
        assert (
            torch.abs(q_high - true_quantile) < 2.0
        ), "High bounds give very wrong result"
        assert (
            torch.abs(q_low - true_quantile) < 2.0
        ), "Low bounds give very wrong result"

    def test_gaussian_negative_values(self):
        """Test that Gaussian quantiles can be negative."""
        model = Constant("gaussian")
        y_train = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])  # Mean = 0
        X_train = np.zeros((5, 1))
        model.fit(X_train, y_train)
        X_test = np.zeros((1, 1))

        # Low percentiles should be negative
        q_low = model.icdf(X_test, 0.1)  # 10th percentile
        q_high = model.icdf(X_test, 0.9)  # 90th percentile

        print(f"Gaussian 10th percentile: {q_low}")
        print(f"Gaussian 90th percentile: {q_high}")

        # For symmetric distribution around 0, low percentile should be negative
        assert q_low < 0, "Low percentile of Gaussian should be negative"
        assert q_high > 0, "High percentile of Gaussian should be positive"
        assert (
            torch.abs(q_low + q_high) < 0.1
        ), "Gaussian should be roughly symmetric around mean"


if __name__ == "__main__":
    # Run the bounds investigation
    test = TestBoundsIssue()
    test.test_default_bounds_investigation()
    test.test_convergence_with_bad_bounds()
