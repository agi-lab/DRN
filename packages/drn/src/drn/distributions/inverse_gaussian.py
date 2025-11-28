import torch


class InverseGaussian(torch.distributions.Distribution):
    def __init__(self, mean: torch.Tensor, dispersion: torch.Tensor):
        self.mu = mean
        self.dispersion = dispersion
        # Properly initialize the parent class
        batch_shape = mean.shape
        super(InverseGaussian, self).__init__(batch_shape)

    def log_prob(self, y: torch.Tensor) -> torch.Tensor:
        term1 = -0.5 * torch.log(2 * torch.pi * self.dispersion * y**3)
        term2 = -((y - self.mu) ** 2) / (2 * self.dispersion * y * self.mu**2)
        return term1 + term2

    @property
    def mean(self) -> torch.Tensor:
        return self.mu

    def cdf(self, y: torch.Tensor) -> torch.Tensor:
        lambda_ = 1.0 / self.dispersion

        sqrt_term = torch.sqrt(lambda_ / y)
        z1 = sqrt_term * (y / self.mu - 1.0)
        z2 = -sqrt_term * (y / self.mu + 1.0)

        standard_normal = torch.distributions.Normal(0.0, 1.0)
        term1 = standard_normal.cdf(z1)
        term2 = torch.exp(2.0 * lambda_ / self.mu) * standard_normal.cdf(z2)

        return term1 + term2
