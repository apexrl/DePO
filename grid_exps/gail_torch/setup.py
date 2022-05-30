from setuptools import setup

setup(
    name="gail_torch",
    version="0.0.1",
    description="Pytorch implementation of GAIfO.",
    packages=[
        "gail_torch",
        "gail_torch.model",
        "gail_torch.policy",
        "gail_torch.utils",
        "gail_torch.sampler",
    ],
)
