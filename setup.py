import setuptools

setuptools.setup(
    name="vqvae-maxjcohen",
    version="0.0.1",
    author="Max Cohen",
    author_email="lol44zla5@relay.firefox.com",
    description="Minimal VQ-VAE implementation.",
    url="https://github.com/maxjcohen/vqvae",
    packages=["vqvae"],
    python_requires=">=3.6",
    install_requires=[
        "torch",
    ],
)
