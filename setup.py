import os

from setuptools import find_packages, setup

from kheperax import __version__

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(CURRENT_DIR, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="kheperax",
    version=__version__,
    packages=find_packages(),
    url="https://github.com/adaptive-intelligent-robotics/Kheperax",
    license="MIT",
    author="Luca Grillotti and Paul Templier",
    author_email="luca.grillotti16@imperial.ac.uk",
    description="A-maze-ing environment in jax",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "jax>=0.4.28",
        "jaxlib>=0.4.28",
        "flax>=0.8.5",
        "qdax>=0.4.0",
        "tqdm>=4.66.5",
        "imageio>=2.35.1",
    ],
    extras_require={
        "cuda12": ["jax[cuda12]>=0.4.30"],
    },
    keywords=["Quality-Diversity", "Reinforcement Learning", "JAX"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
