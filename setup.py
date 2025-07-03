from setuptools import setup, find_packages
import os


def get_requirements(filename, exclude=None):
    exclude = exclude or []
    with open(filename) as f:
        return [
            line.strip()
            for line in f
            if line.strip()
            and not line.startswith("#")
            and not any(line.strip().startswith(pkg) for pkg in exclude)
        ]


setup(
    name="adversarial-lab",
    version="0.0.3",
    description="A unified library for performing adversarial attacks on ML models",
    long_description=open("README.md").read(
    ) if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="Pavan Reddy",
    author_email="preddy.osdev@gmail.com",
    url="https://github.com/pavanreddyml/adversarial-lab",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=get_requirements(
        "requirements.txt", exclude=["tensorflow", "torch"]),
    extras_require={
        "tensorflow": ["tensorflow==2.18.0"],
        "torch": ["torch==2.4.1"],
        "full": [
            "tensorflow==2.18.0",
            "torch==2.4.1",
        ],
        "testing": [
            "pytest==8.3.3",
            "pytest-cov==5.0.0",
            "tensorflow==2.18.0",
            "torch==2.4.1",
        ],
        "build": [
            "pytest==8.3.3",
            "pytest-cov==5.0.0",
            "flake8==7.1.1",
            "mypy==1.12.0",
            "Sphinx==8.2.3",
            "sphinx-autodoc-typehints==3.1.0",
            "sphinx-rtd-theme==3.0.2",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_data={"": ["*.md", "*.txt"]},
)
