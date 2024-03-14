from setuptools import setup, find_packages

setup(
    name="chaski",
    version="0.1.0",
    description="An LLM python server leveraging llama-cpp-python",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="cck",
    author_email="christopher.kroenke@gmail.com",
    url="https://github.com/yourusername/llama-app",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "flask",
        "llama-cpp-python",
        "fire",
        "fastcore",
        "nbdev",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
            "sphinx",
            "sphinx-rtd-theme",
        ],
    },
    entry_points={
        "console_scripts": [
            "chaski-llm=chaski.app:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.9",
)