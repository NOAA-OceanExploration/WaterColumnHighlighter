from setuptools import setup, find_packages

setup(
    name="owl_highlighter",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.36.2",
        "Pillow>=9.0.0",
        "opencv-python>=4.9.0.80",
        "colorama>=0.4.6",
        "toml",
        "safetensors",
        "scipy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "tqdm"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ]
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for detecting and highlighting objects in videos using OWL-ViT",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/owl_highlighter",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    package_data={
        "owl_highlighter": ["*.ttf"],
    },
    entry_points={
        "console_scripts": [
            "owl-evaluate=owl_highlighter.evaluate_detections:main",
        ],
    },
)