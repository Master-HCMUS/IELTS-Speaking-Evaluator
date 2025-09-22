"""
Setup configuration for IELTS Speaking Audio Recorder.

This setup.py file enables installation of the audio recorder as a Python package
with proper dependency management and entry points for command-line usage.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8') if (this_directory / "README.md").exists() else ""

# Read requirements from requirements.txt
requirements = []
requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith('#')
        ]

setup(
    name="ielts-speaking-recorder",
    version="1.0.0",
    author="IELTS Speaking Evaluator Team",
    author_email="your-email@example.com",
    description="Cross-platform audio recorder for IELTS speaking evaluation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ielts-speaking-recorder",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Topic :: Education :: Testing",
        "Topic :: Multimedia :: Sound/Audio :: Capture/Recording",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Environment :: Console",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ],
        "audio-analysis": [
            "librosa>=0.10.0",
            "pydub>=0.25.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "ielts-recorder=src.cli:main",
            "audio-recorder=src.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src": ["config/*.json"],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/ielts-speaking-recorder/issues",
        "Source": "https://github.com/yourusername/ielts-speaking-recorder",
        "Documentation": "https://github.com/yourusername/ielts-speaking-recorder/wiki",
    },
    keywords="audio recording microphone wav ielts speaking evaluation cross-platform",
)