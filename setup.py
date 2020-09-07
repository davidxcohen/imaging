import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="imaging_simulation-davidxcohen", # Replace with your own username
    version="0.0.1",
    author="David Cohen",
    author_email="davidxcohen@gmail.com",
    description="Physical Simulation of Imnaging System with Active Light",
    long_description='long_description',
    long_description_content_type="text/markdown",
    url="https://github.com/davidxcohen/imaging",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)