import setuptools

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="lungseg",
    version="0.1.0",
    author="Botkin.AI",
    author_email="roman.matantsev@botkin.ai",
    description="Package for automated body orientation agnostic lung segmentation in CT (using neural networks)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    entry_points={},
    install_requires=[
        'numpy',
        'torch',
        'scipy',
        'scikit-image',
        'pydicom',
        'torchvision'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPLv3",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.6',
)