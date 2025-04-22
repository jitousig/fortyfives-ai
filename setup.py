import setuptools

setuptools.setup(
    name="fortyfives",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Fortyfives card game environment based on RLCard",
    url="https://github.com/yourusername/fortyfives",
    keywords=["Reinforcement Learning", "card games", "AI", "Fortyfives"],
    packages=setuptools.find_packages(exclude=('tests',)),
    install_requires=[
        'rlcard>=1.0.7',
        'numpy>=1.16.3',
        'matplotlib>=3.0.3',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
) 