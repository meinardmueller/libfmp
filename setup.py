from setuptools import setup, find_packages


with open('README.md', 'r') as fdesc:
    long_description = fdesc.read()

setup(
    name='libfmp',
    version='1.2.7',
    description='Python module for fundamentals of music processing',
    author='Meinard Müller and Frank Zalkow',
    author_email='meinard.mueller@audiolabs-erlangen.de',
    url='http://audiolabs-erlangen.de/FMP',
    download_url='https://github.com/meinardmueller/libfmp',
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 3",
    ],
    keywords='audio music sound',
    license='MIT',
    install_requires=['ipython >= 8.10.0, < 9.0.0',
                      'librosa >= 0.10.0, < 1.0.0',
                      'matplotlib >= 3.3.0, < 4.0.0',
                      'music21 >= 9.1.0, < 10.0.0',
                      'numba >= 0.58.1, < 1.0.0',
                      'numpy >= 1.19.0, < 3.0.0',
                      'pandas >= 1.1.0, < 3.0.0',
                      'pretty_midi >= 0.2.0, < 1.0.0',
                      'soundfile >= 0.9.0, < 1.0.0',
                      'scipy >= 1.5.0, < 2.0.0'],
    python_requires='>=3.10',
    extras_require={
        'tests': ['pytest >= 8.0.0, < 9.0.0']
    }
)
