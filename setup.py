from setuptools import setup, find_packages


with open('README.md', 'r') as fdesc:
    long_description = fdesc.read()

setup(
    name='libfmp',
    version='1.1.2',
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
    install_requires=['ipython == 7.8.*',
                      'librosa == 0.8.*',
                      'matplotlib == 3.1.*',
                      'music21 == 5.7.*',
                      'numba == 0.51.*',
                      'numpy == 1.17.*',
                      'pandas == 1.0.*',
                      'pretty_midi == 0.2.*',
                      'pysoundfile == 0.9.*',
                      'scipy == 1.3.*'],
    python_requires='>=3.6, <=3.8',
    extras_require={
        'tests': ['pytest == 6.2.*']
    }
)
