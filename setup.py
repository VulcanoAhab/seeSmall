from distutils.core import setup

setup(
    name="seeSmall",
    version='0.1.0',
    author="VulcanoAhab",
    packages=["seeSmall",],
    url="https://github.com/VulcanoAhab/seeSmall.git",
    description="OCR for screenshots",
    install_requires=[
        "tensorflow==1.4.1",
        "scikit-image==0.13.1",
        ]
)
