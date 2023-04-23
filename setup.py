from setuptools import setup

setup(name="flops_profile",
      version="0.1.0",
      description="a flops profile w. pytorch",
      author="burnessduan",
      author_email="burness1990@gmail.com",
      url="",
      license="apache 2.0",
      keywords="flops profile",
      project_urls={
        "Documentation": "",
        "source": "",
        "Tracker": ""
      },
      packages=["flops_profiler"],
      install_requires=['torch>=1.10.0'],
      python_requires=">=3"
      )