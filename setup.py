from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='diffeqpy',
      version='2.5.2',
      description='Solving Differential Equations in Python',
      long_description=readme(),
      long_description_content_type="text/markdown",
      classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics'
      ],
      url='http://github.com/SciML/diffeqpy',
      keywords='differential equations stochastic ordinary delay differential-algebraic dae ode sde dde oneapi AMD ROCm CUDA Metal GPU',
      author='Chris Rackauckas and Takafumi Arakaki',
      author_email='contact@sciml.ai',
      license='MIT',
      packages=['diffeqpy','diffeqpy.tests'],
      install_requires=['juliacall>=0.9.14', 'jill'],
      include_package_data=True,
      zip_safe=False)
