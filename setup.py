from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='diffeqpy',
      version='1.1.0',
      description='Solving Differential Equations in Python',
      long_description=readme(),
      long_description_content_type="text/markdown",
      classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics'
      ],
      url='http://github.com/JuliaDiffEq/diffeqpy',
      keywords='differential equations stochastic ordinary delay differential-algebraic dae ode sde dde',
      author='Chris Rackauckas and Takafumi Arakaki',
      author_email='contact@juliadiffeq.org',
      license='MIT',
      packages=['diffeqpy','diffeqpy.tests'],
      install_requires=['julia>=0.2'],
      include_package_data=True,
      zip_safe=False)
