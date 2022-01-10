from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()


version = {}
with open("./diffeqpy/_version.py") as fp:
    exec(fp.read(), version)


setup(name='diffeqpy',
      version=version['__version__'],
      description='Solving Differential Equations in Python',
      long_description=readme(),
      long_description_content_type="text/markdown",
      classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics'
      ],
      url='http://github.com/SciML/diffeqpy',
      keywords='differential equations stochastic ordinary delay differential-algebraic dae ode sde dde',
      author='Chris Rackauckas and Takafumi Arakaki',
      author_email='contact@juliadiffeq.org',
      license='MIT',
      packages=['diffeqpy','diffeqpy.tests'],
      install_requires=['julia>=0.2',
                        'julia_project>=0.0.21'],
      include_package_data=True,
      zip_safe=False)
