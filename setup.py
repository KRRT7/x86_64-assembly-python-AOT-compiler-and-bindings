from setuptools import setup, find_packages

setup(
       name='x86_64_assembly_bindings',  # Your package name
       version='0.4.2',  # Your package version
       packages=find_packages(where='.', include=['aot', 'aot.*']),
       py_modules=['x86_64_assembly_bindings']
)