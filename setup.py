from setuptools import setup, find_packages

setup(
       name='x86_64_assembly_bindings',
       version='0.6.2',
       packages=find_packages(where='.', include=['aot', 'aot.*']),
       py_modules=['x86_64_assembly_bindings']
)