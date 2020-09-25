import pathlib
import setuptools
 
long_description = "Planner"

# This call to setup() does all the work
setuptools.setup(
    name="ltl",
    version="0.0.1",
    description="ltl and env",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    author="Yen-Ling Kuo",
    author_email="author@email.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python"
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.7"
)
