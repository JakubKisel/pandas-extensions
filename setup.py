
import setuptools

with open ('README.md') as f:
    long_description = f.read()

setuptools.setup(
    name='pdext',
    version='0.0.1',
    description='Pdext extends Pandas with handy methods, properties, accessors and data types without breaking any of its functionality.',
    long_description=long_description
    long_description_content_type='text/markdown '
    packages=setuptools.find_packages(),
    url='https://github.com/JakubKisel/pandas-extensions',
    author='Jakub Kisel',
    author_email='jakub.kisel@gmail.com',
    python_requires='>=3.6'
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scikit-learn'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
        'Operating System :: OS Independent',
        'Topic :: Utilities' 
    ],
) 