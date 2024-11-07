from setuptools import find_packages, setup

VERSION = '1.0.0' 
DESCRIPTION = 'A LMDB wrapper for Python'
LONG_DESCRIPTION = 'A package that allows you to interact with LMDB in a more pythonic way, based on pytorch and numpy.'

# 配置
setup(
        name="utils_lmdb", 
        version=VERSION,
        author="Xuangeng Chu",
        author_email="xg.chu@outlook.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[],
        keywords=['lmdb', 'pytorch', 'numpy'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)
