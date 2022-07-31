from setuptools import setup
import re, io

__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',  # It excludes inline comment too
    io.open('spygrt/__init__.py', encoding='utf_8_sig').read()
    ).group(1)
setup(
    name ='SpyGRT',
    python_requires = '>=3.7, <3.10',
    packages = ['spygrt'],
    version = __version__,
    license = 'MIT',
    description = 'Toolkit to build SGRT applications',
    author = 'Youssef Ben Bouchta',
    author_email='youssef.benbouchta@sydney.edu.au',
    url = 'https://github.com/ACRF-Image-X-Institute/SpyGRT',
    download_url = 'https://github.com/ACRF-Image-X-Institute/SpyGRT/archive/refs/tags/v_01.tar.gz',
    keywords = ['radiotherapy', 'realsense', 'SGRT', 'radiation-oncology'],
    install_requires=['opencv-python', 
                      'pyrealsense2', 
                      'open3d', 
                      'numpy'],
    long_description_content_type = 'text/markdown',
    long_description=open('README.md').read(),
)