from distutils.core import setup

setup(
    name ='SpyGRT',
    packages = ['spygrt'],
    version ='0.1',
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