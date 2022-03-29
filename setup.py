from distutils.core import setup

setup(
    name='SpyGRT',
    version='0.1',
    author='Youssef Ben Bouchta',
    author_email='youssef.benbouchta@sydney.edu.au',
    description='Surface guidance in RT code',
    packages=['spygrt'],
    license='MIT',
    install_requires=['opencv-python', 'pyrealsense2', 'open3d', 'numpy'],
    long_description=open('README.md').read(),
)