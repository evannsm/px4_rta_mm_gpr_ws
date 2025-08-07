from setuptools import find_packages, setup

package_name = 'testtt'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=['testtt', 'testtt.*']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='factslab-egmc',
    maintainer_email='evannsmcuadrado@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'px4_rta_mm_gpr = scripts.px4_rta_mm_gpr:main',
        ],
    },
)
