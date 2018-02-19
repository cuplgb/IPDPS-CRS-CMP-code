# GPU Seismic Processing
In this repo you can find the GPGPU versions of the CMP, CRS and CRS-DE methods.

The first directory level is divided in the three different methods. Inside each method, i.e. in the second directory level, you can find the folders containing the implementation in OpenMP, CUDA, OpenCL and OpenACC. Each implementation is self contained and should be compiled separatly.

# Building
Building each code is highly framework dependent, so we created a list bellow showing how to compile each code for each framework. Note that the instructions can be similiar among some frameworks, like OpenCL and OpenMP.

The minimal requirements for building the code are listed bellow. Other versions may work too, but try them at your own risk.

After building and generating the binaries, you can go to the testing section bellow

## OpenCL
For building the code under the OpenCL folders you will need
- OpenCL 1.2 support
- GCC (7.2.0)
- CMake (3.9.6)

In this example we will build the code under *CRS-DE/OpenCL*. By default, cmake will use a folder called *bin* to generate its temporaries and binary files. Inside the *bin* folder we can call the build tool and CMake should take care of the rest. To build the code in release mode, just type.
```sh
cd CRS-DE/OpenCL
mkdir bin
cd bin
cmake -DCMAKE_BUILD_TYPE="Release" ../ && make -j 4
```
If everything went well, the binary file called *crs-ocl-de* was created and you can move forward to the testing section

## OpenMP
For building the code under the OpenMP folders you will need
- GCC (7.2.0)
- CMake (3.9.6)

In this example we will build the code under *CMP/OpenMP*. By default, cmake will use a folder called *bin* to generate its temporaries and binary files. Inside the *bin* folder we can call the build tool and CMake should take care of the rest. To build the code in release mode, just type.
```sh
cd CMP/OpenMP
mkdir bin
cd bin
cmake -DCMAKE_BUILD_TYPE="Release" ../ && make -j 4
```
If everything went well, the binary file called *cmp-omp2* was created and you can move forward to the testing section

## OpenACC
For building the code under the OpenACC folders you will need
- PGI Community Edition (17.04)
- CMake (3.9.6)
- GCC (5.4.0)

You also need the *pgcc* and *pgc++* under your *PATH*. Maybe it will also be necessary placing your libc libraries in your C path.

In this example we will build the code under *CMP/OpenACC*. By default, cmake will use a folder called *bin* to generate its temporaries and binary files. Inside the *bin* folder we can call the build tool and CMake should take care of the rest. To build the code in release mode, just type.
```sh
cd CMP/OpenACC
mkdir bin
cd bin
cmake -DCMAKE_BUILD_TYPE="Release" ../ && make -j 4
```
If everything went well, the binary file called *cmp-acc* was created and you can move forward to the testing section

## CUDA
For building the code under CUDA folders you will need
- CUDA 8
- Make (4.2.1)
- GCC (7.2.0)

You also need the *nvcc* compiler under your *PATH*. Maybe it will also be necessary placing your libc libraries in your C path.

In this example we will build the code under *CMP/CUDA*. By default, the Makefile will use a folder called *bin* to generate its temporaries and binary files. To build the code in release mode, just type.
```sh
cd CMP/CUDA
mkdir bin
make
```
If everything went well, the binary file called *cmp-cuda* was created in the *bin* foler and you can move forward to the testing section

# Testing
We will show how to use the binaries generated above through an example, but extending it for other binaries and datasets should be simple. In this example, we'll run the CRS-DE method in OpenCL on the dataset *simple-synthetic.su*.

Each method(CMP, CRS, CRS-DE) and framework (OpenCL, CUDA, OpenACC) accepts a wide range of parameters. To list the accepted arguments you can call the binary without any parameters or with the `--help` argument.

Let's start by entering the folder where the binary is located and list the available arguments.
```sh
cd CRS-DE/OpenCL/bin
./crs-ocl-de --help
```

You should get the following output
```
Usage: ./crs-ocl-de [options]
Options:
  -a0: A0 constant
  -a1: A1 constant
  -aph: APH constant
  -apm: APM constant
  -azimuth: Azimuth angle
  -b0: B0 constant
  -b1: B1 constant
  -c0: C0 constant
  -c1: C1 constant
  -d: OpenCL device
  -i: Data path
  -na: NA constant
  -nb: NB constant
  -nc: NC constant
  -ngen: Tau constant
  -tau: Tau constant
  -v: Verbosity Level 0-3
```

The parameter `-i` specifies the dataset path, i.e. the path to the *.su* file. The parameter `-d` specifies the OpenCL device to run the kernels on. It should be a number between one and the number of devices available on the system. To make the program prompt asking for a device during run time, just set it to 0. The `-v` parameter controls the verbosity level of the application. All the other parameters are dataset dependent. Bellow, there is a table with these specific values for some common datasets

| Dataset | simple-synthetic.su | fold1000.su |
|---------|:-------------------:|:-----------:|
| aph     | 600                 | 2600        |
| apm     | 50                  | 50          |
| azimuth | 0                   | 90          |
| a0      | -0.7e-3             | -0.7e-3     |
| a1      | 0.7e-3              | 0.7e-3      |
| b0      | -1e-7               | -1e-7       |
| b1      | 1e-7                | 1e-7        |
| c0      | 1.98e-7             | 1.975e-7    |
| c1      | 1.77e-6             | 1.384e-6    |
| na      | 5                   | 5           |
| nb      | 5                   | 5           |
| nc      | 5                   | 5           |
| ngen    | 30                  | 30          |
| tau     | 0.002               | 0.004       |

An example for running `crs-ocl-de` on the *simple-synthetic* dataset on the first OpenCL device present on the system, with the highest verbosity level would be:

```sh
./crs-ocl-de -ngen 30 -azimuth 0 -a0 -0.7e-3 -a1 0.7e-3 -na 5 -c0 1.98e-7 -c1 1.77e-6 -nc 5 -b0 -1e-7 -b1 1e-7 -nb 5 -aph 600 -apm 50 -tau 0.002 -d 1 -v 4 -i ../../../datasets/simple-synthetic.su
```

If everything succeds, the command above will generate five *.su* files as output that can be viewed using the seismic unix suite.

**Note**: For targeting a specific device while running OpenACC based code, it's necessary to export the `ACC_DEVICE_TYPE` variable with one of the following values: `nvidia,host`

# Bugs
If you've encountered any errors/doubts and would like helping to improve this repo, create an issue or make a pull request.
