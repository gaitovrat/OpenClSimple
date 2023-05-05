#include <CL/opencl.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <vector>

int main() {
  cl::Platform platform;
  cl::Device device;

  // 1. Query for available OpenCL platforms and devices
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  if (platforms.size() == 0) {
    std::cerr << "No platforms found.\n";
    return EXIT_FAILURE;
  }

  platform = platforms[0];
  std::cout << "Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << '\n';

  std::vector<cl::Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
  if (devices.size() == 0) {
    std::cerr << "No devices found\n";
    return EXIT_FAILURE;
  }

  device = devices[0];
  std::cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << '\n';

  // 2. Create a context for one or more OpenCL devices in a platform
  cl::Context context(device);

  // 3. Create and build programs for OpenCL devices in the context
  std::ifstream fin("add.cl");
  if (!fin.is_open()) {
    std::cerr << "add.cl not found.\n";
    return EXIT_FAILURE;
  }
  std::stringstream buffer;
  buffer << fin.rdbuf();

  cl::Program program(context, buffer.str());
  if (program.build() != CL_SUCCESS) {
    std::cerr << "Error building: "
              << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << '\n';
  }

  // 4. Create memory objects for kernels to operate on
  constexpr int n = 100;
  int a[n];
  cl::Buffer A(context, CL_MEM_READ_WRITE, sizeof(int) * n);
  int b[n];
  cl::Buffer B(context, CL_MEM_READ_WRITE, sizeof(int) * n);
  int c[n];
  cl::Buffer C(context, CL_MEM_READ_WRITE, sizeof(int) * n);
  cl::Buffer N(context, CL_MEM_READ_ONLY, sizeof(n));

  // Fill arrays
  for (int i = 0; i < n; i++) {
    a[i] = i;
    b[i] = n - i - 1;
    c[i] = 0;
  }

  // 5. Create command queues to execute commands on an OpenCL device
  cl::CommandQueue queue(context, device);
  queue.enqueueWriteBuffer(A, CL_TRUE, 0, sizeof(int) * n, a);
  queue.enqueueWriteBuffer(B, CL_TRUE, 0, sizeof(int) * n, b);
  queue.enqueueWriteBuffer(N, CL_TRUE, 0, sizeof(n), &n);

  // 6. Create kernels
  cl::Kernel add(program, "simple_add");
  add.setArg(0, A);
  add.setArg(1, B);
  add.setArg(2, C);
  add.setArg(3, N);

  // Execute and read data back
  queue.enqueueNDRangeKernel(add, cl::NullRange, cl::NDRange(10));
  queue.enqueueReadBuffer(C, CL_TRUE, 0, sizeof(int) * n, c);

  std::cout << "A: ";
  for (int i = 0; i < 30; ++i) {
    std::cout << a[i] << ' ';
  }
  std::cout << "\nB: ";
  for (int i = 0; i < 30; ++i) {
    std::cout << b[i] << ' ';
  }
  std::cout << "\nC: ";
  for (int i = 0; i < 30; ++i) {
    std::cout << c[i] << ' ';
  }

  return EXIT_SUCCESS;
}
