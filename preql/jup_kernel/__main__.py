from ipykernel.kernelapp import IPKernelApp
from . import PreqlKernel

IPKernelApp.launch_instance(kernel_class=PreqlKernel)
