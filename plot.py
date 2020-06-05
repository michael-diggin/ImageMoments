import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':

    drt_times = [7.43848e-05, 0.000206031, 0.000401039, 0.000699696, 0.00119656, 0.00205434, 0.00454322, 0.00633684]
    ocv_times = [0.000171485, 0.000392842, 0.00130209, 0.00269601, 0.00434289, 0.00808874, 0.0164434, 0.022194]

    num_pixels = ['200x200', '400x400', '750x750', '1000x1000', '15000x15000', '2000x2000', '3000x3000', '4032x3024']
    num_pixels = np.sqrt([40000, 160000, 562500, 1000000, 2250000, 4000000, 9000000, 12192768])

    plt.plot(num_pixels, drt_times, '-', label='DRT')
    plt.plot(num_pixels, ocv_times, '--', label='OPEN CV')

    plt.ylabel('Time in ms')
    plt.xlabel('Num. of Pixels (sqrt)')
    plt.title('Comparison of Computation Time')
    plt.legend()
    plt.show() 