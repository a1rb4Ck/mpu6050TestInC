#!/usr/bin/python3
"""MPU6050 signals FFT using his fifo buffer
   https://github.com/danjperron/mpu6050TestInC/blob/master/GetFFTmpu6050.py"""

from __future__ import print_function

import math
import time

import MPU6050
import numpy as np
from rpi_audio_levels import AudioLevels


class AccelerometerFFT(object):
    """Perform an FFT on a 1s accelerometer read.

       This class can perform an FFT on a MPU6050 accelerometer.
       It use the built-info fifo buffer of the sensor to achieve a
       0.9 ms max sensors read time, thus is able to sample at full 1Khz rate.
       Raspberry Pi i2c could be set at 400kHz for greater good.

       Attributes:
           sample_rate (int): The accelerometer sample rate in Hertz.
           sample_number (int): The number of sample in the fifo buffer.
           audio_levels:  A GPU FFT wrapper
    """

    def __init__(self, sample_rate=1000, sample_number=1000):
        """FFT configuration and mpu6050 sensor loading."""
        # sample_rate = input("Sample Rate(32.25Hz to 2000Hz) ? ")  # max 1kHz
        # sample_number = input("Number of sample to take (Max buffer~37) ? ")
        self.sample_rate = sample_rate
        self.sample_number = sample_number

        # GPU FFT parameters
        log2_N = math.log(self.sample_number, 2)
        # Preliminary GPU malloc for GPU FFT
        self.audio_levels = AudioLevels(log2_N, self.sample_number)

        self.mpu6050 = MPU6050.MPU6050()
        self.mpu6050.setup()
        self.mpu6050.setGResolution(2)
        self.mpu6050.setSampleRate(self.sample_rate)
        self.mpu6050.enableFifo(False)
        time.sleep(0.01)  # must be in or getting MPU6050 overrun error
        print('Configuration:')
        print('  Capture {0} samples in a fifo buffer at {1} samples/s'.format(
            sample_number, self.mpu6050.SampleRate))

    def run(self):
        """mpu6050 sampling and FFT."""
        self.mpu6050.resetFifo()
        self.mpu6050.enableFifo(True)
        time.sleep(0.01)  # must be in or getting MPU6050 overrun error

        values = []
        total = 0

        t1 = time.time()
        while total < self.sample_number:
            if self.mpu6050.fifoCount == 0:
                status = self.mpu6050.readStatus()
                if (status & 0x10) == 0x10:
                    raise ValueError(
                        'MPU6050 buffer overrun')
                if (status & 0x01) == 0x01:
                    values.extend(self.mpu6050.readDataFromFifo())
            else:
                values.extend(self.mpu6050.readDataFromFifo())

            # Read Total number of data taken
            total = len(values) / 14
        print('Time to read 1 sensor measurement: ',
              (time.time() - t1) / self.sample_number, 's')
        print(" ")

        # Now that we have the data let's write the files
        if total > 0:
            status = self.mpu6050.readStatus()
            if(status & 0x10) == 0x10:
                raise ValueError('MPU6050 buffer overrun')

            file_raw = open('RawData.txt', 'w')
            file_raw.write(
                'GT\tGx\tGy\tGz\tTemperature\tGyrox\tGyroy\tGyroz\n')
            fft_input = []
            for loop in range(self.sample_number):
                simple_sample = values[loop * 14: loop * 14 + 14]
                intensity = self.mpu6050.convertData(simple_sample)
                current_force = math.sqrt(
                    (intensity.Gx * intensity.Gx) + (
                        intensity.Gy * intensity.Gy) + (
                            intensity.Gz * intensity.Gz))
                fft_input.append(current_force)
                file_raw.write(
                    '{0:6.3f}\t{1:6.3f}\t{2:6.3f}\t{3:6.3f}\t'.format(
                        current_force,
                        intensity.Gx, intensity.Gy, intensity.Gz))
                file_raw.write(
                    '{0:5.1f}\t{1:6.3f}\t{2:6.3f}\t{3:6.3f}\n'.format(
                        intensity.Temperature, intensity.Gyrox,
                        intensity.Gyroy, intensity.Gyroz))
            file_raw.close()

            # Hanning window for random signals
            window = np.hanning(len(fft_input)).astype(np.float32)
            fft_input_windowed = fft_input * window

            t1 = time.time()
            fft_data = np.fft.fft(fft_input_windowed)
            print('  Numpy %d samples fft computation time: %fs' % (
                self.sample_number, time.time() - t1))

            t1 = time.time()
            # bands_indexes = np.arange(0, len(fft_input) + 1).tolist()
            bands_indexes = [[i, i + 1] for i in range(
                0, len(fft_input))]
            gpu_fft_data = self.audio_levels.compute(
                np.asarray(
                    fft_input_windowed, dtype=np.float32), bands_indexes)[0]
            # gpu_fft_data[0] are values
            # gpu_fft_data[1] are means
            # gpu_fft_data[2] are std
            print('  GPU %d samples fft computation time: %fs' % (
                self.sample_number, time.time() - t1))

            file_data = open('FFTData.txt', 'w')
            fft_data = np.abs(
                fft_data[0:int(len(fft_data) / 2 + 1)]) / self.sample_number
            frequency = []
            file_data.write('Frequency\tFFT\n')
            peak = 0
            peak_index = 0
            for loop in range(int(self.sample_number / 2 + 1)):
                frequency.append(loop * self.sample_rate / self.sample_number)
                file_data.write(
                    '{0}\t{1}\n'.format(frequency[loop], fft_data[loop]))
                if loop > 0:
                    if fft_data[loop] > peak:
                        peak = fft_data[loop]
                        peak_index = loop
            file_data.close()
            print('  Numpy FFT: Peak at {0}Hz = {1}'.format(
                frequency[peak_index], peak))

            file_data = open('FFTGPUData.txt', 'w')
            gpu_fft_data = np.abs(
                gpu_fft_data[0:int(
                    len(gpu_fft_data) / 2 + 1)]) / self.sample_number
            file_data.write('Frequency\tFFT\n')
            peak = 0
            peak_index = 0
            for loop in range(int(self.sample_number / 2 + 1)):
                file_data.write(
                    '{0}\t{1}\n'.format(frequency[loop], gpu_fft_data[loop]))
                if loop > 0:
                    if gpu_fft_data[loop] > peak:
                        peak = gpu_fft_data[loop]
                        peak_index = loop
            file_data.close()
            print('  GPU FFT: Peak at {0}Hz = {1}'.format(
                frequency[peak_index], peak))

            print("Numpy FFT vs GPU FFT:")
            print(
                "Diff min:", np.min(fft_data - gpu_fft_data),
                " max:", np.max(fft_data - gpu_fft_data),
                " mean:", np.mean(fft_data - gpu_fft_data),
                " std:", np.std(fft_data - gpu_fft_data))
        del self.audio_levels


def main():
    """Benchmark accelerometer acquisition and Numpy/GPU FFT computation."""
    bench_sample_number = [256, 512, 1024]  # , 512, 1000, 1024, 10000, 100000]
    for sample_number in bench_sample_number:
        acc_fft = AccelerometerFFT(sample_number=sample_number)
        # input('Press enter to start')
        acc_fft.run()

    # Performance results:
    # PYTHON
    # Numpy  256 samples fft computation time: 0.356ms
    # GPU    256 samples fft computation time: 2.097ms
    # Numpy  512 samples fft computation time: 0.507ms
    # GPU    512 samples fft computation time: 2.546ms
    # Numpy 1024 samples fft computation time: 0.744ms
    # GPU   1024 samples fft computation time: 6.110ms

    # Same FFT performances on Pi3 CPU vs GPU. Numpy is c vectorized.
    # python3 -c "import numpy.distutils.system_info as sysinfo; print(sysinfo.show_all())" > system_info.txt
    # FOUND ATLAS

    # C++
    # pi@raspberrypi:~/GitHub/mpu6050TestInC/cpp $ sudo ./MPU6050
    # C GPU FFT  256 samples time: 0.910ms, maxi 1.241ms
    # C GPU FFT  512 samples time: 1.811ms
    # C GPU FFT 1024 samples time: 3.560ms


if __name__ == '__main__':
    main()
