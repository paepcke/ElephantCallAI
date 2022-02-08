import Jetson.GPIO as GPIO
import time
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--interval-sec', type=int, default=2, help="number of seconds LED will be on/off")

    return parser.parse_args()


def main():
    """
    This is a small script intended to test LED functionality in isolation. Attach the negative end of your
    LED to a GND pin on the Jetson and the positive end to the pin labeled 7 (or some other pin and change the variable
    'led_pin' in the function below). Consider using a resistor in series with the LED to avoid damaging the LED.

    If everything is connected properly, the LED should blink on and off approximately in time with the command line
    output indicating the corresponding behavior.
    """
    args = get_args()

    # Pin Definition
    led_pin = 7

    # Set up the GPIO channel
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(led_pin, GPIO.OUT, initial=GPIO.HIGH)

    while True:
        time.sleep(args.interval_sec)

        # importantly, this GPIO.output() function is 'sticky' - the voltage will remain set this way until changed,
        # even after this python process dies
        GPIO.output(led_pin, GPIO.HIGH)
        print("LED is ON")
        time.sleep(args.interval_sec)
        GPIO.output(led_pin, GPIO.LOW)
        print("LED is OFF")


if __name__ == "__main__":
    main()
