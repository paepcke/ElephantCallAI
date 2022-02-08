import time
from datetime import datetime, timedelta
from threading import Lock, Thread
import Jetson.GPIO as GPIO

from embedded.Closeable import Closeable

ILLUMINATE_DURATION_SEC = 2
ILLUMINATE_CHECK_PERIOD_SEC = 0.5

STARTUP_FLASH_CYCLES = 5
STARTUP_FLASH_PERIOD_SEC = 1


class LedController(Closeable):
    """
    An object that encapsulates the handling of an LED connected to a GPIO port on the Jetson board.
    """
    def __init__(self, led_pin: int = 7, verbose: bool = False):
        super().__init__()
        self.led_lock: Lock = Lock()
        self.led_pin: int = led_pin
        self.verbose: bool = verbose
        self.on_until_time = datetime.now()
        self.closed: bool = False

        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.led_pin, GPIO.OUT, initial=GPIO.LOW)

        # flash the LED a few times on startup so we know it's working correctly
        self._flash_led_on_startup()

        self.led_thread: Thread = Thread(target=self._periodically_set_led, daemon=True)
        self.led_thread.start()

    def illuminate(self):
        if self.verbose:
            print("Illuminating LED...")

        self.led_lock.acquire()
        self.on_until_time = datetime.now() + timedelta(seconds=ILLUMINATE_DURATION_SEC)
        self.led_lock.release()

    def _flash_led_on_startup(self):
        """
        This method helps a user verify their LED is properly connected and functional.
        """
        if self.verbose:
            print(f"Flashing LED {STARTUP_FLASH_CYCLES} times to verify it's properly connected - please check LED")

        for i in range(STARTUP_FLASH_CYCLES):
            GPIO.output(self.led_pin, GPIO.HIGH)
            time.sleep(STARTUP_FLASH_PERIOD_SEC)
            GPIO.output(self.led_pin, GPIO.LOW)
            time.sleep(STARTUP_FLASH_PERIOD_SEC)

        if self.verbose:
            print(f"LED startup sequence complete. If you did not see the LED flash, there is a problem.")

    def _periodically_set_led(self):
        while not self.closed:
            self.led_lock.acquire()
            on_until_time = self.on_until_time
            self.led_lock.release()

            if on_until_time > datetime.now():
                GPIO.output(self.led_pin, GPIO.HIGH)
            else:
                GPIO.output(self.led_pin, GPIO.LOW)

            time.sleep(ILLUMINATE_CHECK_PERIOD_SEC)

    def close(self):
        self.closed = True
        self.led_thread.join()
        GPIO.output(self.led_pin, GPIO.LOW)
