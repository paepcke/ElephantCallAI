from time import sleep


def yield_to_thread_scheduler():
    """
    Gives other threads competing for the same resources as the calling thread the opportunity to acquire them.
    Prevents thread starvation.

    :return:
    """
    # This time doesn't matter, we just need to force the scheduler to schedule another thread if possible
    sleep(0.00001)
