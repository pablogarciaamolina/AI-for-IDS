import logging

def log_backoff(details):
    """
    Method for logging a warning when a retry is triggered
    """

    wait_time = details['wait']
    attempt = details['tries']
    exception = details['exception']
    logging.warning(f"Rate limit hit. Retrying in {wait_time:.2f}s... (Attempt {attempt}) | Error: {exception}")