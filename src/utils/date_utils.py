"""Date utility functions for business day calculations."""

from datetime import date, timedelta

from dateutil.relativedelta import relativedelta


def get_last_business_day(target_date: date | None = None) -> date:
    """Get the last business day on or before the target date.

    If target_date is a business day, returns target_date.
    Otherwise, returns the most recent business day before target_date.

    Args:
        target_date: Target date. If None, uses today.

    Returns:
        Last business day (Monday-Friday) on or before target_date.
    """
    if target_date is None:
        target_date = date.today()

    # If target_date is already a business day, return it
    if target_date.weekday() < 5:  # Monday-Friday (0-4)
        return target_date

    # Otherwise, go back to find the last business day
    last_business = target_date - timedelta(days=1)
    while last_business.weekday() > 4:  # Saturday = 5, Sunday = 6
        last_business -= timedelta(days=1)
    return last_business


def get_business_day_months_ago(months: int, target_date: date | None = None) -> date:
    """Get a business day N months ago from the target date.

    Args:
        months: Number of months to go back (negative for past).
        target_date: Target date. If None, uses today.

    Returns:
        Business day N months ago.
    """
    if target_date is None:
        target_date = date.today()

    last_business = get_last_business_day(target_date)
    months_ago = last_business + relativedelta(months=-months)

    while months_ago.weekday() > 4:
        months_ago -= timedelta(days=1)
    result: date = months_ago
    return result
