"""Date utility functions for business day calculations."""

from datetime import date, timedelta
from dateutil.relativedelta import relativedelta


def get_last_business_day(target_date: date | None = None) -> date:
    """Get the last business day before or on the target date.

    Args:
        target_date: Target date. If None, uses today.

    Returns:
        Last business day (Monday-Friday).
    """
    if target_date is None:
        target_date = date.today()

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
    return months_ago

