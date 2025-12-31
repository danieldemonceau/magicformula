"""Ranking utilities for investment strategies."""

import pandas as pd

from src.data.types import StrategyResultDict


def rank_series(
    data: list[StrategyResultDict],
    key: str,
    ascending: bool = False,
    method: str = "min",
) -> list[StrategyResultDict]:
    """Rank a list of dictionaries by a specific key.

    Uses pandas ranking to handle ties properly.

    Args:
        data: List of dictionaries to rank.
        key: Key to rank by.
        ascending: If True, lower values get better ranks.
        method: Ranking method ('min', 'max', 'average', 'first', 'dense').

    Returns:
        List of dictionaries with added rank column.
    """
    if not data:
        return data

    df = pd.DataFrame(data)
    rank_key = f"{key}_rank"

    if key not in df.columns:
        return data

    df[rank_key] = df[key].rank(ascending=ascending, method=method).astype(int)

    result: list[StrategyResultDict] = df.to_dict("records")
    return result


def calculate_combined_rank(
    data: list[StrategyResultDict],
    rank_keys: list[str],
    output_key: str = "combined_rank",
) -> list[StrategyResultDict]:
    """Calculate combined rank from multiple rank columns.

    Uses vectorized pandas operations for efficiency.

    Args:
        data: List of dictionaries with rank columns.
        rank_keys: List of rank column keys to sum.
        output_key: Key name for the combined rank.

    Returns:
        List of dictionaries with combined rank added.
    """
    if not data:
        return data

    df = pd.DataFrame(data)

    # Vectorized sum operation
    if all(key in df.columns for key in rank_keys):
        df[output_key] = df[rank_keys].sum(axis=1)
    else:
        # Fallback for missing columns
        for item in data:
            combined = 0
            for key in rank_keys:
                val = item.get(key, 0)
                if isinstance(val, (int, float)):
                    combined += int(val)
            item[output_key] = combined
        return data

    result: list[StrategyResultDict] = df.to_dict("records")
    return result
