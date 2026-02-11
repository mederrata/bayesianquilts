#!/usr/bin/env python3
"""RWA (Right-Wing Authoritarianism) scale data loader.

Downloads and preprocesses the RWA dataset from OpenPsychometrics.org.
22 items, 9 response categories (0-8), 2 latent dimensions.
"""

import os
import urllib.request
import zipfile
from pathlib import Path

import grain
import polars as pl

# =========================================================================
# Module constants
# =========================================================================

item_keys = [f"Q{i}" for i in range(1, 23)]

item_text = [
    'The established authorities generally turn out to be right about things, while the radicals and protestors are usually just "loud mouths" showing off their ignorance.',
    "Women should have to promise to obey their husbands when they get married.",
    "Our country desperately needs a mighty leader who will do what has to be done to destroy the radical new ways and sinfulness that are ruining us.",
    "Gays and lesbians are just as healthy and moral as anybody else.",
    "It is always better to trust the judgement of the proper authorities in government and religion than to listen to the noisy rabble-rousers in our society who are trying to create doubt in people's minds.",
    "Atheists and others who have rebelled against the established religions are no doubt every bit as good and virtuous as those who attend church regularly.",
    "The only way our country can get through the crisis ahead is to get back to our traditional values, put some tough leaders in power, and silence the troublemakers spreading bad ideas.",
    "There is absolutely nothing wrong with nudist camps.",
    "Our country needs free thinkers who have the courage to defy traditional ways, even if this upsets many people.",
    "Our country will be destroyed someday if we do not smash the perversions eating away at our moral fiber and traditional beliefs.",
    "Everyone should have their own lifestyle, religious beliefs, and sexual preferences, even if it makes them different from everyone else.",
    'The "old-fashioned ways" and the "old-fashioned values" still show the best way to live.',
    "You have to admire those who challenged the law and the majority's view by protesting for women's abortion rights, for animal rights, or to abolish school prayer.",
    "What our country really needs is a strong, determined leader who will crush evil, and take us back to our true path.",
    'Some of the best people in our country are those who are challenging our government, criticizing religion, and ignoring the "normal way things are supposed to be done."',
    "God's laws about abortion, pornography and marriage must be strictly followed before it is too late, and those who break them must be strongly punished.",
    "There are many radical, immoral people in our country today, who are trying to ruin it for their own godless purposes, whom the authorities should put out of action.",
    'A "woman\'s place" should be wherever she wants to be. The days when women are submissive to their husbands and social conventions belong strictly in the past.',
    'Our country will be great if we honor the ways of our forefathers, do what the authorities tell us to do, and get rid of the "rotten apples" who are ruining everything.',
    'There is no "one right way" to live life; everybody has to create their own way.',
    "Homosexuals and feminists should be praised for being brave enough to defy \"traditional family values.\"",
    "This country would work a lot better if certain groups of troublemakers would just shut up and accept their group's traditional place in society.",
]

# 0-indexed item positions to reverse-score
to_reverse = [3, 5, 7, 8, 10, 12, 14, 17, 19, 20]

# Scale indices: two latent dimensions
scale_indices = [
    [1, 3, 5, 7, 8, 10, 12, 14, 15, 17, 19, 20],
    [0, 2, 4, 6, 9, 11, 13, 16, 18, 21],
]

# =========================================================================
# Data source for grain
# =========================================================================

_DATA_URL = "https://openpsychometrics.org/_rawdata/RWAS.zip"


class _ArrayDataSource(grain.sources.RandomAccessDataSource):
    """Wraps a polars DataFrame as a grain RandomAccessDataSource."""

    def __init__(self, df: pl.DataFrame):
        self._data = {col: df[col].to_list() for col in df.columns}
        self.n = len(df)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self._data.items()}

    def __len__(self):
        return self.n


# =========================================================================
# Public API
# =========================================================================


def get_data(reorient=False, polars_out=False, cache_dir=None):
    """Load the RWA dataset.

    Downloads from OpenPsychometrics if not already cached.

    Args:
        reorient: If True, reverse-score items in ``to_reverse``.
        polars_out: If True, return a polars DataFrame instead of grain dataset.
        cache_dir: Directory to cache downloaded data. Defaults to cwd.

    Returns:
        Tuple of (dataset, num_people) where dataset is a grain MapDataset
        (default) or polars DataFrame (if polars_out=True).
    """
    if cache_dir is None:
        cache_dir = Path.cwd()
    else:
        cache_dir = Path(cache_dir)

    csv_path = cache_dir / "RWAS" / "data.csv"

    if not csv_path.exists():
        zip_path = cache_dir / "RWAS.zip"
        os.makedirs(cache_dir, exist_ok=True)
        urllib.request.urlretrieve(_DATA_URL, str(zip_path))
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(cache_dir))

    data = pl.read_csv(str(csv_path))
    data = data.select(item_keys)

    # Shift from 1-9 to 0-8
    data = data.with_columns([
        (pl.col(k) - 1).alias(k) for k in item_keys
    ])

    num_people = len(data)

    if reorient:
        data = data.with_columns([
            (8 - pl.col(item_keys[i])).alias(item_keys[i])
            for i in to_reverse
        ])

    # Mark invalid values as -1
    data = data.with_columns([
        pl.when(pl.col(k) > 8).then(-1).otherwise(pl.col(k)).alias(k)
        for k in item_keys
    ])

    # Add person index
    data = data.with_row_index("person")

    if polars_out:
        return data, num_people

    dataset = grain.MapDataset.source(_ArrayDataSource(data))
    return dataset, num_people
