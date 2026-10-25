"""Product identity and theme settings, independent of detection behavior."""

import os
import re


def _theme_color(name, default):
    value = os.getenv(name, default)
    return value if re.fullmatch(r"#[0-9a-fA-F]{6}", value) else default


BRAND = {
    "name": os.getenv("PRODUCT_NAME", "Verity AI"),
    "tagline": os.getenv("PRODUCT_TAGLINE", "Know what you're reading."),
    "footer": os.getenv("PRODUCT_FOOTER", "Text analysis, with context."),
    "primary": _theme_color("PRODUCT_PRIMARY_COLOR", "#3867e8"),
    "background": _theme_color("PRODUCT_BACKGROUND_COLOR", "#f5f7fb"),
}