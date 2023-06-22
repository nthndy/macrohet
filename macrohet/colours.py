"""
colours.py

This module defines custom color maps for visualizations.

Custom Color Maps:
- lavender_raisin: A color map featuring shades of lavender raisin.
- expanded_piyg: An expanded version of the 'PiYG' color map.

Each color map is represented as a list of color codes.

"""

# Define the lavender_raisin color map
lavender_raisin = [
    'd8d8f6',   # Light Lavender Raisin
    'b18fcf',   # Medium Lavender Raisin
    '978897',   # Rose Quartz
    '494850',   # Dark Lavender Raisin
    '2c2c34'    # Darkest Lavender Raisin
]

# Define the expanded_piyg color map
expanded_piyg = [
    '#1a9641',  # First color from sns.color_palette('PiYG')
    '#a6d96a',  # Second color from sns.color_palette('PiYG')
    '#978897',  # Rose Quartz in hexadecimal format
    '#d1d1ca',  # Timberwolf in RGB normalized form
    '#f1b6da',  # Second-to-last color from sns.color_palette('PiYG')
    '#d02c91'   # Last color from sns.color_palette('PiYG')
]

# Define the custom_colours dictionary
custom_colours = {
    'lavender_raisin': lavender_raisin,
    'expanded_piyg': expanded_piyg
}
