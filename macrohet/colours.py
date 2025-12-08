"""colours.py

This module defines custom color maps for visualizations.

Custom Color Maps:
- lavender_raisin: A color map featuring shades of lavender raisin.
- expanded_piyg: An expanded version of the 'PiYG' color map.
- vaporwave: A vaporwave-inspired color map.
- mint_taupe: A color map inspired by natural tones of mint and taupe.
- yellow_pig: A new color map featuring a combination of vibrant colors.

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

# Define the vaporwave color map
vaporwave = [
    '#D02C91',  # Purple Sunset Glow: A rich, warm purple, like the vibrant colors of a sunset.
    '#F1C2F2',  # Blush of the Morning Sky: A soft pastel pink, like the first light of dawn.
    '#C291F2',  # Lavender Breeze: A soothing lavender, like the gentle breeze over a lavender field.
    '#564D8C',  # Midnight Mountain: A deep, cool purple, evoking the silhouette of mountains under the night sky.
    '#57AAF2',  # Ocean's Whisper: A tranquil blue, reflecting the calm of the sea.
    '#A0D9D9',  # Misty Lagoon: A soft, serene blue-green, like mist hovering over a quiet lagoon.
    '#A6D96A',  # Spring Meadow: A vibrant green, capturing the freshness of a meadow in spring.
    '#1A9641'   # Pine Grove Green: A deep, earthy green, reminiscent of a dense pine forest.
]

# Define the mint_taupe color map
mint_taupe = [
    '#0D1321',  # Midnight Horizon: A deep, dark navy, like the sky just before dawn.
    '#1D8FE0',  # Cerulean Splash: A bright, refreshing blue, reminiscent of a clear, sunny day at the ocean.
    '#C5D86D',  # Meadow Mist: A soft, gentle green, like dew on the early morning grass.
    '#ADD9C5',  # Seafoam Whisper: A calming, muted aqua, like the soft lull of ocean waves.
    '#FFEDDF',  # Morning Glow: A light, warm peach, like the first rays of sunlight touching the clouds.
    '#8C6764',  # Rustic Canyon: An earthy, rich brown, reminiscent of the hues of a rugged canyon landscape.
    '#CC5105',  # Autumn Ember: A fiery orange, like the vibrant leaves of fall or the warmth of a glowing hearth.
    '#8D7494'   # Dusky Plum: A muted purple, like the soft twilight just before the night fully sets in.
]

# Define the super_expiyg color map (extracted from the provided image)
super_expiyg = [
    '#1A9641',  # Deep Green
    '#7BDAA4',  # Mint Green
    '#F0BE38',  # Golden Yellow
    '#F0D795',  # Light Beige
    '#EF93B9',  # Soft Pink
    '#D12B82',  # Bright Magenta
    '#9F9BC7',  # Lavender Blue
    '#452A61'   # Dark Purple
]

# Define the custom_colours dictionary
custom_colours = {
    'lavender_raisin': lavender_raisin,
    'expanded_piyg': expanded_piyg,
    'vaporwave': vaporwave,
    'mint_taupe': mint_taupe,
    'super_expiyg': super_expiyg
}
