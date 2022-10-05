from typing import Final

CURET_INDEX_TO_LABELS: Final[dict[int, str]] = {
    1: "Felt",              2: "Polyester",         3: "Terrycloth",
    4: "Rough Plastic",     5: "Leather",           6: "Sandpaper",
    7: "Velvet",            8: "Pebbles",           9: "Frosted Glass",
    10: "Plaster_a",        11: "Plaster_b",        12: "Rough Paper",
    13: "Artificial Grass", 14: "Roof Shingle",     15: "Aluminum Foil",
    16: "Cork",             17: "Rough Tile",       18: "Rug_a",
    19: "Rug_b",            20: "Styrofoam",        21: "Sponge",
    22: "Lambswool",        23: "Lettuce Leaf",     24: "Rabbit Fur",
    25: "Quarry Tile",      26: "Loofa",            27: "Insulation",
    28: "Crumpled Paper",   29: "(2 zoomed)",       30: "(11 zoomed)",
    31: "(12 zoomed)",      32: "(14 zoomed)",      33: "Slate_a",
    34: "Slate_b",          35: "Painted Spheres",  36: "Limestone",
    37: "Brick_a",          38: "Ribbed Paper",     39: "Human Skin",
    40: "Straw",            41: "Brick_b",          42: "Corduroy",
    43: "Salt Crystals",    44: "Linen",            45: "Concrete_a",
    46: "Cotton",           47: "Stones",           48: "Brown Bread",
    49: "Concrete_b",       50: "Concrete_c",       51: "Corn Husk",
    52: "White Bread",      53: "Soleirolia Plant", 54: "Wood_a",
    55: "Orange Peel",      56: "Wood_b",           57: "Peacock Feather",
    58: "Tree Bark",        59: "Cracker_a",        60: "Cracker_b",
    61: "Moss"
}
"""
Map 1-based index to the associated descriptive class name.
"""

CURET_INDEX_TO_FOLDER: Final[dict[int, str]] = {
    x: f"sample{x:02}" for x in range(1, 61 + 1)}
"""
Map 1-based index to the associated sample folder names.
"""

# some sanity checks
assert len(CURET_INDEX_TO_LABELS) == len(CURET_INDEX_TO_FOLDER)
assert CURET_INDEX_TO_LABELS.keys() == CURET_INDEX_TO_FOLDER.keys()
