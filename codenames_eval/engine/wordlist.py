"""Word pools for board construction and plant injection.

BOARD_WORD_POOL: codenames-style common nouns used to build 25-word boards.
BINDING_WORD_POOL: words reserved for planted binding episodes. Disjoint from
    the board pool so a binding word is always a legal clue (never on board).
FILLER_TOPIC_WORDS: vocabulary for template-mode filler conversation, disjoint
    from both pools so filler turns can never create accidental co-occurrences
    of plantable concepts.
"""

BOARD_WORD_POOL = [
    "apple", "anchor", "arm", "arrow", "badge", "ball", "band", "bank",
    "bark", "barrel", "bat", "beach", "bear", "beard", "bell", "belt",
    "bench", "berry", "board", "bolt", "bomb", "bone", "book", "boot",
    "bottle", "bow", "box", "brick", "bridge", "brush", "bucket", "bug",
    "button", "cable", "cake", "camp", "candle", "cap", "card", "carpet",
    "carrot", "castle", "cat", "cave", "chain", "chair", "chalk", "cheese",
    "cherry", "chest", "chicken", "church", "circle", "cliff", "clock",
    "cloud", "clown", "coach", "coast", "coat", "coin", "comb", "comet",
    "compass", "cook", "copper", "corn", "court", "crane", "cross", "crown",
    "cup", "curtain", "cycle", "dance", "deck", "deer", "desk", "diamond",
    "dice", "dog", "doll", "door", "dragon", "dress", "drill", "drum",
    "duck", "eagle", "egg", "engine", "fair", "fan", "farm", "feather",
    "fence", "field", "file", "film", "fire", "fish", "flag", "flute",
    "fog", "fork", "fort", "frame", "frog", "game", "garden", "gate",
    "ghost", "giant", "glass", "glove", "goat", "gold", "grass", "guitar",
    "hammer", "harbor", "hat", "hawk", "heart", "hedge", "helmet", "hill",
    "hook", "horn", "horse", "hotel", "house", "ice", "iron", "island",
    "ivory", "jacket", "jail", "jam", "jewel", "judge", "kettle", "key",
    "king", "kite", "knife", "knight", "ladder", "lake", "lamp", "lantern",
    "leaf", "lemon", "letter", "lion", "lock", "log", "luck", "magnet",
    "mail", "mammoth", "map", "marble", "mask", "match", "maze", "medal",
    "mine", "mint", "mirror", "monkey", "moon", "moss", "moth", "mountain",
    "mouse", "mouth", "mule", "nail", "needle", "nest", "net", "night",
    "nurse", "oak", "oar", "ocean", "office", "onion", "opera", "orange",
    "organ", "owl", "paddle", "paint", "palace", "palm", "paper", "parade",
    "park", "pearl", "pen", "piano", "pie", "pillow", "pilot", "pipe",
    "pirate", "pistol", "pitch", "plane", "plate", "pond", "pool", "port",
    "pot", "prince", "queen", "quilt", "rabbit", "raft", "rail", "rain",
    "rake", "ranch", "rat", "razor", "ring", "river", "robot", "rock",
    "rocket", "rope", "rose", "ruler", "saddle", "sail", "salt", "sand",
    "satellite", "saw", "scale", "school", "scorpion", "screen", "seal",
    "shadow", "shark", "shed", "sheep", "shell", "shield", "ship", "shoe",
    "shop", "shovel", "silk", "silver", "sink", "skate", "ski", "sled",
    "smoke", "snake", "snow", "soap", "sock", "spear", "sphinx", "spider",
    "sponge", "spoon", "spring", "spy", "square", "stable", "stadium",
    "stamp", "star", "stone", "stool", "storm", "stove", "straw", "string",
    "sugar", "suit", "sun", "swamp", "swan", "sword", "table", "tail",
    "tank", "tape", "teacher", "telescope", "temple", "tent", "theater",
    "thief", "thumb", "ticket", "tiger", "tin", "torch", "tower", "track",
    "train", "trap", "tree", "triangle", "trophy", "truck", "trumpet",
    "tunnel", "turkey", "turtle", "umbrella", "vase", "vault", "violin",
    "wagon", "wall", "watch", "wave", "web", "well", "whale", "wheel",
    "whip", "whistle", "window", "wing", "witch", "wolf", "wood", "wool",
    "worm", "yard", "zebra", "zoo",
]

BINDING_WORD_POOL = [
    "gondola", "zeppelin", "thimble", "marmalade", "periscope", "sundial",
    "abacus", "gargoyle", "hammock", "kazoo", "monocle", "origami",
    "pinwheel", "quicksand", "rickshaw", "scarecrow", "tambourine",
    "unicycle", "vinegar", "waffle", "xylophone", "yodel", "avalanche",
    "bagpipe", "catapult", "dominoes", "eclipse", "flamingo", "geyser",
    "harmonica", "iceberg", "jigsaw", "kaleidoscope", "labyrinth",
    "metronome", "nutmeg", "obelisk", "parachute", "quill", "rhubarb",
    "stalactite", "toboggan", "ukulele", "volcano", "windmill", "yo-yo",
    "zamboni", "accordion", "boomerang", "carousel", "dandelion",
    "escalator", "fondue", "gazebo", "hologram", "igloo", "jukebox",
    "kayak", "lasso", "mosaic", "nectarine", "octagon", "pendulum",
]

FILLER_TOPIC_WORDS = [
    "meeting", "deadline", "recipe", "weekend", "traffic", "weather",
    "podcast", "workout", "coffee", "budget", "birthday", "vacation",
    "neighbor", "laundry", "grocery", "commute", "haircut", "plumber",
    "insurance", "taxes", "furniture", "internet", "phone", "email",
    "calendar", "appointment", "dentist", "doctor", "restaurant",
    "leftovers", "gardening", "painting", "shelf", "closet", "garage",
]

assert not set(BOARD_WORD_POOL) & set(BINDING_WORD_POOL)
assert not set(BOARD_WORD_POOL) & set(FILLER_TOPIC_WORDS)
assert not set(BINDING_WORD_POOL) & set(FILLER_TOPIC_WORDS)
