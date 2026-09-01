CLASS_LABELS = {0: "bare", 1: "grass", 2: "shrub", 3: "tree"}
CLASS_NAMES = ["bare", "grass", "shrub", "tree"]
CLASS_CODES = [0, 1, 2, 3]
CLASS_COLORS = {0: "#c2b280", 1: "#7cb342", 2: "#8d6e63", 3: "#1b5e20"}
CLASS_ORDER = ["bare", "grass", "shrub", "tree"]
UNLABELLED_COLOR = "#e6007e"  # deliberately outside the earth/green family
CLUSTER_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
]

NODATA = 255

# Shadow mask codes, written by run_stage1_6_detect_shadows.py.
# ONLY SHADOW_IS_NODATA IS A LOSS. SHADOW_IS_TREE is shadow within
# SHADOW_TREE_RADIUS of CHM >= H_TREE_MIN, which instructions5.md section 5
# Step 1c assigns to the tree class - those pixels are classified, not
# discarded. Treating SHADOW_IS_TREE as loss overstates the cost of shadow by
# roughly ten times (2.0-3.1% per tile against a true loss of 0.05-1.0%), which
# is a mistake that has already been made once and reached the results notes.
NO_SHADOW = 0
SHADOW_IS_TREE = 1
SHADOW_IS_NODATA = 2

SHADOW_CODE_LABELS = {NO_SHADOW: "not shadow", SHADOW_IS_TREE: "resolved to tree", SHADOW_IS_NODATA: "masked to nodata"}

FRAMEWORK_ORDER = ["A", "B", "C", "D", "E"]

# used for spacing
SEVENTY = 70
