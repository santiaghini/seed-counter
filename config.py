BRIGHTFIELD = "BF"
FLUORESCENT = "FL"

# NOTE: This is a number between 0 and 255 (max value) that will be used to threshold image to capture seeds according to brightness
# Increase if seeds are brighter, decrease if seeds are darker
# Balance: the smaller this is set, the more likely seeds will be captured BUT the more likely noise will be captured
INITIAL_BRIGHTNESS_THRESHOLDS = {
    BRIGHTFIELD: 30,
    FLUORESCENT: 15
}

# Change according to scale of image and size of seeds
SMALL_AREA_PRE_PASS = 200
SMALL_AREA_POST_PASS = 5

# Change according to size of scale bar
SCALE_BAR_HEIGHT = 50
SCALE_BAR_WIDTH = 500

# NOTE: This will be the distance threshold used to separate seeds that are too close together
# Increase if seeds are closer together, decrease if seeds are farther apart
# Balance: the smaller this is set, the more likely seeds will be separated BUT the more likely smaller (or dimmer) seeds will be left out
# NOTE: DEPRECATED, now we calculate it based on reference values
DISTANCE_THRESHOLDS = {
    BRIGHTFIELD: 10,
    FLUORESCENT: 12
}

TARGET_RATIO = 0.75