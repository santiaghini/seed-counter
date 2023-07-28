# NOTE: This will be the distance threshold used to separate seeds that are too close together
# Increase if seeds are closer together, decrease if seeds are farther apart
# Balance: the smaller this is set, the more likely seeds will be separated BUT the more likely smaller (or dimmer) seeds will be left out
DISTANCE_THRESHOLDS = {
    'BFTL': 10,
    'RFP': 14
}

# NOTE: This is a number between 0 and 255 (max value) that will be used to threshold image to capture seeds according to brightness
# Increase if seeds are brighter, decrease if seeds are darker
# Balance: the smaller this is set, the more likely seeds will be captured BUT the more likely noise will be captured
INITIAL_BRIGHTNESS_THRESHOLDS = {
    'BFTL': 235,
    'RFP': 230
}

# Change according to scale of image and size of seeds
SMALL_AREA_PRE_PASS = 100
SMALL_AREA_POST_PASS = 5

# Change according to size of scale bar
SCALE_BAR_HEIGHT = 50
SCALE_BAR_WIDTH = 500