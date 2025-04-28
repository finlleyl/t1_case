def detect_text_blocks(image):

    height, width, _ = image.shape

    return [
        {
            "x": width * 0.25,
            "y": height * 0.25,
            "width": width * 0.5,
            "height": height * 0.1,
        },
        {
            "x": width * 0.25,
            "y": height * 0.25,
            "width": width * 0.5,
            "height": height * 0.1,
        },
    ]
