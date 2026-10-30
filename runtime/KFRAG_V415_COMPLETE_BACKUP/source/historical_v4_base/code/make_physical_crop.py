def make_physical_crop(
    image,
    crop_type,
    retain,
    rng,
):
    """
    image:
        [1,3,256,256]

    returns:
        crop:
            [1,3,h,w]

        metadata:
            dict with x/y/w/h
    """

    if image.shape != (1, 3, 256, 256):
        raise ValueError(
            f"Expected [1,3,256,256], got {tuple(image.shape)}"
        )


    # ----------------------------------------------------------------------------------------------
    # RANDOM SQUARE
    # ----------------------------------------------------------------------------------------------

    if crop_type == "random_square":

        side = int(
            round(
                256 * retain
            )
        )

        side = max(
            16,
            min(
                256,
                side,
            ),
        )

        max_xy = (
            256
            -
            side
        )

        x = rng.randint(
            0,
            max_xy,
        )

        y = rng.randint(
            0,
            max_xy,
        )

        w = side
        h = side


    # ----------------------------------------------------------------------------------------------
    # CENTER
    # ----------------------------------------------------------------------------------------------

    elif crop_type == "center":

        side = int(
            round(
                256 * retain
            )
        )

        side = max(
            16,
            min(
                256,
                side,
            ),
        )

        w = side
        h = side

        x = (
            256
            -
            w
        ) // 2

        y = (
            256
            -
            h
        ) // 2


    # ----------------------------------------------------------------------------------------------
    # CORNER
    # ----------------------------------------------------------------------------------------------

    elif crop_type == "corner":

        side = int(
            round(
                256 * retain
            )
        )

        side = max(
            16,
            min(
                256,
                side,
            ),
        )

        w = side
        h = side

        corner = rng.choice(
            [
                "top_left",
                "top_right",
                "bottom_left",
                "bottom_right",
            ]
        )

        if corner == "top_left":

            x = 0
            y = 0

        elif corner == "top_right":

            x = 256 - w
            y = 0

        elif corner == "bottom_left":

            x = 0
            y = 256 - h

        else:

            x = 256 - w
            y = 256 - h


    # ----------------------------------------------------------------------------------------------
    # EDGE
    # ----------------------------------------------------------------------------------------------

    elif crop_type == "edge":

        side = int(
            round(
                256 * retain
            )
        )

        side = max(
            16,
            min(
                256,
                side,
            ),
        )

        w = side
        h = side

        edge = rng.choice(
            [
                "top",
                "bottom",
                "left",
                "right",
            ]
        )

        if edge == "top":

            y = 0

            x = rng.randint(
                0,
                256 - w,
            )

        elif edge == "bottom":

            y = 256 - h

            x = rng.randint(
                0,
                256 - w,
            )

        elif edge == "left":

            x = 0

            y = rng.randint(
                0,
                256 - h,
            )

        else:

            x = 256 - w

            y = rng.randint(
                0,
                256 - h,
            )


    # ----------------------------------------------------------------------------------------------
    # RANDOM RECTANGLE
    #
    # Original benchmark formulation:
    #
    # target area = 256^2 * retain^2
    # aspect ratio sampled log-uniformly 0.60 - 1.67
    # ----------------------------------------------------------------------------------------------

    elif crop_type == "random_rectangle":

        target_area = (
            256
            *
            256
            *
            retain
            *
            retain
        )

        ratio = math.exp(
            rng.uniform(
                math.log(
                    0.60
                ),
                math.log(
                    1.67
                ),
            )
        )

        w = int(
            round(
                math.sqrt(
                    target_area
                    *
                    ratio
                )
            )
        )

        h = int(
            round(
                math.sqrt(
                    target_area
                    /
                    ratio
                )
            )
        )

        w = max(
            32,
            min(
                256,
                w,
            ),
        )

        h = max(
            32,
            min(
                256,
                h,
            ),
        )

        x = rng.randint(
            0,
            256 - w,
        )

        y = rng.randint(
            0,
            256 - h,
        )


    else:

        raise ValueError(
            f"Unknown crop type: {crop_type}"
        )


    crop = image[
        :,
        :,
        y:y+h,
        x:x+w,
    ]


    metadata = {
        "crop_type":
            crop_type,

        "retain":
            float(
                retain
            ),

        "x":
            int(
                x
            ),

        "y":
            int(
                y
            ),

        "w":
            int(
                w
            ),

        "h":
            int(
                h
            ),
    }


    return (
        crop,
        metadata,
    )
