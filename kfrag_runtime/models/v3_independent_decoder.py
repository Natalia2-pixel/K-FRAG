import copy
import torch
import torch.nn as nn

class V3IndependentIndexDecoder(nn.Module):

    def __init__(
        self,
        highpass,
        carrier,
    ):

        super().__init__()

        self.highpass = copy.deepcopy(
            highpass
        )

        self.carrier = copy.deepcopy(
            carrier
        )

        self.image_size = (
            self.carrier.image_size
        )

        self.cell_size = (
            self.carrier.cell_size
        )

        self.mix = nn.Linear(
            4,
            4,
        )

        nn.init.eye_(
            self.mix.weight
        )

        nn.init.zeros_(
            self.mix.bias
        )


    def forward(
        self,
        questioned_image,
    ):

        if (
            questioned_image.ndim != 4
            or
            tuple(
                questioned_image.shape[
                    1:
                ]
            )
            !=
            (
                3,
                self.image_size,
                self.image_size,
            )
        ):

            raise ValueError(
                "V3 index decoder expects [B,3,64,64]"
            )


        b = len(
            questioned_image
        )


        cells = (
            questioned_image
            .reshape(
                b,
                3,
                4,
                self.cell_size,
                4,
                self.cell_size,
            )
            .permute(
                0,
                2,
                4,
                1,
                3,
                5,
            )
            .reshape(
                b * 16,
                3,
                self.cell_size,
                self.cell_size,
            )
        )


        hp = self.highpass(
            cells
        )


        bases = (
            self.carrier.bases
            *
            self.carrier.window
        )


        hp_bases = self.highpass(
            bases[
                :,
                None,
            ].expand(
                -1,
                3,
                -1,
                -1,
            )
        )


        hp_bases = (
            hp_bases
            /
            hp_bases
            .square()
            .mean(
                (
                    -2,
                    -1,
                ),
                keepdim=True,
            )
            .sqrt()
            .clamp_min(
                1e-6
            )
        )


        matched = (
            25
            *
            torch.einsum(
                "bchw,kchw->bk",
                hp,
                hp_bases,
            )
            /
            (
                hp.shape[1]
                *
                hp.shape[2]
                *
                hp.shape[3]
            )
        )


        return (
            self.mix(
                matched
            )
            .reshape(
                b,
                4,
                4,
                4,
            )
        )
# ============================================================
# 2. LIGHTWEIGHT INDEPENDENT AUTH DECODER
#
# Reproduces BlindTagHead computation but owns only:
#
#   highpass
#   carrier
#   mix
#
# This matches the six saved auth.* entries exactly.
# ============================================================

class V3IndependentAuthDecoder(nn.Module):

    def __init__(
        self,
        highpass,
        carrier,
    ):

        super().__init__()

        self.highpass = copy.deepcopy(
            highpass
        )

        self.carrier = copy.deepcopy(
            carrier
        )

        self.image_size = (
            self.carrier.image_size
        )

        self.cell_size = (
            self.carrier.cell_size
        )

        self.mix = nn.Linear(
            32,
            32,
        )

        nn.init.eye_(
            self.mix.weight
        )

        nn.init.zeros_(
            self.mix.bias
        )


    def forward(
        self,
        questioned_image,
    ):

        if (
            questioned_image.ndim != 4
            or
            tuple(
                questioned_image.shape[
                    1:
                ]
            )
            !=
            (
                3,
                self.image_size,
                self.image_size,
            )
        ):

            raise ValueError(
                "V3 auth decoder expects [B,3,64,64]"
            )


        b = len(
            questioned_image
        )


        cells = (
            questioned_image
            .reshape(
                b,
                3,
                4,
                self.cell_size,
                4,
                self.cell_size,
            )
            .permute(
                0,
                2,
                4,
                1,
                3,
                5,
            )
            .reshape(
                b * 16,
                3,
                self.cell_size,
                self.cell_size,
            )
        )


        hp = self.highpass(
            cells
        )


        bases = (
            self.carrier.bases
            *
            self.carrier.window
        )


        responses = self.highpass(
            bases[
                :,
                None,
            ].expand(
                -1,
                3,
                -1,
                -1,
            )
        )


        responses = (
            responses
            /
            responses
            .square()
            .mean(
                (
                    -2,
                    -1,
                ),
                keepdim=True,
            )
            .sqrt()
            .clamp_min(
                1e-6
            )
        )


        matched = (
            25
            *
            torch.einsum(
                "bchw,kchw->bk",
                hp,
                responses,
            )
            /
            (
                hp.shape[1]
                *
                hp.shape[2]
                *
                hp.shape[3]
            )
        )


        return (
            self.mix(
                matched
            )
            .reshape(
                b,
                4,
                4,
                32,
            )
        )


# ============================================================
# 3. V3 INDEPENDENT DECODER
#
# source = independent BlindRegionalDecoder
# index  = independent analytical index decoder
# auth   = independent analytical tag decoder
# ============================================================

class V3IndependentDecoder(nn.Module):

    def __init__(
        self,
        v2_model,
    ):

        super().__init__()


        # ----------------------------------------------------
        # Source / RS-symbol decoder
        #
        # Saved state:
        #   source.* = 48 tensors
        #
        # Exactly matches BlindRegionalDecoder state.
        # ----------------------------------------------------

        self.source = copy.deepcopy(
            v2_model
            .parent
            .stage_c
            .decoder
        )


        # ----------------------------------------------------
        # Independent index decoder
        # ----------------------------------------------------

        self.index = (
            V3IndependentIndexDecoder(
                highpass=(
                    v2_model
                    .parent
                    .stage_c
                    .decoder
                    .decoder
                    .highpass
                ),

                carrier=(
                    v2_model
                    .parent
                    .index_carrier
                ),
            )
        )


        # ----------------------------------------------------
        # Independent auth/tag decoder
        # ----------------------------------------------------

        self.auth = (
            V3IndependentAuthDecoder(
                highpass=(
                    v2_model
                    .parent
                    .stage_c
                    .decoder
                    .decoder
                    .highpass
                ),

                carrier=(
                    v2_model
                    .tag_carrier
                ),
            )
        )


    def forward(
        self,
        questioned_image,
    ):

        source_logits = self.source(
            questioned_image
        )


        index_logits = self.index(
            questioned_image
        )


        auth32_logits = self.auth(
            questioned_image
        )


        auth8_logits = (
            auth32_logits[
                ...,
                :8
            ]
        )


        logits20 = torch.cat(
            [
                index_logits,
                source_logits,
                auth8_logits,
            ],
            dim=-1,
        )


        return {

            "index_logits":
                index_logits,

            "source_logits":
                source_logits,

            "auth32_logits":
                auth32_logits,

            "auth8_logits":
                auth8_logits,

            # Compatibility alias used by later code.
            "auth_logits":
                auth8_logits,

            "logits20":
                logits20,
        }
