from toolkit.extension import Extension


class SDTrainerExtension(Extension):
    uid = "sd_trainer"
    name = "SD Trainer"

    @classmethod
    def get_process(cls):
        from .SDTrainer import SDTrainer
        return SDTrainer


class TextualInversionTrainer(SDTrainerExtension):
    uid = "textual_inversion_trainer"


AI_TOOLKIT_EXTENSIONS = [
    SDTrainerExtension,
    TextualInversionTrainer,
]
