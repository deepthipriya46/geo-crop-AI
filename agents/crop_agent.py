from data.crop_database import CROP_DATABASE


class CropAgent:

    def get_crop_data(self, crop):

        crop = crop.lower()

        if crop not in CROP_DATABASE:
            raise ValueError("Crop not found in database")

        return CROP_DATABASE[crop]