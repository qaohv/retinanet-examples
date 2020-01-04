from typing import List, Dict
import albumentations as A

DEFAULT_COCO_BBOXES = {
    "format": "coco",
    "min_area": 0,
    "min_visibility": 0,
    "label_fields": ['category_id']
}


def create_augmentations(transforms_config: List[Dict], bbox_params: dict = DEFAULT_COCO_BBOXES):
    if len(transforms_config) == 0:
        return None

    transforms = []
    for transform_cfg in transforms_config:
        try:
            transform_class = getattr(A, transform_cfg['name'])
            del transform_cfg["name"]
            print(transform_cfg)
            transforms.append(transform_class(**transform_cfg))
        except AttributeError as ex:
            print(f"Wrong augmentation name, list of supported augmentations here: "
                  f"https://albumentations.readthedocs.io/en/latest/api/augmentations.html")
            exit(1)

    return A.Compose(transforms, A.BboxParams(**bbox_params))
