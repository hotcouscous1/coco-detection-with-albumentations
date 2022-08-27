# COCO-Detection Dataset with Albumentations

<p align="center">
  <img src="https://github.com/hotcouscous1/Logo/blob/main/TensorBricks_Logo.png" width="500" height="120">
</p>

Albumentations is an image augmentation library using OpenCV + Numpy, which is faster than PIL and torchvision.transforms.

About Albumentations;
- docs: https://albumentations.ai/docs/
- github: https://github.com/albumentations-team/albumentations/

This implementation suggests Bbox_Augmentor class to compose augmentation much easier, and COCO_Detection class which loads images by cv2, rearranges the categories to 80, and can be passed to torch DataLoader same as CocoDetection of torchvision.  

### Example
```
import albumentations as A

T = Bbox_Augmentor(1, 'coco', min_area=512, min_visibility=0.2)

T.append(A.RandomResizedCrop(512, 512, (0.2, 1.0), p=0.5))
T.append(A.RandomScale((-0.5, 0), p=0.5))
T.append(A.HorizontalFlip(p=0.5))
T.append(A.PadIfNeeded(512, 512, border_mode=cv2.BORDER_CONSTANT, value=imagenet_fill(), p=0.5))
T.append(A.Resize(512, 512, p=1))

T.make_compose()

dataset = COCO_Detection(root='...',annFile='...', bbox_augmentor=T)
train_loader = DataLoader(dataset, batch_size=4, num_workers=4, collate_fn=make_mini_batch)
```

## License
BSD 3-Clause License Copyright (c) 2022, hotcouscous1
