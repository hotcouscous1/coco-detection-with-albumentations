from utils.utils import *
from bbox_augmentor import Bbox_Augmentor
from pycocotools.coco import COCO
from torchvision.datasets.vision import VisionDataset



class COCO_Detection(VisionDataset):

    __doc__ = r"""
        The main difference from torchvision.datasets.CocoDetection is to load an image 
        by cv2 instead of PIL, which is faster. 
        
        COCO annotations have 91 categories, but the detection task includes only 80 classes.
        For predictions to be matched to labels precisely, 11 missing categories should be excluded.
        
        Output:
            image: 3D tensor in RGB and CHW format
            label: concatenated tensor of (x_min, y_min, width, height) bbox and one-hot vector of 80 categories
    """

    num_classes = 80
    coco_cat = (i for i in range(1, 91))
    missing_cat = (12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91)

    def __init__(self,
                 root: str,
                 annFile: str,
                 bbox_augmentor: Optional[Bbox_Augmentor]):

        super().__init__(root)

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.augmentor = bbox_augmentor
        if self.augmentor:
            assert self.augmentor.format == 'coco', \
                'the bounding box format must be coco, (x_min, y_min, width, height)'
            assert self.augmentor.ToTensor is True, \
                'the image should be returned as a tensor'

        self.cat_table = category_filter(self.coco_cat, self.missing_cat)


    def __len__(self):
        return len(self.ids)


    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        coco = self.coco
        img_id = self.ids[index]

        img_path = coco.loadImgs(img_id)[0]['file_name']
        image = cv2.imread(os.path.join(self.root, img_path))
        target = coco.loadAnns(coco.getAnnIds(imgIds=img_id))

        bboxes, category_ids = [], []

        for i, t in enumerate(target):
            bbox = t['bbox']
            if 0.0 in bbox[2:]:
                bbox[2] += 1e-5
                bbox[3] += 1e-5

            bboxes.append(bbox)
            category_ids.append(t['category_id'])


        if self.augmentor:
            transform = self.augmentor(image, bboxes, category_ids)
            image, bboxes, category_ids = transform.values()
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image)

        image = image.to(device=device)


        for i, cat_id in enumerate(category_ids):
            new_id = self.cat_table[cat_id]
            category_ids[i] = make_one_hot(self.num_classes, new_id)

        if category_ids:
            category_ids = torch.stack(category_ids)
        else:
            category_ids = torch.tensor([], dtype=torch.int8, device=device)

        bboxes = torch.from_numpy(np.asarray(bboxes)).to(device=device)
        label = torch.cat((bboxes, category_ids), dim=1)

        if len(label) == 0:
            label = torch.zeros((0, 84), dtype=torch.int8, device=device)

        return image, label

