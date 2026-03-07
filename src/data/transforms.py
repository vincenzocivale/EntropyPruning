import torchvision.transforms as T


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transform(img_size=224):
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomApply([T.RandomRotation((90, 90))], p=0.5),
        T.RandomApply([T.ColorJitter(0.2, 0.2, 0.1, 0.05)], p=0.5),
        T.Resize((img_size, img_size)),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_eval_transform(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
