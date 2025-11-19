# calculating mean and std for Normalize()
# from https://saturncloud.io/blog/how-to-normalize-image-dataset-using-pytorch/

def get_mean_std(loader):
    mean = 0.
    std = 0.
    n_samples = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)  # flatten H*W
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        n_samples += batch_samples

    mean /= n_samples
    std /= n_samples
    return mean.tolist(), std.tolist()
