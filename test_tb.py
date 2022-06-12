from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    # create a folder name logs to store historical info
    # in terminal, tensorboard --logdir=logs --port =6007
    # create different log folder for different tasks
    writer = SummaryWriter("logs")

    # Example 1
    for i in range(100):
        writer.add_scalar("y=x", i, i)
        writer.add_scalar("y=2x", i, 2 * i)

    # Example 2
    import numpy as np
    from PIL import Image
    img = Image.open("data/p1.png")
    img = np.array(img)
    writer.add_image("test", img, 1, dataformats='HWC')
    # writer.add_scalar()
    writer.close()
