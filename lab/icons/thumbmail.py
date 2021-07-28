from PIL import Image
import glob


def make_square(im, min_size=64, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

items = glob.glob("*.jpg")
for it in items:
    im = Image.open(it)
    im=make_square(im)
    im.save(it)
