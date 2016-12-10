from neon.backends import gen_backend
from neon.data import ArrayIterator

from PIL import Image, ImageDraw

im = Image.open('1.jpg')
draw = ImageDraw.Draw(im)
draw.rectangle(((100,100),(120.4111,200)),outline=(255,0,0))

im = Image.open('train/'+bonnets[0]['filename'])    
im = im.resize((256,256))

# generate the HDF5 file
datsets = {'train': (X_train, y_train),
           'test': (X_test, y_test)}

for ky in ['train', 'test']:
    df = h5py.File('mnist_%s.h5' % ky, 'w')

    # input images
    in_dat = datsets[ky][0]
    df.create_dataset('input', data=in_dat)
    df['input'].attrs['lshape'] = (1, 28, 28)  # (C, H, W)

    # can also add in a mean image or channel by channel mean for color image
    # for mean subtraction during data iteration
    # e.g.
    if ky == 'train':
        mean_image = np.mean(X_train, axis=0)
    # use training set mean for both train and val data sets
    df.create_dataset('mean', data=mean_image)

    target = datsets[ky][1].reshape((-1, 1))  # make it a 2D array
    df.create_dataset('output', data=target)
    df['output'].attrs['nclass'] = 10
df.close()