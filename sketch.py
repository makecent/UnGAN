from mmgen.datasets.pipelines import Resize, LoadImageFromFile
results = {'real_img_path': 'captcha_data/00001.jpg'}
t = LoadImageFromFile(key='real_img', io_backend='disk')(results)
tt = Resize(keys=['real_img'], scale=(-1, 19))(t)
print(tt['real_img'].shape)
print('haha')