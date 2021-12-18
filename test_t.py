from PIL import Image

img1 = Image.open('p.jpg')
img1 = img1.resize((1000,900))

img2 = Image.open('test1.jpg')

img1.paste(img2)
img1.save('test2.jpg')
