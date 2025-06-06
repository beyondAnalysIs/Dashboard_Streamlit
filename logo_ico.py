from PIL import Image

# ruta de imagen
img = Image.open('Logo.png')

# dimension de la imagen
img = img.resize((32,32))

# guardar la imagen en formato ICO
img.save('logo.ico', format='ICO')

print("Imagen convertida a ICO")