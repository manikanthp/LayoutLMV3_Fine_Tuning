import fitz
import os

dpi = 300
zoom = dpi/72
magnify = fitz.Matrix(zoom, zoom)

path = "./TC.pdf"
count = 0

doc = fitz.open(path)

for page in doc:
    count+=1
    pix = page.get_pixmap(matrix=magnify)
    pix.save(f"./image/TC_page_{count}.png")