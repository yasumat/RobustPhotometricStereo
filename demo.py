import time
from rps import RPS

light_filename = "./data/lights.txt"
mask_filename = "./data/caesar_mask.png"
img_foldername = "./data/caesar_specular/"
img_extension = "png"

rps = RPS()
rps.load_mask(filename=mask_filename)
rps.load_lighttxt(filename=light_filename)
rps.load_images(foldername=img_foldername, ext=img_extension)
start = time.time()
rps.solve(RPS.L2_SOLVER)
elapsed_time = time.time() - start
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
rps.disp_normalarray()
rps.save_normalarray()