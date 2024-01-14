#import colorcorectionBW as ccSW
#import src.colorcorrectionCOLOR as ccC
import src.image_processing as imPr
import src.namespace as names

import cv2

proc = imPr.ImageProcessing()
ns = names.Names()

def main():
    ### Load Image ###
    # for img in Dataset
    img = cv2.imread('Images/35mmSW.jpg')

    boundaryType = ns.name_small_format
    negativeType = ns.name_negative_bw

    # Show loaded image
    proc.showImg(window_name='loaded Image', img=img)

    ### Cut Images ###
    strips = proc.cutStrip(img, boundaryType=boundaryType, visualizeSteps=True)

    for strip in strips:
        # proc.showImg(window_name='strip', img=strip)
        height, width = img.shape[:2]
        if height > width:
            print('[INFO] Rotation have to be done')
            strip = cv2.rotate(src=strip, rotateCode=cv2.ROTATE_90_CLOCKWISE)

        single_images, strip = proc.cutSingleImgs(strip, visualizeSteps=False, boundaryType=boundaryType)
        print(f'[INFO] {len(single_images)} Single Images')

        if boundaryType != ns.name_dia:
            finished_imgs = []
            for img in single_images:

                ### Bilder invertieren ###
                if strip is not None:
                    invertedImage = proc.invertImg(negative_img=img, offset_img=strip, negative_type=negativeType, visualizeSteps=False)
                    finished_imgs.append(invertedImage)
                    #proc.showImg(window_name='invertedImage', img=invertedImage)

    ### Bilder Speichern ###

if __name__ == '__main__':
    main()