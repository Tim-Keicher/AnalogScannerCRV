#import colorcorectionBW as ccSW
#import src.colorcorrectionCOLOR as ccC
import src.image_processing as imPr

import cv2

proc = imPr.ImageProcessing()


def main():
    ### Load Image ###
    # for img in Dataset
    img = cv2.imread('Images/35mmSW.jpg')

    # boundaryType = 'DIAS', '35',  or '120'
    # boundaryType = '120'
    boundaryType = '35'
    negativeType = proc.type_bw
    # Show loaded image
    proc.showImg(window_name='loaded Image', img=img)

    ### Cut Images ###
    strips = proc.cutStrip(img, boundaryType=boundaryType, visualizeSteps=True)
    for strip in strips:
        #proc.showImg(window_name='strip', img=strip)

        single_images, strip = proc.cutSingleImgs(strip, visualizeSteps=False, boundaryType=boundaryType)
        print(f'[INFO] {len(single_images)} Single Images')
        for img in single_images:

            ### Bilder invertieren ###
            if strip is not None:
                invertedImage = proc.invertImg(negative_img=img, offset_img=strip, negative_type=negativeType, visualizeSteps=True)
                #proc.showImg(window_name='invertedImage', img=invertedImage)

    ### Bilder Speichern ###

if __name__ == '__main__':
    main()