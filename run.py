import numpy as np
import cv2 
import tracker
import os
import argparse
import sys
import time
import util
import matplotlib.pyplot as plt
def main(args):
    # Load  arg
    dataset = args.dataset_descriptor
    save_directory = args.save_directory
    img_channel = args.multi_channel
    HOG_flag = args.HOG_feature
    scale_factor = args.scale_factor
    save_flag = args.save_img
    resize = args.resize
    show_result = args.show_result
    padding = 2
    
   

    # Load dataset information and get start position
    title = dataset.split('/')
    title = [ t for t in title if t][-1]
    img_lst = util.load_imglst(dataset)
    #bbox_lst = util.load_bbox(os.path.join(dataset+'/groundtruth.txt'),resize)
    init_img = cv2.imread(img_lst[0])
    #init_img = init_img.astype(np.float32)
    bbox_lst = cv2.selectROI('window',init_img,False,False)
    py,px,h,w = bbox_lst
    pos = (px,py,w,h)
    o_w,o_h = w,h
    frames = len(img_lst)


    # Get image information and init parameter
    output_sigma_factor = 1 / float(16)
    img = cv2.imread(img_lst[0],img_channel)
    if resize:
        img_size = np.int(img.shape[0]/2),np.int(img.shape[1]/2)
    else:
        img_size = img.shape[:2]
    if HOG_flag:
        target_size = 64,64
        l = 0.0001
        sigma = 0.6
        inter_factor = 0.012
        scale_weight = 0.95
    else:
        target_size = np.int(padding/2*w)*2,np.int(padding/2*h)*2
        l = 0.0001
        sigma = 0.2
        inter_factor = 0.02
        scale_weight = 0.95
    f = inter_factor

    # Generate y label
    output_sigma = np.sqrt(np.prod(target_size)) * output_sigma_factor
    cos_window = np.outer(np.hanning(target_size[0]),np.hanning(target_size[1]))
    y = tracker.generate_gaussian(target_size,output_sigma)#高斯图响应在四个角
    # y1 = np.uint8(y*255)
    # cv2.imshow('1',y1)
    # cv2.waitKey(0)
    rez_shape = y.shape

    # Create file to save result
    tracker_bb =[]
    result_file = os.path.join(save_directory,title+'_'+'result.txt')
    file = open(result_file,'w')
    start_time = time.time()

    # Tracking
    for i in range(frames):
        img = cv2.imread(img_lst[i],img_channel)#默认读RGB格式时参数为1
        if resize:
           img = cv2.resize(img,img_size[::-1])
        if i==0:
            x =  tracker.get_window(img, pos, padding, scale_factor, rez_shape)
            x = tracker.getFeature(x,cos_window,HOG_flag)#经过归一化及余弦窗处理后的图像数据
            alpha = tracker.train(x,y,sigma,l)
            z = x
            best_scale = 1
        else:
            x = tracker.get_window(img, pos, padding, scale_factor,rez_shape)
            x = tracker.getFeature(x,cos_window,HOG_flag)
            response = tracker.detect(alpha,x,z,sigma)
            best_scale = 1
            peak_res = response.max()
            # if scale_factor!=1:
            #     Allscale = [1.0/scale_factor,scale_factor,0.8]
            if scale_factor ==1:
                Allscale = [0.95,0.985,0.995,1,1.005,1.015,1.025]
                for scale in Allscale:
                    x = tracker.get_window(img, pos, padding, scale,rez_shape)
                    x = tracker.getFeature(x,cos_window,HOG_flag)
                    res = tracker.detect(alpha,x,z,sigma)
                    if res.max() > peak_res:
                        peak_res = res.max()
                        best_scale = scale
                        response = res

            # response_img = np.uint8(response*255)
            # cv2.imshow('gaoshi', response_img)
            # cv2.waitKey(1)
            # plt.imshow(response)
            # plt.pause(0.1)
            # Update position x z alpha
            new_pos = tracker.update_tracker(response,img_size,pos,HOG_flag,best_scale)
            x = tracker.get_window(img, new_pos, padding, 1, rez_shape)
            x = tracker.getFeature(x,cos_window,HOG_flag)
            new_alpha = tracker.train(x,y,sigma,l)
            alpha = f*new_alpha+(1-f)*alpha
            new_z = x
            z = (1-f)*z+f*new_z
            pos = new_pos

        # Write the position
        if resize:
            out_pos = [int(pos[1]*2),int(pos[0]*2),int(pos[3]*2),int(pos[2]*2)]
        else:
            out_pos = [pos[1],pos[0],pos[3],pos[2]]
        win_string = [ str(int(p)) for p in out_pos]
        win_string = ",".join(win_string)
        tracker_bb.append(win_string)
        file.write(win_string+'\n')
        #visual(img,out_pos)


    duration = time.time()-start_time
    fps = int(frames/duration)
    print ('each frame costs %3f second, fps is %d'%(duration/frames,fps))
    file.close()
    
    result = util.load_bbox(result_file,0)
    if show_result:
        util.display_tracker(img_lst,result,save_flag)
def visual(img,bbox):
    (x,y,w,h) = bbox
    x=int(x)
    y=int(y)
    w=int(w)
    h=int(h)
    pt1,pt2 = (x,y),(x+w,y+h)
    img_rec = cv2.rectangle(img,pt1,pt2,(255,0,0),2)
    cv2.imshow('window',img_rec)
    cv2.waitKey(1)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_descriptor', type=str,
        help='The directory of video and groundturth file',default='surfer')
    parser.add_argument('--save_directory', type=str,
        help='The directory of result file',default='save')
    parser.add_argument('--show_result', type=int, 
        help='Show result or not',default=1)
    parser.add_argument('--resize', type=float, 
        help='Resize img or not',default=0)
    parser.add_argument('--multi_channel', type=int, 
        help='Use multi channel image or not',default=1)
    parser.add_argument('--HOG_feature', type=int, 
        help='Use HOG or not',default=1)
    parser.add_argument('--scale_factor', type=float, 
        help='bbox scale factor',default=1)
    parser.add_argument('--save_img', type=int, 
        help='save img or not',default=0)

    return parser.parse_args()
    

if __name__ == "__main__":
    # main(parse_arguments(sys.argv[1:]))
    main(parse_arguments())

