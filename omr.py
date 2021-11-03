###################################
# CS B657 Spring 2021, Assignment #1
# 
# Optical Music Recognition
#
# Cody Harris
# Neelan Schueman
# Emma Cai

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import sys
import itertools
import time

# Pads image with mirror image
# 
# INPUT
# img - 2D np.array()
# Kernal size - int
# RETURN
# 2D np.array()
def mirror_pad(img, kern_size):
    pad_size = kern_size // 2
    img = np.array(img)
    row, col = img.shape
    pad_img = np.zeros((row+(2*pad_size), col+(2*pad_size)))
    #left mirror
    pad_img[pad_size:row+pad_size,0:pad_size] = np.flip(img[:,0:pad_size], axis = 1)
    #right mirror
    pad_img[pad_size:row+pad_size,col+pad_size:col+pad_size*2] = np.flip(img[:,-pad_size:], axis = 1)
    #insert image in middle
    pad_img[pad_size:row+pad_size,pad_size: col+pad_size] = img
    #top mirror
    pad_img[0:pad_size,:] = np.flip(pad_img[pad_size:pad_size*2,:], axis = 0)
    #bottom mirror
    pad_img[row+pad_size:row+pad_size*2,:] = np.flip(pad_img[row-1:row-1+pad_size], axis = 0)
    return pad_img

# Generic convolution
#
# INPUT
# img - 2D np.array()
# h - 2D np.array() must be square and have odd dimensions
# RETURN
# 2D np.array() with convolution 

def conv(img, h):
    if h.shape[0]%2 == 0:
        raise(ValueError("This function does not currently support even sized kernals"))
    if h.shape[0] != h.shape[1]:
        raise(ValueError("The kernal is not square"))
    kern_size = h.shape[0]
    orig_row, orig_col = img.shape
    h_flip = np.flip(np.flip(h, axis = 1), axis = 0)
    new_img = np.zeros((orig_row, orig_col))
    padded_img = mirror_pad(img, kern_size)
    for i in range(orig_row):
        for j in range(orig_col):
            img_slice = padded_img[i:i+kern_size,j:j+kern_size]
            new_img[i,j] = (h_flip * img_slice).sum()
    return new_img

def seperable_conv(img, hx, hy):
    '''
    img: filepath to an image that will be convolved with the two kernels
    hx: kernel in the x-direction
    hy: kernel in the y-direction
    returns the iamges convolved with each kernel
    '''
    k_len = len(hx)
    im_arr = np.array(Image.open(img))
    row_len = len(im_arr[0])
    col_len = len(im_arr)
    x_conv_arr = np.zeros((col_len,row_len))
    r = 0
    for row in im_arr:
        for c in range(0, row_len - (k_len - 1)):
            x_conv_arr[r,c] = sum(row[c:c+k_len] * hx)
        r+=1
    x_conv_arr = np.transpose(x_conv_arr)
    final_conv_arr = np.zeros((row_len, col_len))
    r = 0
    for row in x_conv_arr:
        for c in range(0, col_len - (k_len - 1)):
            final_conv_arr[r,c] = sum(row[c:c+k_len] * hy)
        r+=1
    final_conv_arr = final_conv_arr[0:(col_len - k_len+1),0:(row_len - k_len+1)]
    new_im = Image.new('L', (len(final_conv_arr), len(final_conv_arr[0])), color =0)
    for x in range(len(final_conv_arr[0])):
        for y in range(len(final_conv_arr)):
            val = int(final_conv_arr[x,y])
            new_im.putpixel((x,y), val)              
    new_im.show()
    return new_im


def hough_transform(img, num_lines=10):
    '''
    img: a binary image(2D) array where 1s or above indicate black lines and 0s indicate white background.
    num_line: please specify the number of lines needed based on max votes.
    return (k,c, v) which (k,c) are the top n lines in Cartesian coordinate and v is the votes.
    
    '''
    theta_range = np.deg2rad(np.arange(-90.0, 90.0))
    width,height = img.shape
    rho_range = np.arange(-(width+height),width+height)
    votes = np.zeros((len(rho_range), len(theta_range)))
    # indices of non zero elements of a binary image
    y_idxs, x_idxs = np.nonzero(img)
    for i in range(len(x_idxs)):
        for idx, theta in enumerate(theta_range):
            rho = int(x_idxs[i]*np.cos(theta)+y_idxs[i]*np.sin(theta)+width+height)
            votes[rho,idx] += 1
    # indexs are from an flattened array
    indexs = np.argpartition(votes, -num_lines, axis=None)[-num_lines:]
    rhos = rho_range[indexs//len(rho_range)]
    thetas = theta_range[indexs%len(theta_range)]
    a = np.cos(thetas)
    b = np.sin(thetas)
    k = -a/b
    c = rhos/b
    return k,c,votes

def hough_transform_v2(img, groups_num = 10):
    '''
    img: a music image with staves perfectly horizontal. 
    return
    positions: an array of y coordinates of the first line for every staff with treble staff and bass staff in turns
    spaces: an array of coresponding space between each group of five lines
    votes: the hough space
    '''
    gray_scale_img = img.convert('L')
    img_array = np.array(gray_scale_img)  
    binary_array = np.where(img_array>250, 0, 1)
    height,width = binary_array.shape
    # the row coordinate of the first line.
    pos_range = np.arange(0,height-20)
    # the space range between staves
    space_range = np.arange(0,60)
    votes = np.zeros((len(pos_range), len(space_range)))
    # dynamic programming is used to save time.
    line_votes = np.zeros(height)
    for i in range(height):
        num=binary_array[i,:].sum()
        if num>(width*0.4):
            line_votes[i]=num
    line_votes=[binary_array[i,:].sum() for i in range(height)]
    for pos in pos_range:
        for space in space_range:
            if pos+space*4 <height and space>4:
                for i in range(5):
                    votes[pos,space] += line_votes[pos+i*space]
    indexs = np.argpartition(votes, -groups_num, axis=None)[-groups_num:]
    positions = indexs//len(space_range)
    spaces = indexs%len(space_range)
    return positions,spaces,votes

def hough_transform_v3(img):
    '''
    hough transform with non maximum suppression, able to detected number of staves automaticlly
    img: a music image with staves largely horizontal and parallel
    return
    positions: an array of y coordinates of the first line for every staff with treble staff and bass staff in turns
    spaces: an array of corresponding space between each group of five lines
    '''
    gray_scale_img = img.convert('L')
    img_array = np.array(gray_scale_img)  
    binary_array = np.where(img_array>250, 0, 1)
    height,width = binary_array.shape
    # the row coordinate of the first line.
    pos_range = np.arange(0,height-20)
    # the space range between staves
    space_range = np.arange(0,60)
    # dynamic programming is used to save time.
    line_votes = np.zeros(height)
    for i in range(height):
        num=binary_array[i,:].sum()
        if num>(width*0.3):
            line_votes[i]=num
    positions = []
    spaces = []
    threshold = width
    # threshold > width*0.3 means 
    # the average length of the selected five staves should be longer than one third of picture width
    while threshold > width*0.3:
        votes = np.zeros((len(pos_range), len(space_range)))
        for pos in pos_range:
            for space in space_range:
                if pos+space*4 <height and space>4:
                    for i in range(5):
                        votes[pos,space] += line_votes[pos+i*space]
        max_index = np.argmax(votes)
        position = max_index//len(space_range)
        space = max_index%len(space_range)
        positions.append(position)
        spaces.append(space)
        threshold = votes[position, space]/5
        # suppress neighbours
        if position-2*space >=0 and position+6*space< height:
            line_votes[position-2*space:position+6*space] = 0
        else:
            line_votes[position:position+4*space] = 0
    return positions[:-1],spaces[:-1]
        
            
    
def find_space(spaces):
    '''
    find the most frequent integer for a list of spaces between the lines.
    spaces: an array/list of numbers
    returns b: the most frequent number
    '''
    b = max(set(spaces), key=list(spaces).count)
    return b
    

def plot_lines_with_image(img, positions, spaces):
    '''
    positions: an array of the row coordinate for the first line
    spaces: an array for coresponding space between lines.
    '''
    y = []
    for idx,pos in enumerate(positions):
        y.extend([pos+spaces[idx]*i for i in range(5)])
    plt.figure(figsize = (img.size[0]/320,img.size[1]/320))
    plt.hlines(positions, 0, img.width, color='b')
    plt.imshow(img)
    plt.show()
    return

def binary_convert(img_arr, thresh, how = 0):
    if how == 0:
        img_arr[img_arr < thresh] = -1
        img_arr[img_arr >= thresh] = 1
    elif how == 1:
        img_arr[img_arr < thresh] = 1
        img_arr[img_arr >= thresh] = 0
    return img_arr

def find_symbols(img, temp, black_thresh, prob_thresh, line_list, line_h, is_note):
    img = np.array(img.convert('L')).astype(np.float64)
    temp = temp.astype(np.float64)
    img_copy = img.copy()
    temp_copy = temp.copy()
    temp_row, temp_col = temp.shape
    music_row, music_col = img.shape
    img_bin = binary_convert(img, black_thresh)
    temp_bin = binary_convert(temp, black_thresh)
    top_score = temp_row*temp_col
    #Find black percentage
    img_bin_blk = binary_convert(img_copy, black_thresh, 1)
    temp_bin_blk = binary_convert(temp_copy, black_thresh, 1)
    blk_pct = temp_bin_blk.sum()/(temp_row*temp_col)
    above_thresh = []
    skip_list = []
    #Find best last row
    last_row = music_row - temp_row
    if line_list[-1]+line_h*8 < last_row:
        last_row = line_list[-1]+line_h*8
    #Find best first row
    first_row = 0
    if line_list[0]-int(round(line_h*3.5)) > first_row:
        first_row = line_list[0]-int(round(line_h*3.5))
    #Build row ignore list
    good_rows = []
    for i in line_list:
        if is_note:
            good_rows = good_rows + list(range(first_row,last_row))
        else:
            good_rows = good_rows + list(range(i,last_row))
    for i in range(first_row, last_row):
        for j in range(music_col - temp_col):
            if (i,j) in skip_list:
                skip_list.remove((i,j))
                continue
            else:
                music_rect = img_bin[i:i+temp_row, j:j+temp_col]
                rect_blk_pct = img_bin_blk[i:i+temp_row, j:j+temp_col].sum()/(temp_row*temp_col)
                score = (((temp_bin*music_rect).sum() / top_score)+1)/2
                if score>prob_thresh and rect_blk_pct <= blk_pct + 0.7 and rect_blk_pct >= blk_pct - 0.7:
                    i_val = list(range(i,i + int(temp_row /1.25)))
                    j_val = list(range(j - int(temp_col/1.25), j + int(temp_col / 1.25)))
                    skip_list = skip_list + list(itertools.product(i_val, j_val))
                    if is_note:
                        pitch = get_pitch(line_list, line_h, i)
                        above_thresh.append(((i,j), temp_col, temp_row, round(score, 5), pitch))
                    else:
                        above_thresh.append(((i,j), temp_col, temp_row, round(score, 5)))
    return above_thresh

def resample(img, new_height):
    '''
    img = an array of grayscale values
    new_height = the new height that you want for the resulting image (array of grayscale values)
    '''
    old_height, old_width = img.shape
    new_width = int((old_width/old_height) * new_height)
    h_ratio = old_height / new_height
    w_ratio = old_width / new_width
    new_height = int(new_height)
    new_img = np.zeros((int(new_height), new_width))
    for r in range(new_height):
        for c in range(new_width):
            p = np.array((min(r * h_ratio,old_height-2),min(c * w_ratio,old_width-2)))
            c1 = np.array((int(p[0]), int(p[1])))
            v1 = img[c1[0], c1[1]]
            v2 = img[c1[0], c1[1]+1]
            v3 = img[c1[0]+1, c1[1]]
            v4 = img[c1[0]+1, c1[1]+1]
            p_val = bilinear(p, c1, v1, v2, v3, v4)
            new_img[r,c] = p_val
    '''
    #This is just to visually see if it is working  
    pic = Image.new('L', (new_width, new_height), color =0)
    for x in range(new_width):
        for y in range(new_height):
            val = int(new_img[y,x])
            pic.putpixel((x,y), val)
    pic.show()
    '''
    return new_img

def bilinear(p, c1, v1, v2, v3, v4):
    a, b = p - c1
    return (1-b)*(1-a)*v1 + (1-b)*a*v3 + b*(1-a)*v2 + b*a*v4

def img2gray2array(path):
    '''
    path: the image path
    returns a gray scaled image array
    '''
    img = Image.open(path)
    gray_scale_img = img.convert('L')
    img_array = np.array(gray_scale_img) 
    return img_array

def new_height(height, b):
    '''
    height: the original template height 
    b: the space between the lines.
    '''
    return height*b/11

def draw_notes(img, data):
    img_draw = ImageDraw.Draw(img)
    for key in data.keys():
        if key == 'offilled_note':
            for it in data[key]:
                img_draw.rectangle([(it[0][1],it[0][0]), (it[0][1]+it[1],it[0][0]+it[2])],\
                                   fill = None, outline = 'red', width = 2)
                img_draw.text((it[0][1] - int(it[1]/1.5),it[0][0]), it[4], fill = 'red')
        elif key == 'quarter_rest':
            for it in data[key]:
                img_draw.rectangle([(it[0][1],it[0][0]), (it[0][1]+it[1],it[0][0]+it[2])],\
                                   fill = None, outline = '#01FF00', width = 2)
        else:
            for it in data[key]:
                img_draw.rectangle([(it[0][1],it[0][0]), (it[0][1]+it[1],it[0][0]+it[2]*2.39875)],\
                                   fill = None, outline = '#004DFF', width = 2)
    return img

def write_to_text(data, output):
    f = open(output, 'w')
    for key in data.keys():
        if key == 'offilled_note':
            for it in data[key]:
                f.write(str(it[0][0]) + ' ' + str(it[0][1]) + ' ' + str(it[2]) + ' ' + str(it[1]) +\
                        ' ' + key + ' ' + it[4] + ' ' + str(it[3]) + '\n')
        else:
            for it in data[key]:
                f.write(str(it[0][0]) + ' ' + str(it[0][1]) + ' ' + str(it[2]) + ' ' + str(it[1]) +\
                        ' ' + key + ' _ ' + str(it[3]) + '\n')

#assumes y_b and y_t are the top line of the staff
def get_pitch(line_list, lh, y_note):
    staff_num = 'NONE'
    for i, line in enumerate(line_list):
        if y_note <= line+lh*8 and y_note >= line-lh*3.5:
            staff_num = i+1
            staff_line = line
    #If note is not within the range of a staff number
    #We are returning 'A'
    if staff_num == 'NONE':
        return 'A'
    if staff_num%2 == 0:
        clef = 'bass'
    else:
        clef = 'treble'
    treb_notes = ['E', 'D', 'C', 'B', 'A', 'G', 'F', 'E', 
                  'D', 'C', 'B', 'A', 'G', 'F', 'E', 
                  'D', 'C', 'B', 'A', 'G', 'F', 'E', 'D']
    bass_notes = ['G', 'F', 'E', 'D', 'C', 'B', 'A', 'G', 
                  'F', 'E', 'D', 'C', 'B', 'A', 'G', 
                  'F', 'E', 'D', 'C', 'B', 'A', 'G', 'F']
    h = lh / 2
    top = staff_line - lh*3.5
    diff = int(round((y_note - top)/h))
    if clef == 'treble':
        pitch = treb_notes[diff]
    else:
        pitch = bass_notes[diff]
    return pitch

def main():
    #t0 = time.time()
    BLACK = 150
    THRESH = 0.85
    music_img = Image.open(sys.argv[1])
    music_img_rgb = music_img.convert('RGB')
    ys_spaces_vote = hough_transform_v3(music_img) 
    # ys is the y coordinates of all the first lines of the staff.
    # ys is used to define the pitch later
    ys = ys_spaces_vote[0] 
    # after sorting, the even numbers indicate the treble while the odd numbers indicate the bass.
    ys = np.sort(ys)
    bs = ys_spaces_vote[1]
    # b is the space between lines
    b = find_space(bs)
    offilled_note_img = img2gray2array('templates/template1.png')
    quarter_rest_img = img2gray2array('templates/template2.png')
    eight_rest_img = img2gray2array('templates/template3.png')
    if b != 12:
        new_offilled_note = np.array(resample(offilled_note_img, b))
        new_quarter_rest = np.array(resample(quarter_rest_img, new_height(quarter_rest_img.shape[0], b)))
        new_eight_rest = np.array(resample(eight_rest_img, new_height(eight_rest_img.shape[0], b)))
    else:
        new_offilled_note = offilled_note_img
        new_quarter_rest = quarter_rest_img
        new_eight_rest = eight_rest_img
    #Show transformed image
    #new_image = Image.fromarray(np.uint8(new_eight_rest), 'L')
    #new_image.show()
    detected = {'offilled_note': [], 'quarter_rest': [], 'eighth_rest': []}
    templates = {'offilled_note': new_offilled_note, 
                 'quarter_rest': new_quarter_rest,
                 'eighth_rest': new_eight_rest}

    #ys override
    for key in templates.keys():
        note = False
        if key == 'offilled_note':
            note = True 
        detected[key] = find_symbols(music_img,templates[key], BLACK, THRESH, ys, b, note)
    
    drawn = draw_notes(music_img_rgb, detected)
    write_to_text(detected, 'detected.txt')
    #print(time.time()-t0)
    #drawn.show()
    drawn.save('detected.png')
    #print(detected)
    

main()