
import os
import sys
import Image
import math
import matplotlib.image as mpimg
import numpy as np

def sym_hor_patch(patch):
    """ Take as input a patch and returns its symmetric with respect to the horizontal direction"""
    
    #Extract number of pixel in the patch
    nx = patch.shape[0]
    ny = patch.shape[1]
    
    new_patch = np.zeros([nx,ny,3])
    
    # For each pixel take its symmetric
    
    for i in range(nx):
        for j in range(ny):
            new_patch[i][j] = patch[nx-1-i][j]
            
    return new_patch

def sym_ver_patch(patch):
    """ Take as input a patch and returns its symmetric with respect to the vertical direction"""
    
    #Extract number of pixel in the patch
    nx = patch.shape[0]
    ny = patch.shape[1]
    
    new_patch = np.zeros([nx,ny,3])
    
    # For each pixel take its symmetric
    
    for i in range(nx):
        for j in range(ny):
            new_patch[i][j] = patch[i][ny-1-j]
            
    return new_patch

def rotate_90_patch(patch):
    """ Take as input a patch and returns its rotated version by 90 degrees clockwise"""
    
    #Extract number of pixel in the patch
    nx = patch.shape[0]
    ny = patch.shape[1]
    
    new_patch = np.zeros([nx,ny,3])
    
    #Rotating the matrix by 90 degrees is equivalent to taking its transpose and then symmetrize  
    for i in range(nx):
        for j in range(ny):
            new_patch[i][j] = patch[j][i]
            
    new_patch = sym_ver_patch(new_patch)
    
    return new_patch

def rotate_180_patch(patch):
    """ Take as input a patch and returns its rotated version by 180 degrees clockwise"""
    
    new_patch = rotate_90_patch(patch)
    new_patch = rotate_90_patch(new_patch)
    
    return new_patch

def rotate_270_patch(patch):
    """ Take as input a patch and returns its rotated version by 270 degrees clockwise"""
    
    new_patch = rotate_180_patch(patch)
    new_patch = rotate_90_patch(new_patch)
    
    return new_patch

        
def add_new_patches(img_patches):
    """Generate new patches by rotating and symmetrizing each patch.
       Each patch generates 7 new patches"""
    
    N = img_patches.shape[0]
    
    for i in range(N):
        patch = img_patches[i]
        
        print('Evaluating image {0} out of {1}'.format(i+1,N))
        
        #Geometrically transformations of each patch
        patch1 = rotate_90_patch(patch)
        patch2 = rotate_180_patch(patch)
        patch3 = rotate_270_patch(patch)
        patch4 = sym_hor_patch(patch)
        patch5 = sym_ver_patch(patch)
        patch6 = sym_hor_patch(patch1)
        patch7 = sym_ver_patch(patch1)
        
        #Reshaping in order to match dimensions for concatenate
        patch1 = patch1[np.newaxis,:,:,:]
        patch2 = patch2[np.newaxis,:,:,:]
        patch3 = patch3[np.newaxis,:,:,:]
        patch4 = patch4[np.newaxis,:,:,:]
        patch5 = patch5[np.newaxis,:,:,:]
        patch6 = patch6[np.newaxis,:,:,:]
        patch7 = patch7[np.newaxis,:,:,:]
        
        #Adding new patches
        img_patches = np.concatenate((img_patches,patch1),axis=0)
        img_patches = np.concatenate((img_patches,patch2),axis=0)
        img_patches = np.concatenate((img_patches,patch3),axis=0)
        img_patches = np.concatenate((img_patches,patch4),axis=0)
        img_patches = np.concatenate((img_patches,patch5),axis=0)
        img_patches = np.concatenate((img_patches,patch6),axis=0)
        img_patches = np.concatenate((img_patches,patch7),axis=0)
        
    return img_patches
        
def add_gt_patches(gt_patches):
    """Generate new patches for grondtruth to match the added patches in 'add_new_patches'. """
    
    N = gt_patches.shape[0]
    
    for i in range(N):
        patch = gt_patches[i]
        
        print('Evaluating image {0} out of {1}'.format(i+1,N))
        
                
        #Reshaping in order to match dimensions for concatenate
        patch = patch[np.newaxis,:,:,:]
                
        #Adding new patches
        for j in range(7):
            gt_patches = np.concatenate((gt_patches,patch),axis=0)
                
    return gt_patches
    
    