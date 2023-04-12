import copy
import os
import shutil

import cv2
import gradio as gr
import numpy as np
import modules.scripts as scripts

from modules import images, processing
from modules.processing import process_images, Processed
from modules.shared import opts
from PIL import Image, ImageFilter, ImageColor, ImageOps
from pathlib import Path
from typing import List, Tuple, Iterable


#Returns a list of images located in the input path. For ControlNet iamges
def get_all_frames_from_path(path):
    if not os.path.isdir(path):
        return None
    frame_list = []
    for filename in sorted(os.listdir(path)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                frame_list.append(img)
    frame_list.insert(0, frame_list[0])            
    return frame_list  
    

#Returns a list of images located in the input path. For Color iamges
def get_images_from_path(path):
    if not os.path.isdir(path):
        return None
    images = []
    for filename in os.listdir(path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(path, filename)
            img = Image.open(img_path)
            images.append(img)
    images.append(images[-1])
    images.insert(0, images[0])
    return images

#Returns the number of the smallest number in the entire image sequence list. For ControlNet
def get_min_frame_num(video_list):
    min_frame_num = -1
    for video in video_list:
        if video is None:
            continue
        else:
            frame_num = len(video)
            print(frame_num)
            if min_frame_num < 0:
                min_frame_num = frame_num
            elif frame_num < min_frame_num:
                min_frame_num = frame_num
    return min_frame_num


#Blende method


def basic(target, blend, opacity):
    return target * opacity + blend * (1-opacity)

def blender(func):
    def blend(target, blend, opacity=1, *args):
        res = func(target, blend, *args)
        res = basic(res, blend, opacity)
        return np.clip(res, 0, 1)
    return blend


class Blend:
    @classmethod
    def method(cls, name):
        return getattr(cls, name)
    
    normal = basic
    
    @staticmethod
    @blender
    def darken(target, blend, *args):
        return np.minimum(target, blend)
    
    @staticmethod
    @blender
    def multiply(target, blend, *args):
        return target * blend
    
    @staticmethod
    @blender
    def color_burn(target, blend, *args):
        return 1 - (1-target)/blend
    
    @staticmethod
    @blender
    def linear_burn(target, blend, *args):
        return target+blend-1
    
    @staticmethod
    @blender
    def lighten(target, blend, *args):
        return np.maximum(target, blend)
    
    @staticmethod
    @blender
    def screen(target, blend, *args):
        return 1 - (1-target) * (1-blend)
    
    @staticmethod
    @blender
    def color_dodge(target, blend, *args):
        return target/(1-blend)
    
    @staticmethod
    @blender
    def linear_dodge(target, blend, *args):
        return target+blend
    
    @staticmethod
    @blender
    def overlay(target, blend, *args):
        return  (target>0.5) * (1-(2-2*target)*(1-blend)) +\
                (target<=0.5) * (2*target*blend)
    
    @staticmethod
    @blender
    def soft_light(target, blend, *args):
        return  (blend>0.5) * (1 - (1-target)*(1-(blend-0.5))) +\
                (blend<=0.5) * (target*(blend+0.5))
    
    @staticmethod
    @blender
    def hard_light(target, blend, *args):
        return  (blend>0.5) * (1 - (1-target)*(2-2*blend)) +\
                (blend<=0.5) * (2*target*blend)
    
    @staticmethod
    @blender
    def vivid_light(target, blend, *args):
        return  (blend>0.5) * (1 - (1-target)/(2*blend-1)) +\
                (blend<=0.5) * (target/(1-2*blend))
    
    @staticmethod
    @blender
    def linear_light(target, blend, *args):
        return  (blend>0.5) * (target + 2*(blend-0.5)) +\
                (blend<=0.5) * (target + 2*blend)
    
    @staticmethod
    @blender
    def pin_light(target, blend, *args):
        return  (blend>0.5) * np.maximum(target,2*(blend-0.5)) +\
                (blend<=0.5) * np.minimum(target,2*blend)
    
    @staticmethod
    @blender
    def difference(target, blend, *args):
        return np.abs(target - blend)
    
    @staticmethod
    @blender
    def exclusion(target, blend, *args):
        return 0.5 - 2*(target-0.5)*(blend-0.5)

blend_methods = [i for i in Blend.__dict__.keys() if i[0]!='_' and i!='method']



def blend_images(base_img, blend_img, blend_method, blend_opacity, do_invert):
    
    img_base = np.array(base_img.convert("RGB")).astype(np.float64)/255
    
    if do_invert:
        img_to_blend = ImageOps.invert(blend_img.convert('RGB'))       
    else:
        img_to_blend = blend_img
                
    img_to_blend = img_to_blend.resize((int(base_img.width), int(base_img.height)))
                
    img_to_blend = np.array(img_to_blend.convert("RGB")).astype(np.float64)/255
            
    img_blended = Blend.method(blend_method)(img_to_blend, img_base, blend_opacity)
                
    img_blended *= 255
                
    img_blended = Image.fromarray(img_blended.astype(np.uint8), mode='RGB')
    
    return img_blended


#Define UI and script properties.
class Script(scripts.Script):  
    
    def title(self):
        return "controlnet I2I sequence_toyxyz_v2"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):

        ctrls_group = ()
        max_models = opts.data.get("control_net_max_models_num", 1)
        
        input_list = []

        with gr.Group():
            with gr.Accordion("ControlNet-I2I-sequence-toyxyz", open = True): 
                with gr.Column():
                                    
                    feed_prev_frame = gr.Checkbox(value=False, label="Feed previous frame / Reduce flickering by feeding the previous frame image generated by Img2Img")
                    
                    use_init_img = gr.Checkbox(value=False, label="Blend color image / Blend the color image sequence with the initial Img2Img image or previous frame")
                    
                    use_TemporalNet = gr.Checkbox(value=False, label="Use TemporalNet / Using TemporalNet to reduce flicker between image sequences. Add TemporalNet in addition to the multi-controlnet you need. It should be placed at the end of the controlnet list.")

                    blendmode = gr.Dropdown(blend_methods, value='normal', label='Blend mode / Choose how to blend the color image with the Previous frame or Img2Img initial image')
                    
                    opacityvalue = gr.Slider(0, 1, value=0, label="Opacity / Previous frame or Img2Img initial image + (color image * opacity)", info="Choose betwen 0 and 1")
                    

                    for i in range(max_models):
                        input_path = gr.Textbox(label=f"ControlNet-{i}", placeholder="image sequence path")
                        input_list.append(input_path)
                    
                    tone_image_path = gr.Textbox(label=f"Color_Image / Color images to be used for Img2Img in sequence", placeholder="image sequence path")
                    
                    output_path = gr.Textbox(label=f"Output_path / Deletes the contents located in the path, and creates a new path if it does not exist", placeholder="Output path")

        ctrls_group += tuple(input_list) + (use_TemporalNet, use_init_img, opacityvalue, blendmode, feed_prev_frame, tone_image_path, output_path)

        return ctrls_group
        


    #Image Generate Definition
    def run(self, p, *args):

        path = p.outpath_samples
        
        output_path = args[-1]  # get the last argument, which is the output path
        
        feedprev = args[-3]
        
        blendm = args[-4]
        
        opacityval = args[-5]
        
        useinit = args[-6]

        usetempo = args[-7]
        
        
        # Check whether the output path exists, if it does, delete it and create a new one.
        if os.path.isdir(output_path):
            for file in os.scandir(output_path):
                os.remove(file.path)
        else :    
            os.mkdir(output_path)

        #Get the number of controlnet models.
        video_num = opts.data.get("control_net_max_models_num", 1)

        # Get the ControlNet image sequence list.
        image_list = [get_all_frames_from_path(image) for image in args[:video_num]]
        
        # Get a list of color image sequences.
        color_image_list = get_images_from_path(args[-2])
        
        # Get the first frame
        previmg = p.init_images

        tempoimg = p.init_images[0]
        
        #If img2img color correction is enabled in webui settings, color correction is performed based on the first frame.
        initial_color_corrections = [processing.setup_color_correction(p.init_images[0])]
        
        #Save initial img2img image
        initial_image = p.init_images[0]
        
        # Get the total number of frames.
        frame_num = get_min_frame_num(image_list)

        # image processing
        if frame_num > 0:
            output_image_list = []
            
            for frame in range(frame_num):
                copy_p = copy.copy(p)
                copy_p.control_net_input_image = []
                for video in image_list:
                    if video is None:
                        continue
                    copy_p.control_net_input_image.append(video[frame])

                    if usetempo == True :
                        copy_p.control_net_input_image.append(tempoimg)

                    
                    if color_image_list and feedprev == False:
                        
                        if frame<len(color_image_list):
                            tone_image = color_image_list[frame+1]
                        
                        if useinit:
                            tone_image = blend_images(initial_image, tone_image, blendm, opacityval, False)
                            
                        p.init_images = [tone_image.convert("RGB")]
                        
                proc = process_images(copy_p) 
                
                
                
                if feedprev == True and useinit == False:
                    if previmg is None:
                        continue
                    else:
                        previmg = proc.images[0]
                        
                        if frame == 0:
                            previmg = initial_image
                            
                        p.init_images = [previmg]
                        
                        if opts.img2img_color_correction:
                            p.color_corrections = initial_color_corrections
                            
                            
                if feedprev == True and color_image_list and useinit:
                    if previmg is None:
                        continue
                    else:
                        previmg = proc.images[0]
                        
                        if frame == 0:
                            previmg = initial_image
                            
                        previmg = blend_images(previmg, color_image_list[frame+1], blendm, opacityval, False)

                            
                        p.init_images = [previmg]

                        if opts.img2img_color_correction:
                            p.color_corrections = initial_color_corrections

                img = proc.images[0]

                if usetempo == True :
                    if frame > 0 :
                        tempoimg = proc.images[0]


                #Save image
                if(frame>0):
                    images.save_image(img, output_path, f"Frame_{frame}")
                copy_p.close()
                

        else:
            proc = process_images(p)
        
        return proc