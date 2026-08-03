# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 10:48:53 2024

@author: jared
"""

import numpy as np
import cv2
import sys
import math
import os, os.path
from matplotlib import pyplot as plt

""" 

Please refer to README at bottom of script to learn how to use!

"""
    
class image_Convert:
    
    #1.) 
    def __init__(self,argv,camera_parameters,external_optics,camera_range,image_visuals, Image_directory = "Images",verbose = False):
        
        # print("Flir Convert Init!")
        # print("__"*10)
        
        # Image file path handling 
        self.filepath = "0"
        self.photo_directory = Image_directory
        
        # Camera parameters
        self.camera_range = camera_range
        self.Emiss = camera_parameters["Emiss"]
        self.Tau = camera_parameters["Tau"]
        self.TAtm = camera_parameters["TAtm"]
        self.TAtmC = camera_parameters["TAtmC"]
        self.TRefl = camera_parameters["TRefl"]
        self.Humidity = camera_parameters["Humidity"]
        self.distance = camera_parameters["distance"]
        
        # External optics parameters
        self.TransmissionExtOptics = external_optics["TransmissionExtOptics"]
        self.TextOptics = external_optics["TextOptics"]
        
        # imaging and plotting
        self.colormap = image_visuals["colormap"]
        self.ROI_loc = image_visuals["ROI Center"]
       
        ## debugging
        self.verbose = verbose
        
    def getFileList(self): # 2.) 
        filetype=".TIFF" ## needs to be TIFF from HT-DABI Control Software
        file_list = []
        path_present = os.path.isdir(self.photo_directory)
        
        if path_present == True:
            for file in os.listdir(self.photo_directory):
                #print(file)
                if file.endswith(filetype.upper()) or file.endswith(filetype.lower()):
                    file_list.append(file)
                
            if len(file_list) == 0:
                print("ERROR!")
                print("incorrect path, filetype, or no pictures in folder!")
        else:
            os.makedirs(self.photo_directory)
            print("ERROR!")
            print("No directory present, creating Images folder")
            
        return file_list

    def set_Camera_Range(self): # 3.) 
        
        # Determines range of IR camera from the three FLIR presets
        if self.camera_range ==  0:
            self.R = 14513.9
            self.B = 1437.3
            self.F = 1.0
            self.J1 = 68.43
            self.J0 = 4342
            # print("Camera Range: 0 Selected, -40 - 150C")
        
        
        elif self.camera_range ==  1:
            self.R = 14204.6
            self.B = 1372
            self.F = 1.75
            self.J1 = 7.91534
            self.J0 = 5741
            
            # print("Camera Range: 1 Selected, 100 - 650 C")
        
        elif self.camera_range ==  2:
            self.R = 19574
            self.B = 1503.7
            self.F = 1.1
            self.J1 = 1.4803
            self.J0 = 6091
            
            # print("Camera Range: 2 Selected, 300 - 2000 C")
            
        elif self.camera_range ==  3:
            self.R = 14513.9
            self.B = 1388.5
            self.F = 1.0
            self.J1 = 93
            self.J0 = 6540
            
            # print("MDI Flir0")
    
    def counts2temp(self, data_counts): # 4.) 
    
        K1 = 1 / (self.Tau * self.Emiss * self.TransmissionExtOptics)
        
        # Pseudo radiance of the reflected environment
        r1 = ((1-self.Emiss)/self.Emiss) * (self.R/(np.exp(self.B/self.TRefl)-self.F))
        
        # Pseudo radiance of the atmosphere
        r2 = ((1 - self.Tau)/(self.Emiss * self.Tau)) * (self.R/(np.exp(self.B/self.TAtm)-self.F)) 
        
        # Pseudo radiance of the external optics
        r3 = ((1-self.TransmissionExtOptics) / (self.Emiss * self.Tau * self.TransmissionExtOptics)) *(self.R/(np.exp(self.B/self.TextOptics)-self.F))
                
        K2 = r1 + r2 + r3
        
        data_obj_signal = (data_counts - self.J0)/self.J1
        log_arg = abs(self.R/((K1 * data_obj_signal) - K2) + self.F)
        
        data_temp = (self.B / np.log(log_arg)) - 273.15
        
        return data_temp
    
    def convert(self): #5.) 
        
        try:
            unprocessed_image = cv2.imread(self.fullpath,cv2.IMREAD_UNCHANGED) #zero loads image in 16 bit gray
            
            if self.verbose == True:
                y = 90
                x = 230
                _intensity = unprocessed_image[y,x]
                in_min, in_max = unprocessed_image.min(), unprocessed_image.max()
                # print(unprocessed_image.dtype)
                # print(in_min, in_max)

                # print("intensity ",  _intensity)

                cv2.imshow('original', unprocessed_image)
                cv2.waitKey(0)

            ### Perform Temp Conversion ###
            temperature_image = np.zeros_like(unprocessed_image, dtype=np.float32)

            for y in range(unprocessed_image.shape[0]):
                for x in range(unprocessed_image.shape[1]):
                    _intensity = unprocessed_image[y, x]
                    temperature_image[y, x] = self.counts2temp(_intensity)
            
            
            if self.verbose == True:
                y = 220
                x = 300
                print( temperature_image[y, x], "r")
               
                
                temp_min, temp_max = temperature_image.min(), temperature_image.max()
                
                print(temp_min, temp_max)
                
                temperature_image_scaled = cv2.normalize(temperature_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
                # Show the processed image
                cv2.circle(temperature_image_scaled, (x, y), 1, (0, 0, 255), -1)
                cv2.line(temperature_image_scaled,(x,y),(x,y+5),(0,0,255),1)
                cv2.imshow('Temperature Image', temperature_image_scaled)
                
                cv2.waitKey(0)
        
           
        except:
            print("Convert func, processing error!")
            temperature_image = np.zeros_like(unprocessed_image, dtype=np.float32)
    
        return temperature_image
    
    
    def show_temperature_image(self, temperature_data): # 6.) 
        try:
            plt.imshow(temperature_data,self.colormap)
            cbar = plt.colorbar()
            cbar.set_label('Temperature (°C)')
            
            y,x = self.ROI_loc[1], self.ROI_loc[0]
            temp_value = temperature_data[y, x]
            plt.scatter(x, y, color='red', label=f"Temp: {temp_value:.2f} °C", zorder=5, s=  4)
            plt.legend()

            plt.title(self.filename)
            plt.savefig( "Converted_" + self.filename)
        except:
            print("Plotting Error!")
            
            
    def process(self): ## 7.) 
        
        file_list = self.getFileList() 
        
        counter = 0
        for name in file_list:
            counter = counter + 1
            print("Processing : ",  str(counter) + "/" + str(len(file_list)) , " " ,name   )
        
            self.set_Camera_Range()
            
            self.fullpath = os.path.join(self.photo_directory, name)
            self.filename = name
            
            temperature_data = self.convert()
            self.show_temperature_image(temperature_data)
            #zoomed_data = self.show_temperature_zoom(temperature_data)
            
            return temperature_data
            
## main is the same script, you can create your own seperate script and call this class if it is in same parent folder.
## for example 'import  FLIR_Convert' then call functions
## import Flir_Convert
## from Flir_Convert import Convert
if __name__ == "__main__":
    
    
    
    ''' 
                                README!!
    class FLIR_Convert can be used to automate many temp images at once. 
    
        functions:
        1.) __init__ : uses the initialized dictionaries seen below.
        2.) getFileList: gets all the TIFF photo files in local directory's "Images" subfolder. 
            if no "Images" subfolder, it will create one. Then place .TIFF photos inside.
        3.) set_Camera_Range: sets the specified camera range. 
        4.) counts2temp: called by convert func to convert the raw image pixel intensity to temperature
        5.) convert: reads unprocessed image into counts2temp. returns temperature image
        6.) show_temperature_image: will plot the image on a graph with a selectable ROI point. Also saves the figure.
        7.) process: main function that calls all of subfunctions. iterates thru all raw unprocessed images. 
    
    '''
    
    ## these are dictionaries used in FLIR_Convert class, they are used to initial parameters
    camera_parameters = {}

    external_optics = {}
    
    image_visuals = {}
    
    # reflected energy
    camera_parameters["Emiss"] = 1.0
    camera_parameters["distance"] = 1.0
    camera_parameters["TRefl"] = 293.14

    # atmospheric attenuation
    camera_parameters["TAtmC"] = 293.14
    camera_parameters["TAtm"] = camera_parameters["TAtmC"] + 273.15
    camera_parameters["Humidity"] = 0.0/100
    camera_parameters["Tau"] = 1.0

    # external optics
    external_optics["TextOptics"] = 20 ## external optics temp. 
    external_optics["TransmissionExtOptics"] = 1.0
    
    ## visuals and plotting
    image_visuals["colormap"] = 'viridis' ## more options https://matplotlib.org/stable/users/explain/colors/colormaps.html
    image_visuals["colormap"] = 'plasma'
    image_visuals["ROI Center"] = [295,260] ## x,y. note HT-DABI has inverted y coordinates, see non-raw saved image. 


    ## camera range w/ camera specific constants. 0 = low range, 1 = mid range: 100 - 650 C, 2 = # high range:  300 - 2000 C
    camera_range = 2
    
    # create an instance with desired inital settings, ie the above dictionaries. 
    _instance = image_Convert(sys.argv,camera_parameters,external_optics,camera_range,image_visuals)
    
    ## uncomment to use own specified photo directory
    #Image_directory = "Images2"
    #_instance = image_Convert(sys.argv,camera_parameters,external_optics,camera_range,image_visuals,Image_directory)
    # call process to convert and save the images
    temp_data = _instance.process()