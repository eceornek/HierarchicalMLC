import cv2
import xml.etree.ElementTree as ET
import os
import math
from PIL import Image
import torch
import networkx as nx
from networkx.algorithms.dag import ancestors

my_tree = nx.DiGraph()
my_tree.add_node('insan')
my_tree.add_node('arac')
my_tree.add_edge('insan', 'askeri')
my_tree.add_edge('askeri', 'silahlı')
my_tree.add_edge('askeri', 'silahsız')
my_tree.add_edge('insan', 'sivil')
my_tree.add_edge('arac', 'sivil arac')
my_tree.add_edge('arac', 'askeri arac')
my_tree.add_edge('askeri arac', 'lastikli')
my_tree.add_edge('askeri arac', 'paletli')
my_tree.add_edge('paletli', 'tank')
my_tree.add_edge('paletli', 'ZMA')
my_tree.add_edge('tank', 'leopard')
my_tree.add_edge('tank', 'm60')
my_tree.add_edge('tank', 'm48')

c_list = {'insan':0, 'arac':1, 'askeri':2, 'sivil':3, 'silahlı':4, 'silahsız':5, \
    'sivil arac':6, 'askeri arac':7, 'lastikli':8, 'paletli':9, 'tank':10, 'ZMA':11,\
    'leopard':12, 'm60':13, 'm48':14}

def process(xmlfiles, videofiles):
    file = open('dataset.txt','w', encoding="utf-8")
    i=0
    length=len(xmlfiles)
    for k in range(length):
        video = videofiles[k]
        xmlfile = xmlfiles[k]
        cap = cv2.VideoCapture(video)
        xml_tree = ET.parse(xmlfile)
        root = xml_tree.getroot()
        images = root.iter('image')
        while cap.isOpened():
            retval, frame = cap.read()
            if not retval: break
            img = next(images)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            for box in img.iter('box'):
                box_dict = {}
                label = box.get('label')
                box_dict['label'] = label
                xtl = int(round(float(box.get('xtl'))))
                if xtl<0:
                    xtl=0
                ytl = int(round(float(box.get('ytl'))))
                xbr = int(round(float(box.get('xbr'))))
                ybr = int(round(float(box.get('ybr'))))  
                width = xbr-xtl
                height = ybr-ytl
                img_size = width*height
                if img_size<1200: continue
                crop_frame = frame.crop((xtl, ytl, xbr, ybr))
                path = 'dataset'
                crop_frame.save(os.path.join(path , f'{str(i)}.png'))
                for attr in box.iter('attribute'):
                    attr_name = attr.get('name')
                    answer=attr.text
                    box_dict[attr_name] = answer
                # for output vector
                label_list = []
                output = [0 for i in range(15)]
                if label=="insan":
                    label_list.append(label)
                    label_list.append(box_dict['tür'])
                    label_list.append(box_dict['silah nitelik'])       
                else:
                    if box_dict['ek nitelik'] != "yok":
                        supers = list(ancestors(my_tree, box_dict['ek nitelik']))
                        label_list = label_list + supers
                        label_list.append(box_dict['ek nitelik'])
                    else:
                        label_list.append(label)
                        label_list.append(box_dict['tür']+" arac")
                for label in label_list:
                    output[c_list[label]] = 1
                for x in output:
                    file.write(str(x)+" ")
                file.write("\n")
                i+=1
    cap.release()
    cv2.destroyAllWindows()
    file.close()

xml_files=["annotations_1.xml","annotations_2.xml","annotations_6.xml","annotations_7.xml"]
video_files=["front_1_speedup_cropped.mp4","front_2_speedup_cropped.mp4","front_6_speedup_cropped.mp4","front_7_speedup_cropped.mp4"]
process(xml_files, video_files)
