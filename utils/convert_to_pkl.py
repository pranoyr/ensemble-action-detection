import os
import xml.etree.ElementTree as ET
import pickle as pk

list_path = '/Volumes/Seagate Expansion Drive/VehicleData/ImageSets/train.txt'
with open(list_path, "r") as file:
    img_files = file.read()
img_files = img_files.split()

dict_xml = {'bounding_boxes':[],'frame_no':"","labels":[]}
list_vehicles = []
list_color = []
list_labels = []
for file in img_files :
    root = ET.parse(os.path.join('/Volumes/Seagate Expansion Drive/VehicleData/Annotations',file)+'_v3.xml').getroot()    
    for root1 in root.findall('frame'):
        frame_no =  root1.get('num')
        for root2,root3 in zip(root1.findall('target_list/target/box'), root1.findall('target_list/target/attribute')):

            height = root2.get('height')
            width = root2.get('width')
            left = root2.get('left')
            top = root2.get('top')

            color = root3.get('color')
            vehicle_type = root3.get('vehicle_type')

            list_vehicles.append(vehicle_type)
            list_color.append(color)

            dict_xml['bounding_boxes'].append(int(float(left)))
            dict_xml['bounding_boxes'].append(int(float(top)))
            dict_xml['bounding_boxes'].append(int(float(left))+int(float(width)))
            dict_xml['bounding_boxes'].append(int(float(top))+int(float(height)))
            dict_xml['frame_no']=frame_no
            dict_xml['filename']=file

            dict_xml['labels'].append(color)
            dict_xml['labels'].append(vehicle_type)
            
            list_labels.append(dict_xml.copy())
            dict_xml = {'bounding_boxes':[],'frame_no':"","labels":[],'filename':""}


print(set(list_vehicles))
print(set(list_color))

with open('annotations.pkl', 'wb') as f:
    pk.dump(list_labels,f)

