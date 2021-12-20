import copy
#pip install streamlit-drawable-canvas 
#설치해야함.


def make_detection_canvas(points):
    #(x,y,width,height)

    object = {
        "type":"rect",
        "fill" : "rgba(255,165,0,0.3)",
        "stroke" : "#000"}

    objects_list = []

    for x1,y1,x2,y2 in points:
        object['left'] = x1
        object['top'] = y1
        object['width'] = x2 - x1
        object['height'] = y2 - y1

        copy_dict = copy.deepcopy(object)
        objects_list.append(copy_dict)

    json_file ={
        'version': '4.4.0',
        'objects': objects_list,
        'background': '#eee'
        }
    
    return json_file