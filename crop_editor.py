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

    print(points)
    for x,y,width,height in points:
        object['left'] = x
        object['top'] = y
        object['width'] = width - x
        object['height'] = height - y

        copy_dict = copy.deepcopy(object)
        objects_list.append(copy_dict)

    json_file ={
        'version': '4.4.0',
        'objects': objects_list,
        'background': '#eee'
        }
    
    return json_file