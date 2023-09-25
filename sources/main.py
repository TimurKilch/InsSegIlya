# # #!/usr/bin/python3
# # import os
# # import json
# # from ultralyticsplus import YOLO, render_result
# # import matplotlib.pyplot as plt
# # import numpy as np
# # import matplotlib
# # import cv2
# # import geopandas as gpd
# # import rasterio
# # from shapely.geometry import Polygon

# # matplotlib.use('Qt5Agg')

# # try:
# #     from osgeo import gdal
# # except ImportError:
# #     import gdal

# # def check_tif(tif_path):
# #     # Проверка, существует ли файл по указанному пути
# #     if not os.path.exists(tif_path):
# #         print("HTTP/1.1 404 Not Found")
# #         print("Content-Type: text/plain")
# #         print("")
# #         print("Bad Request")
# #         return

# #     # Проверка, является ли файл Tiff
# #     if not tif_path.lower().endswith('.tif') or not tif_path.lower().endswith('.tiff') or not tif_path.lower().endswith('.geotiff'):
# #         print("HTTP/1.1 415 Unsupported Media Type")
# #         print("Content-Type: text/plain")
# #         print("")
# #         print("Not tiff input")
# #         return

# #     print("HTTP/1.1 200 OK")
# #     print("Content-Type: text/plain")
# #     print("")
# #     print(f"Tif file correct")

# # def get_instances_dict(image): # image = tif_path, output_image = output_path
# #     # load model
# #     model = YOLO('keremberke/yolov8m-building-segmentation')

# #     # set model parameters
# #     model.overrides['conf'] = 0.25  # NMS confidence threshold
# #     model.overrides['iou'] = 0.45  # NMS IoU threshold
# #     model.overrides['agnostic_nms'] = True  # NMS class-agnostic
# #     model.overrides['max_det'] = 1000  # maximum number of detections per image

# #     # perform inference
# #     results = model.predict(image)
# #     print(type(results[0]))

# #     boxes = results[0].boxes.xyxy.tolist()
# #     masks = results[0].masks.masks.tolist()
# #     # probs = results[0].probs.probs.tolist()
# #     # orig_shape = results[0].orig_shape

# #     result_dict = {
# #         "boxes": boxes,
# #         "masks": masks,
# #         # "probs": probs,
# #         # "orig_shape": orig_shape.tolist()
# #     }

# #     render = render_result(model=model, image=image, result=results[0])
# #     render.show()

# #     masks = results[0].masks.masks

# #     # Initialize an empty image with the same shape as an individual mask
# #     contour_image = np.zeros(masks[0].shape, dtype=np.uint8)

# #     # Define the number of vertices for the approximated polygons
# #     n_vertices = 50

# #     instance_polygons = {}

# #     # Iterate through the masks to find contours and approximate the polygons
# #     for i, mask in enumerate(masks):
# #         img = (mask.cpu().numpy() * 255).astype(np.uint8)
# #         contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# #         # Approximate the contours to polygons with n vertices
# #         approx_contours = []
# #         for cnt in contours:
# #             epsilon = 0.01 * cv2.arcLength(cnt, True)
# #             approx_contours.append(cv2.approxPolyDP(cnt, epsilon, True))

# #         # Store the instance polygon in the dictionary
# #         instance_polygons[i] = []
# #         for polygon in approx_contours:
# #             instance_polygons[i].extend([list(point[0]) for point in polygon])

# #         # Draw the approximated contours on the contour_image
# #         contour_image = cv2.drawContours(contour_image, approx_contours, -1, (255, 255, 255), thickness=2)

# #     orig_height, orig_width = plt.imread(image).shape[:2]
# #     contour_image_resized = cv2.resize(contour_image, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)

# #     # Now instance_polygons is a dictionary with the polygon vertices for each instance
# #     (xs, ys) = ([], [])
# #     for polygon in instance_polygons.values():
# #         polygon_np = np.array(polygon)
# #         xs.append(polygon_np[:, 0].copy())
# #         ys.append(polygon_np[:, 1].copy())

# #     height, width = contour_image.shape[:2]
# #     resized_width_factor, resized_height_factor = orig_width / width, orig_height / height
# #     x_transformed = [np.round(x * resized_width_factor).astype(int) for x in xs]
# #     y_transformed = [np.round(y * resized_height_factor).astype(int) for y in ys]

# #     instance_polygons_transformed = {i + 1: np.column_stack((x_transformed[i], y_transformed[i])) for i in
# #                                      range(len(x_transformed))}


# #     # Save the contour_image ЕСЛИ НАДО БУДЕТ СОХРАНЯТЬ НАДО ВОЗВРАЩАТЬ ЕГО
# #     #cv2.imwrite(output_image, contour_image_resized)

# #     # Now instance_polygons_transformed is a dictionary with the polygon vertices for each instance in original image coordinates
# #     result = {}
# #     for key, value in instance_polygons_transformed.items():
# #         key -= 1  # Декрементируем ключ на 1, чтобы начать с нулевого индекса
# #         result[key] = []
# #         for coord_pair in value:
# #             result[key].append(list(coord_pair))

# #     return (result)

# # def saveDict_to_shapefile(input_dict, tif_path, output_path): # input_dict = get_instances_dict()
# #     # Read GeoTIFF file
# #     with rasterio.open(tif_path) as src:
# #         transform = src.transform
# #         crs = src.crs

# #     # Convert dictionary to a new format
# #     new_dict = []
# #     for key, value in input_dict.items():
# #         coordinates = []
# #         for coords in value:
# #             x_coord = coords[0] * transform.a + coords[1] * transform.b + transform.c
# #             y_coord = coords[0] * transform.d + coords[1] * transform.e + transform.f
# #             coordinates.append((x_coord, y_coord))

# #         entry = {"id": key, "class": "building", "points": Polygon(coordinates)}
# #         new_dict.append(entry)

# #     # Create a GeoDataFrame and set the CRS
# #     gdf = gpd.GeoDataFrame(new_dict, geometry="points", crs=crs)

# #     # Save the GeoDataFrame to a shapefile
# #     gdf.to_file(output_path + "/result_shapefile.shp")

# #     # Проверка, является созданный файл GeoJSON
# #     if not output_path.lower().endswith('.shp'):
# #         print("HTTP/1.1 415 Unsupported Media Type")
# #         print("Content-Type: text/plain")
# #         print("")
# #         print("Not shape created")
# #         return


# # if __name__ == '__main__':
# #     args = os.environ.get("QUERY_STRING","/input/test_365 (19).tif&/output/test.geojson")
# #     args_list = args.split("&")
# #     if len(args_list) != 2:
# #         print("HTTP/1.1 400 Bad request")
# #         print("Content-Type: text/plain")
# #         print("")
# #         print("Not 2 arguments")

# #     #tif_to_geojson(args_list[0], args_list[1])
# #     check_tif(args_list[0])
# #     saveDict_to_shapefile(get_instances_dict(args_list[0]), args_list[0], args_list[1])
# #     print("HTTP/1.1 200 OK")
# #     print("Content-Type: text/plain")
# #     print("")
# #     print(f"Shape created - {os.environ}")


#!/usr/bin/python3
import os
import json
from ultralyticsplus import YOLO, render_result
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import cv2
import geopandas as gpd
import rasterio
from shapely.geometry import Polygon

matplotlib.use('Qt5Agg')

try:
    from osgeo import gdal
except ImportError:
    import gdal

def check_tif(tif_path):
    # Проверка, существует ли файл по указанному пути
    if not os.path.exists(tif_path):
        return False

    # Проверка, являpython3 ./sources/main.py  1> std.txt 2>err.txt'max_det'] = 1000  # maximum number of detections per image

    # perform inference
    results = model.predict(image)
    if not results:
        return None

    boxes = results[0].boxes.xyxy.tolist()
    masks = results[0].masks.masks.tolist()

    result_dict = {
        "boxes": boxes,
        "masks": masks,
    }

    render = render_result(model=model, image=image, result=results[0])
    render.show()

    masks = results[0].masks.masks

    # Initialize an empty image with the same shape as an individual mask
    contour_image = np.zeros(masks[0].shape, dtype=np.uint8)

    # Define the number of vertices for the approximated polygons
    n_vertices = 50

    instance_polygons = {}

    # Iterate through the masks to find contours and approximate the polygons
    for i, mask in enumerate(masks):
        img = (mask.cpu().numpy() * 255).astype(np.uint8)
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Approximate the contours to polygons with n vertices
        approx_contours = []
        for cnt in contours:
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx_contours.append(cv2.approxPolyDP(cnt, epsilon, True))

        # Store the instance polygon in the dictionary
        instance_polygons[i] = []
        for polygon in approx_contours:
            instance_polygons[i].extend([list(point[0]) for point in polygon])

        # Draw the approximated contours on the contour_image
        contour_image = cv2.drawContours(contour_image, approx_contours, -1, (255, 255, 255), thickness=2)

    orig_height, orig_width = plt.imread(image).shape[:2]
    contour_image_resized = cv2.resize(contour_image, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)

    # Now instance_polygons is a dictionary with the polygon vertices for each instance
    (xs, ys) = ([], [])
    for polygon in instance_polygons.values():
        polygon_np = np.array(polygon)
        xs.append(polygon_np[:, 0].copy())
        ys.append(polygon_np[:, 1].copy())

    height, width = contour_image.shape[:2]
    resized_width_factor, resized_height_factor = orig_width / width, orig_height / height
    x_transformed = [np.round(x * resized_width_factor).astype(int) for x in xs]
    y_transformed = [np.round(y * resized_height_factor).astype(int) for y in ys]

    instance_polygons_transformed = {i + 1: np.column_stack((x_transformed[i], y_transformed[i])) for i in
                                     range(len(x_transformed))}

    return instance_polygons_transformed

def saveDict_to_shapefile(input_dict, tif_path, output_path):
    if not input_dict:
        return

    # Read GeoTIFF file
    with rasterio.open(tif_path) as src:
        transform = src.transform
        crs = src.crs

    # Convert dictionary to a new format
    new_dict = []
    for key, value in input_dict.items():
        coordinates = []
        for coords in value:
            x_coord = coords[0] * transform.a + coords[1] * transform.b + transform.c
            y_coord = coords[0] * transform.d + coords[1] * transform.e + transform.f
            coordinates.append((x_coord, y_coord))

        entry = {"id": key, "class": "building", "points": Polygon(coordinates)}
        new_dict.append(entry)

    # Create a GeoDataFrame and set the CRS
    gdf = gpd.GeoDataFrame(new_dict, geometry="points", crs=crs)

    # Save the GeoDataFrame to a shapefile
    shapefile_path = os.path.splitext(output_path)[0]  # Remove the extension
    gdf.to_file(shapefile_path)

# if __name__ == '__main__':
#     args = os.environ.get("QUERY_STRING","/input/test_365 (19).tif&/output/")
#     args_list = args.split("&")
#     if len(args_list) != 2:
#         print("HTTP/1.1 400 Bad request")
#         print("Content-Type: text/plain")
#         print("")
#         print("Not 2 arguments")
#     else:
#         tif_path = args_list[0]
#         output_path = args_list[1]
#     ## TO DO ##
#         if not check_tif(tif_path):
#             print("HTTP/1.1 404 Not Found")
#             print("Content-Type: text/plain")
#             print("")
#             print("Bad Request")
#         else:
#             instance_polygons = get_instances_dict(tif_path)
#             if instance_polygons:
#                 saveDict_to_shapefile(instance_polygons, tif_path, output_path)
#                 print("HTTP/1.1 200 OK")
#                 print("Content-Type: text/plain")
#                 print("")
#                 print(f"Shape created")
#             else:
#                 print("HTTP/1.1 204 No Content")
#                 print("Content-Type: text/plain")
#                 print("")
#                 print("No content found")

if __name__ == '__main__':
    args = os.environ.get("QUERY_STRING", "/input/test_365 (19).tif&./output/")
    args_list = args.split("&")
    if len(args_list) != 2:
        print("HTTP/1.1 400 Bad request")
        print("Content-Type: text/plain")
        print("")
        print("Not 2 arguments")
    else:
        tif_path = args_list[0]
        output_path = args_list[1]
        if not check_tif(tif_path):
            print("HTTP/1.1 404 Not Found")
            print("Content-Type: text/plain")
            print("")
            print("Bad tiff argument")
        elif os.path.splitext(output_path)[1]:
            print("HTTP/1.1 400 Bad request")
            print("Content-Type: text/plain")
            print("")
            print("Bad output argument")
        elif os.path.isdir(os.path.dirname(output_path)):
            print("HTTP/1.1 400 Bad request")
            print("Content-Type: text/plain")
            print("")
            print("Bad output argument")
        else:
            instance_polygons = get_instances_dict(tif_path)
            if instance_polygons:
                saveDict_to_shapefile(instance_polygons, tif_path, output_path)
                print("HTTP/1.1 200 OK")
                print("Content-Type: text/plain")
                print("")
                print(f"Shape created")
            else:
                print("HTTP/1.1 204 No Content")
                print("Content-Type: text/plain")
                print("")
                print("No content found")

'''
Timur, [18.09.2023 11:49]
python3 ./sources/main.py  1> std.txt 2>err.txt

Timur, [18.09.2023 11:49]
QUERY_STRING="./sources/test.tiff&./" python3 ./sources/main.py

Timur, [18.09.2023 11:54]
dup2 - создал файл, знаешь его дескриптор, параметров подаешь, первый аргумент какой потом, заменить на какой. Первый аргумент 2 а второй это дескриптор файла

Timur, [18.09.2023 11:54]
Переведет с стд на дескриптор

Timur, [18.09.2023 11:54]
/dev/null если не нужны эти логи в нее передавать

1) проблема с выводом
2) проблема с выводом пути до папки
'''