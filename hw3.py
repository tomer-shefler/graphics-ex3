from helper_classes import Plane, PointLight, Sphere, DirectionalLight, Ray, normalize, reflected, Triangle
import numpy as np

import matplotlib.pyplot as plt

def render_scene(camera, ambient, lights, objects, screen_size, max_depth):
    width, height = screen_size
    ratio = float(width) / height
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom

    image = np.zeros((height, width, 3))

    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            # screen is on origin
            pixel = np.array([x, y, 0])
            direction = normalize(pixel - camera)
            ray = Ray(camera, direction)

            color = np.zeros(3)
            color = calculate_color(camera, ambient, lights, objects, ray, max_depth, 1)

            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color,0,1)

    return image


def calculate_reflected_color(p, camera, ambient, lights, objects, ray, max_depth, level, normal_of_intersection, nearest_object):
    kr = nearest_object.reflection
    reflected_ray = Ray(p, normalize(reflected(ray.direction, normal_of_intersection)))
    return kr * calculate_color(camera, ambient, lights, objects, reflected_ray, max_depth, level)


def calculate_light_contribution(ambient_color, lights, p, normal_of_intersection, camera, objects, nearest_object, min_distance):
    color = np.float64(ambient_color)
    v = normalize(camera - p)
    for light in lights:
        ray_of_light = light.get_light_ray(p)
        nearest_object_to_light, distance_of_nearease_object_to_light = ray_of_light.nearest_intersected_object(objects)
        if not nearest_object_to_light or distance_of_nearease_object_to_light >= min_distance:
            r = normalize(reflected(ray_of_light.direction, normal_of_intersection))
            diffuse = nearest_object.computeDiffuse(light.get_intensity(p), normal_of_intersection, ray_of_light.direction)
            specular = nearest_object.computeSpecular(light.get_intensity(p), v, r)
            color += diffuse + specular

    return color


def calculate_ambient_color(ambient, nearest_object):
    return nearest_object.ambient * ambient


def initialize_ray_trace(ray, nearest_object, min_distance):
    intersection_p = ray.origin + (min_distance * ray.direction)
    normal_of_intersection = nearest_object.compute_normal(intersection_p)
    p = intersection_p + normal_of_intersection / (np.e ** 2)
    return p, normal_of_intersection


def calculate_color(camera, ambient, lights, objects, ray, max_depth, level):
    if level > max_depth:
        return np.zeros(3)
    level += 1

    nearest_object, min_distance = ray.nearest_intersected_object(objects)
    if nearest_object is None:
        return np.zeros(3)

    p, normal_of_intersection = initialize_ray_trace(ray, nearest_object, min_distance)

    ambient_color = calculate_ambient_color(ambient, nearest_object)
    color = calculate_light_contribution(ambient_color, lights, p, normal_of_intersection, camera, objects, nearest_object, min_distance)
    color += calculate_reflected_color(p, camera, ambient, lights, objects, ray, max_depth, level, normal_of_intersection, nearest_object)

    return color


def your_own_scene():
    camera = np.array([0, 0, 1])

    # Define lights
    lights = [
        PointLight(position=[5, 5, 5], intensity=np.array([0.8, 0.8, 1]), kc=0.1, kl=0.1, kq=0.1),
        DirectionalLight(direction=[-1, -1, -1], intensity=np.array([0.2, 0.5, 1]))
    ]

    # Define objects
    
    sphere1 = Sphere([0, 0, -5], 1)
    sphere2 = Sphere([-2, 1, -4], 1) 
    plane = Plane(point=[0, -1, 0], normal=[0, 1, 0])  # Gray plane
    
    sphere1.set_material(ambient=[0, 1, 0], diffuse=[0, 4, 4], specular=[0.5, 0.5, 0.5], shininess=32, reflection=0.1)
    sphere2.set_material(ambient=[1, 0, 0], diffuse=[0, 4, 4], specular=[0.5, 0.5, 0.5], shininess=32, reflection=0)
    plane.set_material(ambient=[0, 1, 1], diffuse=[5, 5, 0], specular=[0.3, 0.3, 0.3], shininess=8, reflection=0.3)

    v_list = np.array([[-1,0,-1],
                    [1,0,-1],
                    [0,1.5,-1.5]])

    triangle = Triangle(*v_list)
    triangle.set_material([1, 1, 1], [255,255,0], [1, 1, 1], 100, 0.5)


    v_list2 = np.array([[-1,1,-1],
                    [1,1,-1],
                    [0,-0.5,-1.5]])

    triangle2 = Triangle(*v_list2)
    triangle2.set_material([1, 1, 1], [255,255,0], [1, 1, 1], 100, 0.5)


    objects = [plane, triangle,triangle2]
    return camera, lights, objects

