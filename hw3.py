from helper_classes import *
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
            origin = camera
            direction = normalize(pixel - origin)
            ray = Ray(origin, direction)

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
            diffuse = nearest_object.calc_diffuse(light.get_intensity(p), normal_of_intersection, ray_of_light.direction)
            specular = nearest_object.calc_specular(light.get_intensity(p), v, r)
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


# Write your own objects and lights
# TODO
def your_own_scene():
    camera = np.array([0,0,1])
    lights = []
    objects = []
    return camera, lights, objects
