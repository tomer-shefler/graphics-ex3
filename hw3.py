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

            # Main loop for each pixel color computation
            for depth in range(max_depth):
                # Find the nearest intersection point with an object
                hit_obj, t = ray.nearest_intersected_object(objects)
                if hit_obj is None:
                    break

                hit_point = ray.origin + ray.direction * t
                normal = hit_obj.normal_at(hit_point)
                material = hit_obj.material

                # Compute lighting at the intersection point
                for light in lights:
                    light_ray = light.get_light_ray(hit_point)
                    in_shadow = False
                    for obj in objects:
                        if obj != hit_obj and obj.shadow_hit(light_ray):
                            in_shadow = True
                            break

                    if not in_shadow:
                        intensity = light.get_intensity(hit_point)
                        # Diffuse component
                        diffuse = max(np.dot(normalize(light_ray.direction), normal), 0)
                        color += material.diffuse * intensity * diffuse

                        # Specular component
                        if diffuse > 0:
                            reflect_dir = reflected(-light_ray.direction, normal)
                            specular = max(np.dot(normalize(ray.direction), reflect_dir), 0) ** material.shininess
                            color += material.specular * intensity * specular

                # Reflection
                reflect_ray = Ray(hit_point + 0.001 * normal, reflected(ray.direction, normal))
                ray = reflect_ray

            # Add ambient light
            color += ambient

            # Clamp color values to [0, 1]
            color = np.clip(color, 0, 1)

            image[i, j] = color

    return image



# Write your own objects and lights
# TODO
def your_own_scene():
    camera = np.array([0,0,1])
    lights = []
    objects = []
    return camera, lights, objects
