import numpy as np


# This function gets a vector and returns its normalized form.
def normalize(vector):
    return vector / np.linalg.norm(vector)


# This function gets a vector and the normal of the surface it hit
# This function returns the vector that reflects from the surface
def reflected(vector, axis):
    return vector - 2 * (np.dot(vector, axis)) * axis

## Lights


class LightSource:
    def __init__(self, intensity):
        self.intensity = intensity


class DirectionalLight(LightSource):

    def __init__(self, intensity, direction):
        super().__init__(intensity)
        self.direction = normalize(direction)

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self,intersection_point):
        return Ray(intersection_point, self.direction)

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, intersection):
        return np.inf

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        return self.intensity


class PointLight(LightSource):
    def __init__(self, intensity, position, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from a point to the light source
    def get_light_ray(self,intersection):
        return Ray(intersection, normalize(self.position - intersection))

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self,intersection):
        return np.linalg.norm(intersection - self.position)

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        return self.intensity / (self.kc + self.kl * d + self.kq * (d ** 2))


class SpotLight(LightSource):
    def __init__(self, intensity, position, direction, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.direction = np.array(direction)
        self.kc  = kc
        self.kl = kl
        self.kq = kq

    # Returns the ray that goes from the light source to a given point
    def get_light_ray(self, intersection):
        return Ray(intersection, normalize(self.position - intersection))

    def get_distance_from_light(self, intersection):
        return np.linalg.norm(intersection - self.position)

    def get_intensity(self, intersection):
        unit_vector_to_intersection = normalize(intersection - self.position)
        distance = self.get_distance_from_light(intersection)
        unit_light_direction = normalize(-1 * self.direction)
        intensity_factor = np.dot(unit_vector_to_intersection, unit_light_direction)
        attenuation = self.kc + self.kl * distance + self.kq * (
                    distance ** 2)
        return (self.intensity * intensity_factor) / attenuation


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    # The function is getting the collection of objects in the scene and looks for the one with minimum distance.
    # The function should return the nearest object and its distance (in two different arguments)
    def nearest_intersected_object(self, objects):
        nearest_object = None
        dist = np.inf
        for object in objects:
            intersection = object.intersect(self)
            if not intersection:
                # no intersection with this object
                continue

            if intersection[0] < dist:
                dist = intersection[0]
                nearest_object = intersection[1]

        return nearest_object, dist


class Object3D:
    def set_material(self, ambient, diffuse, specular, shininess, reflection):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflection = reflection

    def computeDiffuse(self, intensity, normal, ray_of_light):
        return intensity * self.diffuse * np.dot(normal, ray_of_light)

    def computeSpecular(self, intensity, v, R):
        return self.specular * intensity * (np.power(np.dot(v, R), self.shininess))


class Plane(Object3D):
    def __init__(self, normal, point):
        self.normal = np.array(normal)
        self.point = np.array(point)

    def intersect(self, ray: Ray):
        v = self.point - ray.origin
        t = np.dot(v, self.normal) / (np.dot(self.normal, ray.direction) + 1e-6)
        if t > 0:
            return t, self
        else:
            return None

    def compute_normal(self, *args):
        return self.normal
    
    def computeDiffuse(self, intensity, normal, ray_of_light):
        return super().computeDiffuse(intensity, normal, ray_of_light)

    def computeSpecular(self, intensity, v, R):
        return super().computeSpecular(intensity, v, R)



class Triangle(Object3D):
    """
        C
        /\
       /  \
    A /____\ B

    The fornt face of the triangle is A -> B -> C.
    
    """
    def __init__(self, a, b, c):
        self.a = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)
        self.normal = self.compute_normal()

    # computes normal to the trainagle surface. Pay attention to its direction!
    def compute_normal(self, *args):
        edge1 = self.b - self.a
        edge2 = self.c - self.a
        normal = np.cross(edge1, edge2)
        normal = normal / np.linalg.norm(normal)
        return normal


    def computeDiffuse(self, intensity, normal, ray_of_light):
        return super().computeDiffuse(intensity, normal, ray_of_light)

    def computeSpecular(self, intensity, v, R):
        return super().computeSpecular(intensity, v, R)
    
    def intersect(self, ray: Ray):
        epsilon = 1e-6
        a_b = self.b - self.a
        a_c = self.c - self.a
        p_vec = np.cross(ray.direction, a_c)
        det = np.dot(a_b, p_vec)
        if abs(det) < epsilon:
            return None
        inv_det = 1.0 / det
        t_vec = ray.origin - self.a
        u = np.dot(t_vec, p_vec) * inv_det
        if u < 0 or u > 1:
            return None
        q_vec = np.cross(t_vec, a_b)
        v = np.dot(ray.direction, q_vec) * inv_det
        if v < 0 or u + v > 1:
            return None
        t = np.dot(a_c, q_vec) * inv_det
        intersection_point = ray.origin + t * ray.direction
        normal_at_intersection = normalize(intersection_point - self.normal)
        return t, self, normal_at_intersection

class Pyramid(Object3D):
    """     
            D
            /\*\
           /==\**\
         /======\***\
       /==========\***\
     /==============\****\
   /==================\*****\
A /&&&&&&&&&&&&&&&&&&&&\ B &&&/ C
   \==================/****/
     \==============/****/
       \==========/****/
         \======/***/
           \==/**/
            \/*/
             E 
    
    Similar to Traingle, every from face of the diamond's faces are:
        A -> B -> D
        B -> C -> D
        A -> C -> B
        E -> B -> A
        E -> C -> B
        C -> E -> A
    """
    def __init__(self, v_list):
        self.v_list = v_list
        self.triangle_list = self.create_triangle_list()

    def create_triangle_list(self):
        l = []
        t_idx = [
                [0,1,3],
                [1,2,3],
                [0,3,2],
                 [4,1,0],
                 [4,2,1],
                 [2,4,0]]
        # TODO
        return l

    def apply_materials_to_triangles(self):
        # TODO
        pass

    def intersect(self, ray: Ray):
        # TODO
        pass

class Sphere(Object3D):
    def __init__(self, center, radius: float):
        self.center = center
        self.radius = radius

    def compute_normal(self, *args):
        return normalize(args[0] - self.center)

    def _calculate_discriminant(self, b, c):
        return (b * b) - (4 * c)

    def _find_roots(self, b, discriminant):
        sqrt_delta = np.sqrt(discriminant)
        return (-b + sqrt_delta) / 2, (-1 * b - sqrt_delta) / 2

    def intersect(self, ray):
        normal = ray.origin - self.center
        b = 2 * np.dot(ray.direction, normal)
        c = np.linalg.norm(normal) ** 2 - self.radius ** 2
        discriminant = self._calculate_discriminant(b, c)

        if discriminant > 0:
            root1, root2 = self._find_roots(b, discriminant)
            if root1 >= 0 and root2 >= 0:
                minroot = min(root1, root2)
                intersection = ray.origin + minroot * ray.direction
                normal_at_intersection = normalize(intersection - self.center)
                return minroot, self, normal_at_intersection

        return np.inf, self, normalize(ray.origin - self.center)

    def computeDiffuse(self, intensity, normal, ray_of_light):
        return super().computeDiffuse(intensity, normal, ray_of_light)

    def computeSpecular(self, intensity, v, R):
        return super().computeSpecular(intensity, v, R)


