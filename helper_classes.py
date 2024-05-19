import numpy as np

# Helper Functions

def normalize(vector):
    return vector / np.linalg.norm(vector)

def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis

# Classes

class LightSource:
    def __init__(self, intensity):
        self.intensity = intensity


class DirectionalLight(LightSource):
    def __init__(self, intensity, direction):
        super().__init__(intensity)
        self.direction = normalize(direction)

    def get_light_ray(self, intersection_point):
        return Ray(intersection_point - 1e-5 * self.direction, -self.direction)

    def get_distance_from_light(self, intersection):
        return np.inf

    def get_intensity(self, intersection):
        return self.intensity


class PointLight(LightSource):
    def __init__(self, intensity, position, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    def get_light_ray(self, intersection):
        return Ray(intersection + 1e-5 * normalize(self.position - intersection), normalize(self.position - intersection))

    def get_distance_from_light(self, intersection):
        return np.linalg.norm(intersection - self.position)

    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        return self.intensity / (self.kc + self.kl * d + self.kq * (d ** 2))


class SpotLight(LightSource):
    def __init__(self, intensity, position, direction, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.direction = normalize(direction)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    def get_light_ray(self, intersection):
        return Ray(intersection + 1e-5 * normalize(self.position - intersection), normalize(self.position - intersection))

    def get_distance_from_light(self, intersection):
        return np.linalg.norm(intersection - self.position)

    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        return self.intensity / (self.kc + self.kl * d + self.kq * (d ** 2))


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    def nearest_intersected_object(self, objects):
        nearest_obj = None
        min_distance = np.inf
        for obj in objects:
            t = obj.intersect(self)
            if t is not None and t < min_distance:
                min_distance = t
                nearest_obj = obj
        return nearest_obj, min_distance


class Object3D:
    def set_material(self, ambient, diffuse, specular, shininess, reflection):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflection = reflection


class Plane(Object3D):
    def __init__(self, normal, point):
        self.normal = np.array(normal)
        self.point = np.array(point)

    def intersect(self, ray):
        t = np.dot(self.point - ray.origin, self.normal) / np.dot(ray.direction, self.normal)
        if t > 0:
            return t
        return None


class Triangle(Object3D):
    def __init__(self, a, b, c):
        self.a = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)
        self.normal = normalize(np.cross(b - a, c - a))

    def compute_normal(self):
        return self.normal

    def intersect(self, ray):
        epsilon = 1e-6
        ab = self.b - self.a
        ac = self.c - self.a
        pvec = np.cross(ray.direction, ac)
        det = np.dot(ab, pvec)
        if abs(det) < epsilon:
            return None
        inv_det = 1.0 / det
        tvec = ray.origin - self.a
        u = np.dot(tvec, pvec) * inv_det
        if u < 0 or u > 1:
            return None
        qvec = np.cross(tvec, ab)
        v = np.dot(ray.direction, qvec) * inv_det
        if v < 0 or u + v > 1:
            return None
        t = np.dot(ac, qvec) * inv_det
        return t


class Pyramid(Object3D):
    def __init__(self, v_list):
        self.v_list = v_list
        self.triangle_list = self.create_triangle_list()

    def create_triangle_list(self):
        a, b, c, d, e = self.v_list
        triangle_list = [
            Triangle(a, b, d),
            Triangle(b, c, d),
            Triangle(a, c, b),
            Triangle(e, b, a),
            Triangle(e, c, b),
            Triangle(c, e, a)
        ]
        return triangle_list

    def apply_materials_to_triangles(self):
        for triangle in self.triangle_list:
            triangle.set_material(self.ambient, self.diffuse, self.specular, self.shininess, self.reflection)

    def intersect(self, ray):
        epsilon = 1e-6
        min_t = np.inf
        for triangle in self.triangle_list:
            t = triangle.intersect(ray)
            if t is not None and t < min_t:
                min_t = t
        if min_t < np.inf:
            return min_t
        return None


class Sphere(Object3D):
    def __init__(self, center, radius):
        self.center = np.array(center)
        self.radius = radius

    def intersect(self, ray):
        L = self.center - ray.origin
        tca = np.dot(L, ray.direction)
        if tca < 0:
            return None
        d2 = np.dot(L, L) - tca * tca
        if d2 > self.radius * self.radius:
            return None
        thc = np.sqrt(self.radius * self.radius - d2)
        t0 = tca - thc
        t1 = tca + thc
        if t0 > t1:
            t0, t1 = t1, t0
        if t0 < 0:
            t0 = t1
        if t0 < 0:
            return None
        return t0

