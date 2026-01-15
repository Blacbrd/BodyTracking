# utils.py
import numpy as np
import math

def calculate_angle(a, b, c):
    a2d = np.array(a[:2], dtype=float)
    b2d = np.array(b[:2], dtype=float)
    c2d = np.array(c[:2], dtype=float)
    ba = a2d - b2d
    bc = c2d - b2d
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))

    if denom == 0:
        return 0.0

    cosine_angle = np.dot(ba, bc) / denom
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)

def calculate_midpoint(a, b):
    return [(a[0] + b[0])/2, (a[1] + b[1])/2, (a[2] + b[2])/2]

def multiply_quat(q1, q2):
    q = (
        q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3],
        q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2],
        q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1],
        q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
    )

    return q

def find_quat_from_matrix(rm):
    m = np.array(rm)
    tr = m[0][0] + m[1][1] + m[2][2]

    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (m[2][1] - m[1][2]) / S
        qy = (m[0][2] - m[2][0]) / S
        qz = (m[1][0] - m[0][1]) / S

    elif (m[0][0] > m[1][1]) and (m[0][0] > m[2][2]):
        S = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        qw = (m[2][1] - m[1][2]) / S
        qx = 0.25 * S
        qy = (m[0][1] + m[1][0]) / S
        qz = (m[0][2] + m[2][0]) / S

    elif m[1][1] > m[2][2]:
        S = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        qw = (m[0][2] - m[2][0]) / S
        qx = (m[0][1] + m[1][0]) / S
        qy = 0.25 * S
        qz = (m[1][2] + m[2][1]) / S

    else:
        S = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        qw = (m[1][0] - m[0][1]) / S
        qx = (m[0][2] + m[2][0]) / S
        qy = (m[1][2] + m[2][1]) / S
        qz = 0.25 * S

    norm = math.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    if norm == 0:
        return (1.0, 0.0, 0.0, 0.0)

    return (qw/norm, qx/norm, qy/norm, qz/norm)

def find_matrix_from_quat(q):
    w, x, y, z = q

    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])

def inverse_quat(q):
    return (q[0], -q[1], -q[2], -q[3])

def find_yaw_angle(rotation_matrix, base_rotation):

    if base_rotation is None:
        return 0.0

    q_current = find_quat_from_matrix(rotation_matrix)
    q_base = find_quat_from_matrix(base_rotation)
    q_rel = multiply_quat(q_current, inverse_quat(q_base))
    R_rel = find_matrix_from_quat(q_rel)
    yaw = math.atan2(R_rel[0,2], R_rel[2,2])

    return yaw

def find_pitch_angle(rotation_matrix, base_rotation):

    if base_rotation is None:
        return 0.0

    q_current = find_quat_from_matrix(rotation_matrix)
    q_base = find_quat_from_matrix(base_rotation)
    q_rel = multiply_quat(q_current, inverse_quat(q_base))
    R_rel = find_matrix_from_quat(q_rel)
    pitch = math.asin(-R_rel[1,2])

    return pitch
