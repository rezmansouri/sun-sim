import sys
import numpy as np
from psipy.model import MASOutput
from hux_code.hux_propagation import apply_hux_f_model


def get_hux_f(f, r, p, t):
    r_plot = (695700) * r
    dr_vec = r_plot[1:] - r_plot[:-1]
    dp_vec = p[1:] - p[:-1]
    
    dr_vec = np.array(dr_vec, dtype=np.float32)
    dp_vec = np.array(dp_vec, dtype=np.float32)

    hux_f_res = np.ones((np.shape(f)[0], np.shape(f)[1], np.shape(f)[2]))
    for ii in range(len(t)):
        hux_f_res[:, ii, :] = apply_hux_f_model(f[:, ii, 0], dr_vec, dp_vec).T

    return hux_f_res


def main():
    instance_path = sys.argv[1]
    
    model = MASOutput(instance_path)
    
    vr_model = model['vr']
    
    p = vr_model.phi_coords
    # sin(theta) - (-pi/2, pi/2)
    t = vr_model.theta_coords
    # 30 solar radii to approximately 1 AU
    r = vr_model.r_coords
    # velocity profile 
    f = vr_model.data.squeeze()
    

    print("Velocity matrix shape: ", np.shape(f))
    print("Phi dim: ", np.shape(f)[0])
    print("Theta dim: ", np.shape(f)[1])
    print("Radial dim: ", np.shape(f)[2])
    print("Velocity matrix dtype: ", f.dtype)

    solution = get_hux_f(f, r, p, t)
    
    print(solution.shape)

    np.save(f"{instance_path.split('/')[-2]}-{instance_path.split('/')[-1]}.npy", solution)


if __name__ == "__main__":
    main()
